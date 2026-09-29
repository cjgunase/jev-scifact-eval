# SciFact × Jev Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure Jev (`jev-1.13.0`) on SciFact claim verification (retrieval re-ranking, verdict, and evidence-sentence selection) against official metrics, baselines, and published systems, spending at most $0.90.

**Architecture:** Numbered stage scripts in `scripts/` (`01`–`05`) read and write JSONL under `data/processed/scifact/{split}/`. Shared logic lives in `scripts/lib/`: data loading and state building, a cached, budget-guarded Jev runner, question definitions, the vendored official SciFact metrics, and evaluation statistics. Question wording and thresholds are developed on 100 train claims, then frozen before one held-out run on dev.

**Tech Stack:** Python 3.12 (uv), `typesafe-sdk` 0.7.2, `rank-bm25`, `numpy`, `pandas`, `scikit-learn` (AUROC and the stopword list only), `matplotlib`, `pytest`.

**Spec:** `docs/superpowers/specs/2026-09-29-scifact-jev-evaluation-design.md`

## Global Constraints

- **Budget:** hard cap of **$0.90** cumulative TypeSafe spend, recorded in `data/processed/scifact/spend_ledger.csv`. Price is $0.042 per 1M input tokens; output tokens are free.
- **Model:** pinned to `jev-1.13.0`, never `jev-latest`.
- **API key:** exported only in `~/.zshrc`. Every command that calls the API must run as `zsh -ic 'uv run python …'`. Pure-code steps and tests use plain `uv run …`.
- **Splits:** train = a fixed sample of 100 claims from `claims_train.jsonl` (`numpy.random.default_rng(20260929)`); dev = all 300 claims in `claims_dev.jsonl`, run **once**, after the questions are frozen.
- **Join keys:** `doc_id` is an `int` in `corpus.jsonl` but a `str` key in `claims_*.jsonl` `evidence`. Cast to `int` at load time and never join on the raw strings.
- **Leakage:** a Jev state may contain only `claim` and `abstract` (`title`, `text` or `sentences`). Gold fields never enter a state.
- **Raw data** (`data/raw/scifact/`; claims CC BY 4.0, abstracts ODC-By 1.0) is gitignored and never modified. The Jev cache is gitignored; small processed JSONL and results are committed.
- **Bootstrap:** 1,000 resamples of claims, seed `20260929`, 95% percentile CIs.
- **Plots:** follow the global style in `~/.claude/CLAUDE.md` (white background, no top or right spines, bold labels, the given palette, `dpi=180`, `bbox_inches='tight'`, `tight_layout()`).
- **Commits:** end every commit message with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **Budget exhaustion mid-stage.** A stage that hits the cap must stop cleanly, write no partial output file, and keep every paid response in the cache so that a rerun resumes at $0 for completed calls (Task 3 test `test_budget_exceeded_keeps_cache_and_resumes`).
2. **Very long abstracts** (9 over 40 sentences, max 367). Verify must split the sentence Nouls into several requests and merge them without dropping or shifting sentence indices (Task 6 test `test_verify_merges_chunks_for_long_abstract`).
3. **The int/str doc_id join.** Gold evidence keys must match corpus and prediction IDs. A silent mismatch would make every prediction score as wrong (Task 2 test `test_evidence_keys_are_int_and_in_corpus`; Task 8 test `test_prediction_keys_are_str_ints`).
4. **Transient API failures (429, 5xx, timeouts).** A failed call must release its budget reservation and must not write to the cache. The next run retries it (Task 3 test `test_failed_call_releases_reservation`).
5. **Claims with no predicted evidence** (NEI verdicts, or everything below threshold). These must still produce a valid `{"id": ..., "evidence": {}}` line so that the official scorer counts them (Task 8 test `test_every_claim_gets_a_line`).

---

## File Structure

```
scripts/
  00_pilot.py                 # 5+5 paid calls, projects total cost, gate
  01_bm25_retrieve.py         # BM25 top-30 per claim ($0)
  02_jev_rerank.py            # Score question per (claim, candidate)
  03_jev_verify.py            # verdict Choice + per-sentence Nouls
  04_assemble_predictions.py  # thresholds (train) → SciFact predictions
  05_evaluate.py              # metrics, CIs, baselines, calibration, figures
  fetch_scifact.py            # download + checksum
  lib/
    __init__.py
    data.py                   # corpus/claims loading, splits, states, leak guard, jsonl io
    budget.py                 # ledger + hard cap
    jev_client.py             # cache + budget-guarded async runner, SDK adapter
    questions.py              # all Jev questions, QUESTION_VERSION
    official.py               # adapter onto vendored SciFact metrics
    evalstats.py              # bootstrap, retrieval, ECE, selective prediction
    baselines.py              # majority + lexical baselines
    scifact_official/         # vendored, unmodified allenai/scifact @ 68b98a56
      __init__.py
      data.py
      metrics.py
tests/
  test_official.py  test_data.py  test_budget_client.py  test_questions.py
  test_bm25.py  test_rerank_verify.py  test_assemble.py  test_evalstats.py
results/tables/published_scifact.csv   # hand-transcribed from the papers, with sources
docs/scifact_jev_evaluation_methods.md
```

---

### Task 1: Scaffolding, data fetch, and vendored official metrics

**Files:**
- Modify: `pyproject.toml`, `.gitignore`
- Create: `scripts/fetch_scifact.py`, `scripts/lib/__init__.py`, `scripts/lib/scifact_official/{__init__,data,metrics}.py`, `scripts/lib/official.py`, `tests/test_official.py`

**Interfaces:**
- Produces: `lib.official.claim_counts(pred: dict, gold_evidence: dict[int, tuple[str, list[list[int]]]]) -> collections.Counter` and `lib.official.metrics_from_counts(total: Counter) -> dict[str, dict[str, float]]`, with keys `sentence_selection`, `sentence_label`, `abstract_label_only`, `abstract_rationalized`, each holding `{precision, recall, f1}`.

- [ ] **Step 1: Add dependencies and pytest config**

```bash
cd /Users/gunaseka/projects/jev-test
uv add rank-bm25 scikit-learn
uv add --dev pytest
```

Append to `pyproject.toml`:

```toml
[tool.pytest.ini_options]
pythonpath = ["scripts"]
testpaths = ["tests"]
```

Append to `.gitignore`:

```
# Jev response cache (large, reproducible from ledgered calls)
data/processed/scifact/jev_cache/
```

- [ ] **Step 2: Write `scripts/fetch_scifact.py`**

```python
"""Download the SciFact release and verify its checksum.

Usage: uv run python scripts/fetch_scifact.py
"""

from __future__ import annotations

import hashlib
import tarfile
import urllib.request
from pathlib import Path

URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
SHA256 = "11c621288d41ac144d29b13b0f8503b3820b7d6e8b1f6ff24dff335c196d76be"
DEST = Path(__file__).resolve().parents[1] / "data" / "raw" / "scifact"


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    tarball = DEST / "data.tar.gz"
    if not tarball.exists():
        urllib.request.urlretrieve(URL, tarball)
    digest = hashlib.sha256(tarball.read_bytes()).hexdigest()
    if digest != SHA256:
        raise SystemExit(f"Checksum mismatch: {digest} != {SHA256}")
    with tarfile.open(tarball) as tf:
        tf.extractall(DEST, filter="data")
    print(f"SciFact OK at {DEST / 'data'}")


if __name__ == "__main__":
    main()
```

Run: `uv run python scripts/fetch_scifact.py`. Expected: `SciFact OK at …/data/raw/scifact/data`.

- [ ] **Step 3: Vendor the official code, unmodified**

```bash
mkdir -p scripts/lib/scifact_official
touch scripts/lib/__init__.py
B=https://raw.githubusercontent.com/allenai/scifact/68b98a56d93e0f9da0d2aab4e6c3294699a0f72e/verisci/evaluate/lib
curl -s $B/metrics.py -o scripts/lib/scifact_official/metrics.py
curl -s $B/data.py -o scripts/lib/scifact_official/data.py
```

Create `scripts/lib/scifact_official/__init__.py`:

```python
"""Vendored, UNMODIFIED copy of the official SciFact evaluation code.

Source: https://github.com/allenai/scifact, commit 68b98a56d93e0f9da0d2aab4e6c3294699a0f72e,
files verisci/evaluate/lib/{metrics,data}.py. License: Apache-2.0 (repository).
Do not edit these files; adapt them in lib/official.py instead.
"""
```

- [ ] **Step 4: Write the failing test (the worked example from SciFact `doc/evaluation.md`)**

`tests/test_official.py`:

```python
from collections import Counter

import pytest

from lib.official import claim_counts, metrics_from_counts

GOLD_52 = {11: ("SUPPORT", [[0, 1], [11]]), 15: ("SUPPORT", [[4]])}
PRED_52 = {
    "id": 52,
    "evidence": {
        "11": {"sentences": [1, 11, 13], "label": "SUPPORT"},
        "16": {"sentences": [18, 20], "label": "CONTRADICT"},
    },
}


def test_worked_example_matches_official_doc():
    m = metrics_from_counts(claim_counts(PRED_52, GOLD_52))
    assert m["abstract_rationalized"]["f1"] == pytest.approx(1 / 2)
    assert m["sentence_label"]["f1"] == pytest.approx(2 / 9)


def test_empty_prediction_scores_zero_but_counts_relevant():
    counts = claim_counts({"id": 52, "evidence": {}}, GOLD_52)
    assert counts["relevant_abstract"] == 2
    assert metrics_from_counts(counts)["abstract_label_only"]["f1"] == 0


def test_counts_add_across_claims():
    total = Counter()
    for _ in range(3):
        total += claim_counts(PRED_52, GOLD_52)
    assert metrics_from_counts(total)["abstract_rationalized"]["f1"] == pytest.approx(1 / 2)
```

- [ ] **Step 5: Run it and confirm it fails**

Run: `uv run pytest tests/test_official.py -v`. Expected: FAIL with `ModuleNotFoundError: No module named 'lib.official'`.

- [ ] **Step 6: Implement `scripts/lib/official.py`**

The official functions update a shared `Counter` with the keys `relevant`, `retrieved`, `correct_*`. Abstract and sentence counts use the same key names, so they are kept in separate Counters and prefixed.

```python
"""Adapter from our plain-dict predictions and gold data onto the vendored official SciFact metrics."""

from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

from lib.scifact_official.data import EvidenceAbstract, PredictedAbstract, make_label
from lib.scifact_official.metrics import compute_f1, update_counts_abstract, update_counts_sentence


def _gold_obj(gold_evidence: dict[int, tuple[str, list[list[int]]]]) -> SimpleNamespace:
    return SimpleNamespace(evidence={
        int(doc_id): EvidenceAbstract(int(doc_id), make_label(label), rationales)
        for doc_id, (label, rationales) in gold_evidence.items()
    })


def _pred_obj(pred: dict) -> SimpleNamespace:
    return SimpleNamespace(predictions={
        int(doc_id): PredictedAbstract(int(doc_id), make_label(p["label"], allow_NEI=False), list(p["sentences"]))
        for doc_id, p in pred["evidence"].items()
    })


def claim_counts(pred: dict, gold_evidence: dict[int, tuple[str, list[list[int]]]]) -> Counter:
    """Official counts for one claim, with keys prefixed `_abstract` / `_sentence`."""
    gold, p = _gold_obj(gold_evidence), _pred_obj(pred)
    a = update_counts_abstract(p, gold, Counter())
    s = update_counts_sentence(p, gold, Counter())
    out = Counter({f"{k}_abstract": v for k, v in a.items()})
    out.update({f"{k}_sentence": v for k, v in s.items()})
    # Make sure the zero-valued keys exist so that downstream code can read them.
    for k in ("relevant_abstract", "retrieved_abstract", "relevant_sentence", "retrieved_sentence"):
        out.setdefault(k, 0)
    return out


def metrics_from_counts(total: Counter) -> dict[str, dict[str, float]]:
    a = Counter({k.removesuffix("_abstract"): v for k, v in total.items() if k.endswith("_abstract")})
    s = Counter({k.removesuffix("_sentence"): v for k, v in total.items() if k.endswith("_sentence")})
    return {
        "sentence_selection": compute_f1(s, "selection"),
        "sentence_label": compute_f1(s, "label"),
        "abstract_label_only": compute_f1(a, "label_only"),
        "abstract_rationalized": compute_f1(a, "rationalized"),
    }
```

- [ ] **Step 7: Run the tests and confirm they pass**

Run: `uv run pytest tests/test_official.py -v`. Expected: 3 passed.

- [ ] **Step 8: Commit**

```bash
git add pyproject.toml uv.lock .gitignore scripts/fetch_scifact.py scripts/lib tests/test_official.py
git commit -m "Vendor official SciFact metrics and verify them against the documented worked example

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Data loading, splits, states, and leak guard

**Files:**
- Create: `scripts/lib/data.py`, `tests/test_data.py`

**Interfaces:**
- Produces:
  - `Doc(doc_id: int, title: str, sentences: tuple[str, ...])`
  - `Claim(claim_id: int, text: str, evidence: dict[int, tuple[str, list[list[int]]]])`, where evidence maps a doc to `(label, rationales)`
  - `load_corpus() -> dict[int, Doc]`
  - `load_split(split: str) -> list[Claim]`, where split is `"train"` or `"dev"`
  - `rerank_state(claim, doc) -> dict`
  - `verify_state(claim, doc) -> dict`
  - `read_jsonl(path) -> list[dict]` and `write_jsonl(path, rows)`
  - `split_dir(split) -> Path`
  - Constants: `ROOT`, `RAW_DIR`, `PROCESSED_DIR`, `SEED = 20260929`, `N_TRAIN = 100`

- [ ] **Step 1: Write the failing tests**

`tests/test_data.py`:

```python
import json

from lib.data import N_TRAIN, load_corpus, load_split, rerank_state, verify_state


def test_split_sizes_and_determinism():
    train = load_split("train")
    assert len(train) == N_TRAIN
    assert [c.claim_id for c in train] == [c.claim_id for c in load_split("train")]
    assert len(load_split("dev")) == 300


def test_evidence_keys_are_int_and_in_corpus():
    corpus = load_corpus()
    for split in ("train", "dev"):
        for c in load_split(split):
            for doc_id, (label, rationales) in c.evidence.items():
                assert isinstance(doc_id, int) and doc_id in corpus
                assert label in {"SUPPORT", "CONTRADICT"}
                assert all(0 <= i < len(corpus[doc_id].sentences) for r in rationales for i in r)


def test_states_contain_no_gold_fields():
    corpus = load_corpus()
    claim = next(c for c in load_split("dev") if c.evidence)
    doc = corpus[next(iter(claim.evidence))]
    for state in (rerank_state(claim, doc), verify_state(claim, doc)):
        assert set(state) == {"claim", "abstract"}
        blob = json.dumps(state)
        for forbidden in ('"evidence"', '"label"', '"rationales"', '"cited_doc_ids"', '"doc_id"'):
            assert forbidden not in blob


def test_verify_state_indexes_every_sentence():
    corpus = load_corpus()
    claim = load_split("dev")[0]
    doc = max(corpus.values(), key=lambda d: len(d.sentences))
    s = verify_state(claim, doc)["abstract"]["sentences"]
    assert list(s) == [f"s{i}" for i in range(len(doc.sentences))]
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `uv run pytest tests/test_data.py -v`. Expected: FAIL with `ModuleNotFoundError: No module named 'lib.data'`.

- [ ] **Step 3: Implement `scripts/lib/data.py`**

```python
"""SciFact loading, fixed splits, Jev state construction, and JSONL helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "scifact" / "data"
PROCESSED_DIR = ROOT / "data" / "processed" / "scifact"
SEED = 20260929
N_TRAIN = 100

_ALLOWED_STATE_KEYS = {"claim", "abstract"}
_ALLOWED_ABSTRACT_KEYS = {"title", "text", "sentences"}


@dataclass(frozen=True)
class Doc:
    doc_id: int
    title: str
    sentences: tuple[str, ...]


@dataclass(frozen=True)
class Claim:
    claim_id: int
    text: str
    evidence: dict  # int doc_id -> (label, [[sentence idx, ...], ...])


def read_jsonl(path: Path) -> list[dict]:
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    tmp.replace(path)


def split_dir(split: str) -> Path:
    return PROCESSED_DIR / split


def load_corpus() -> dict[int, Doc]:
    return {
        int(r["doc_id"]): Doc(int(r["doc_id"]), r["title"], tuple(r["abstract"]))
        for r in read_jsonl(RAW_DIR / "corpus.jsonl")
    }


def _parse_claim(r: dict) -> Claim:
    evidence = {}
    for doc_key, sets in r.get("evidence", {}).items():
        labels = {s["label"] for s in sets}
        if len(labels) != 1:
            raise ValueError(f"claim {r['id']} doc {doc_key}: mixed labels {labels}")
        evidence[int(doc_key)] = (labels.pop(), [list(s["sentences"]) for s in sets])
    return Claim(int(r["id"]), r["claim"], evidence)


def load_split(split: str) -> list[Claim]:
    if split == "dev":
        return [_parse_claim(r) for r in read_jsonl(RAW_DIR / "claims_dev.jsonl")]
    if split == "train":
        rows = read_jsonl(RAW_DIR / "claims_train.jsonl")
        idx = np.random.default_rng(SEED).choice(len(rows), size=N_TRAIN, replace=False)
        return sorted((_parse_claim(rows[i]) for i in idx), key=lambda c: c.claim_id)
    raise ValueError(f"unknown split {split!r}")


def _check(state: dict) -> dict:
    if set(state) != _ALLOWED_STATE_KEYS or not set(state["abstract"]) <= _ALLOWED_ABSTRACT_KEYS:
        raise AssertionError(f"state has disallowed keys: {state.keys()}")
    return state


def rerank_state(claim: Claim, doc: Doc) -> dict:
    return _check({"claim": claim.text, "abstract": {"title": doc.title, "text": " ".join(doc.sentences)}})


def verify_state(claim: Claim, doc: Doc) -> dict:
    sentences = {f"s{i}": s for i, s in enumerate(doc.sentences)}
    return _check({"claim": claim.text, "abstract": {"title": doc.title, "sentences": sentences}})
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `uv run pytest tests/test_data.py -v`. Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/lib/data.py tests/test_data.py
git commit -m "Add SciFact loaders with fixed train sample, int doc keys, and leak-guarded Jev states

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Budget ledger and cached Jev runner

**Files:**
- Create: `scripts/lib/budget.py`, `scripts/lib/jev_client.py`, `tests/test_budget_client.py`

**Interfaces:**
- Consumes: `lib.data.PROCESSED_DIR`
- Produces:
  - `Budget(ledger_path: Path, cap_usd: float = BUDGET_CAP_USD)`, with `.spent_usd`, `.reserve(est_tokens) -> float`, `.settle(reserved, input_tokens, stage, split, question_version)` and `.release(reserved)`
  - `BudgetExceeded`, `estimate_tokens(payload: str) -> int`
  - Constants: `PRICE_PER_TOKEN_USD`, `BUDGET_CAP_USD = 0.90`, `LEDGER_PATH`
  - `JevRunner(budget, call, cache_dir=CACHE_DIR, model=MODEL_ID, max_concurrency=16)`, with `async .ask(state, questions, *, stage, split, question_version) -> dict {"answers": dict, "input_tokens": int, "cached": bool}`
  - `make_sdk_call(client) -> Callable`, `run_with_budget(coro) -> Any` (exits with code 2 on `BudgetExceeded`), `MODEL_ID = "jev-1.13.0"`

- [ ] **Step 1: Write the failing tests**

`tests/test_budget_client.py`:

```python
import asyncio
import csv

import pytest

from lib.budget import Budget, BudgetExceeded
from lib.jev_client import JevRunner

STATE = {"claim": "c", "abstract": {"title": "t", "text": "x"}}
QS = {"q": {"type": "noul", "instructions": "Is it?"}}


def fake_call(tokens=1000, fail=False):
    calls = []

    async def call(state, questions, model):
        calls.append((state, questions))
        if fail:
            raise RuntimeError("503")
        return {"model": model, "answers": {"q": {"type": "noul", "noul": 0.7}}, "usage": {"input_tokens": tokens}}

    return call, calls


def run(coro):
    return asyncio.run(coro)


def test_cache_hit_does_not_call_or_charge(tmp_path):
    budget = Budget(tmp_path / "ledger.csv")
    call, calls = fake_call()
    r = JevRunner(budget, call, cache_dir=tmp_path / "cache")
    kw = dict(stage="t", split="train", question_version="v1")
    first = run(r.ask(STATE, QS, **kw))
    second = run(r.ask(STATE, QS, **kw))
    assert len(calls) == 1 and not first["cached"] and second["cached"]
    assert second["answers"] == first["answers"]
    assert budget.spent_usd == pytest.approx(1000 * 0.042 / 1e6)


def test_ledger_persists_total_across_instances(tmp_path):
    ledger = tmp_path / "ledger.csv"
    call, _ = fake_call(tokens=2_000_000)
    run(JevRunner(Budget(ledger), call, cache_dir=tmp_path / "c").ask(STATE, QS, stage="t", split="train", question_version="v1"))
    assert Budget(ledger).spent_usd == pytest.approx(0.084)
    rows = list(csv.DictReader(open(ledger)))
    assert rows[0]["input_tokens"] == "2000000" and rows[0]["question_version"] == "v1"


def test_budget_refuses_call_that_would_exceed_cap(tmp_path):
    budget = Budget(tmp_path / "ledger.csv", cap_usd=1e-9)
    call, calls = fake_call()
    with pytest.raises(BudgetExceeded):
        run(JevRunner(budget, call, cache_dir=tmp_path / "c").ask(STATE, QS, stage="t", split="train", question_version="v1"))
    assert calls == []


def test_budget_exceeded_keeps_cache_and_resumes(tmp_path):
    ledger, cache = tmp_path / "ledger.csv", tmp_path / "c"
    call, calls = fake_call(tokens=1000)
    kw = dict(stage="t", split="train", question_version="v1")
    run(JevRunner(Budget(ledger), call, cache_dir=cache).ask(STATE, QS, **kw))
    tight = Budget(ledger, cap_usd=Budget(ledger).spent_usd)  # no headroom left
    again = run(JevRunner(tight, call, cache_dir=cache).ask(STATE, QS, **kw))
    assert again["cached"] and len(calls) == 1
    with pytest.raises(BudgetExceeded):
        run(JevRunner(tight, call, cache_dir=cache).ask({**STATE, "claim": "new"}, QS, **kw))


def test_failed_call_releases_reservation(tmp_path):
    budget = Budget(tmp_path / "ledger.csv")
    call, _ = fake_call(fail=True)
    r = JevRunner(budget, call, cache_dir=tmp_path / "c")
    with pytest.raises(RuntimeError):
        run(r.ask(STATE, QS, stage="t", split="train", question_version="v1"))
    assert budget._reserved_usd == 0 and budget.spent_usd == 0
    assert not any((tmp_path / "c").glob("*.json"))
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `uv run pytest tests/test_budget_client.py -v`. Expected: FAIL with `ModuleNotFoundError: No module named 'lib.budget'`.

- [ ] **Step 3: Implement `scripts/lib/budget.py`**

```python
"""Persistent spend ledger with a hard USD cap on TypeSafe calls."""

from __future__ import annotations

import csv
import math
from datetime import datetime, timezone
from pathlib import Path

from lib.data import PROCESSED_DIR

PRICE_PER_TOKEN_USD = 0.042 / 1_000_000  # jev-1.13.0, input tokens only; output is free
BUDGET_CAP_USD = 0.90                    # user limit is $1.00; keep a margin
LEDGER_PATH = PROCESSED_DIR / "spend_ledger.csv"
TOKEN_SAFETY_FACTOR = 1.5                # pre-call estimate is deliberately pessimistic
FIELDS = ["timestamp", "stage", "split", "question_version", "input_tokens", "cost_usd", "total_usd"]


class BudgetExceeded(RuntimeError):
    pass


def estimate_tokens(payload: str) -> int:
    return math.ceil(len(payload.encode()) / 4 * TOKEN_SAFETY_FACTOR)


class Budget:
    def __init__(self, ledger_path: Path = LEDGER_PATH, cap_usd: float = BUDGET_CAP_USD):
        self.ledger_path = Path(ledger_path)
        self.cap_usd = cap_usd
        self._reserved_usd = 0.0
        self.spent_usd = 0.0
        if self.ledger_path.exists():
            with open(self.ledger_path) as fh:
                self.spent_usd = sum(float(r["cost_usd"]) for r in csv.DictReader(fh))

    def reserve(self, est_tokens: int) -> float:
        cost = est_tokens * PRICE_PER_TOKEN_USD
        if self.spent_usd + self._reserved_usd + cost > self.cap_usd:
            raise BudgetExceeded(
                f"spent ${self.spent_usd:.4f} + reserved ${self._reserved_usd:.4f} + next ${cost:.6f} "
                f"> cap ${self.cap_usd:.2f}"
            )
        self._reserved_usd += cost
        return cost

    def release(self, reserved: float) -> None:
        self._reserved_usd = max(0.0, self._reserved_usd - reserved)

    def settle(self, reserved: float, input_tokens: int, stage: str, split: str, question_version: str) -> None:
        self.release(reserved)
        cost = input_tokens * PRICE_PER_TOKEN_USD
        self.spent_usd += cost
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        new = not self.ledger_path.exists()
        with open(self.ledger_path, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            if new:
                w.writeheader()
            w.writerow({
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "stage": stage, "split": split, "question_version": question_version,
                "input_tokens": input_tokens, "cost_usd": f"{cost:.8f}", "total_usd": f"{self.spent_usd:.8f}",
            })
```

- [ ] **Step 4: Implement `scripts/lib/jev_client.py`**

The reservation happens **inside** the semaphore. Otherwise thousands of queued tasks would reserve at once and trip the cap spuriously.

```python
"""Cached, budget-guarded async access to Jev."""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Awaitable, Callable

from lib.budget import Budget, BudgetExceeded, estimate_tokens
from lib.data import PROCESSED_DIR

MODEL_ID = "jev-1.13.0"
CACHE_DIR = PROCESSED_DIR / "jev_cache"
Call = Callable[[dict, dict, str], Awaitable[dict]]


def cache_key(model: str, state: dict, questions: dict) -> str:
    blob = json.dumps({"model": model, "state": state, "questions": questions}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


class JevRunner:
    def __init__(self, budget: Budget, call: Call, cache_dir: Path = CACHE_DIR,
                 model: str = MODEL_ID, max_concurrency: int = 16):
        self.budget, self.call, self.model = budget, call, model
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._sem = asyncio.Semaphore(max_concurrency)
        self._stopped = False

    async def ask(self, state: dict, questions: dict, *, stage: str, split: str, question_version: str) -> dict:
        path = self.cache_dir / f"{cache_key(self.model, state, questions)}.json"
        if path.exists():
            data = json.loads(path.read_text())
            return {"answers": data["answers"], "input_tokens": data["usage"]["input_tokens"], "cached": True}
        async with self._sem:
            if self._stopped:
                raise BudgetExceeded("runner stopped after an earlier budget refusal")
            est = estimate_tokens(json.dumps({"state": state, "questions": questions}))
            try:
                reserved = self.budget.reserve(est)
            except BudgetExceeded:
                self._stopped = True
                raise
            try:
                resp = await self.call(state, questions, self.model)
            except BaseException:
                self.budget.release(reserved)
                raise
            tokens = int(resp["usage"]["input_tokens"])
            self.budget.settle(reserved, tokens, stage, split, question_version)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(resp))
            tmp.replace(path)
            return {"answers": resp["answers"], "input_tokens": tokens, "cached": False}


def make_sdk_call(client) -> Call:
    """Wrap an AsyncTypeSafeClient so that it takes plain-dict questions and returns plain dicts."""
    from typesafe_sdk import Choice, Noul, Score

    kinds = {"noul": Noul, "choice": Choice, "score": Score}

    async def call(state: dict, questions: dict, model: str) -> dict:
        sdk_qs = {qid: kinds[q["type"]](**{k: v for k, v in q.items() if k != "type"}) for qid, q in questions.items()}
        resp = await client.system_one(state=state, questions=sdk_qs, model=model)
        return {
            "model": resp.model,
            "answers": {qid: a.model_dump() for qid, a in resp.answers.items()},
            "usage": {"input_tokens": resp.usage.input_tokens},
        }

    return call


async def gather_or_stop(coros: list[Awaitable[Any]]) -> list[Any]:
    """Run all coroutines. If any raised BudgetExceeded, re-raise it after the others finish."""
    results = await asyncio.gather(*coros, return_exceptions=True)
    for r in results:
        if isinstance(r, BudgetExceeded):
            raise r
    for r in results:
        if isinstance(r, BaseException):
            raise r
    return results


def run_with_budget(main_coro) -> Any:
    try:
        return asyncio.run(main_coro)
    except BudgetExceeded as e:
        print(f"STOPPED: budget guard refused a call: {e}\nCompleted calls are cached; no output file written.",
              file=sys.stderr)
        sys.exit(2)
```

- [ ] **Step 5: Run the tests and confirm they pass**

Run: `uv run pytest tests/test_budget_client.py -v`. Expected: 5 passed.

- [ ] **Step 6: Commit**

```bash
git add scripts/lib/budget.py scripts/lib/jev_client.py tests/test_budget_client.py
git commit -m "Add persistent spend ledger with \$0.90 hard cap and a cached Jev runner

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Question definitions

**Files:**
- Create: `scripts/lib/questions.py`, `tests/test_questions.py`

**Interfaces:**
- Produces:
  - `QUESTION_VERSION: str`
  - `relevance_questions() -> dict`, whose question id is `relevance` (Score, 4 levels)
  - `verify_question_chunks(n_sentences: int) -> list[dict]`: the first chunk contains `verdict` (Choice with options `supports`, `contradicts`, `not_enough_info`), and every chunk contains `evidence_s{i}` Nouls
  - `MAX_SENTENCE_QUESTIONS_PER_REQUEST = 40`
  - `VERDICT_TO_SCIFACT = {"supports": "SUPPORT", "contradicts": "CONTRADICT"}`

- [ ] **Step 1: Write the failing tests**

`tests/test_questions.py`:

```python
from lib.questions import (MAX_SENTENCE_QUESTIONS_PER_REQUEST, QUESTION_VERSION, relevance_questions,
                           verify_question_chunks)


def test_relevance_is_single_ordered_score():
    q = relevance_questions()
    assert list(q) == ["relevance"] and q["relevance"]["type"] == "score"
    assert 2 <= len(q["relevance"]["criteria"]) <= 10


def test_short_abstract_is_one_chunk_with_verdict():
    chunks = verify_question_chunks(8)
    assert len(chunks) == 1
    assert chunks[0]["verdict"]["type"] == "choice"
    assert set(chunks[0]["verdict"]["criteria"]) == {"supports", "contradicts", "not_enough_info"}
    assert [k for k in chunks[0] if k.startswith("evidence_")] == [f"evidence_s{i}" for i in range(8)]


def test_long_abstract_chunks_cover_every_sentence_once():
    n = 367
    chunks = verify_question_chunks(n)
    ids = [k for c in chunks for k in c if k.startswith("evidence_")]
    assert ids == [f"evidence_s{i}" for i in range(n)]
    assert all(sum(k.startswith("evidence_") for k in c) <= MAX_SENTENCE_QUESTIONS_PER_REQUEST for c in chunks)
    assert sum("verdict" in c for c in chunks) == 1 and "verdict" in chunks[0]


def test_evidence_question_points_at_its_sentence():
    q = verify_question_chunks(3)[0]["evidence_s2"]
    assert q["type"] == "noul" and "`abstract.sentences.s2`" in q["instructions"]


def test_version_is_set():
    assert QUESTION_VERSION.startswith("v")
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `uv run pytest tests/test_questions.py -v`. Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `scripts/lib/questions.py`**

```python
"""All Jev questions used in the SciFact evaluation.

Bump QUESTION_VERSION whenever any wording changes. The ledger and outputs record it,
and the dev split may only be run with the frozen version (see Task 9).
"""

from __future__ import annotations

QUESTION_VERSION = "v1"
MAX_SENTENCE_QUESTIONS_PER_REQUEST = 40
VERDICT_TO_SCIFACT = {"supports": "SUPPORT", "contradicts": "CONTRADICT"}

RELEVANCE_LEVELS = [
    "The abstract is about a different topic from the claim.",
    "The abstract is on the same general topic, but it does not study what the claim asserts.",
    "The abstract studies what the claim is about, but its reported findings neither confirm nor refute the claim.",
    "The abstract reports findings that directly confirm or refute the claim.",
]

VERDICT_CRITERIA = {
    "supports": "The abstract reports findings showing that the claim is true.",
    "contradicts": (
        "The abstract reports findings showing that the claim is false, including an effect in the "
        "opposite direction to the claim or no effect where the claim asserts one."
    ),
    "not_enough_info": (
        "The abstract does not report a finding that shows whether the claim is true or false, "
        "including when it is on the same topic but studies something different."
    ),
}


def relevance_questions() -> dict:
    return {
        "relevance": {
            "type": "score",
            "instructions": "How directly does the abstract in `abstract` address the scientific claim in `claim`?",
            "criteria": RELEVANCE_LEVELS,
        }
    }


def _verdict_question() -> dict:
    return {
        "type": "choice",
        "instructions": "Based only on `abstract`, does it support or contradict the scientific claim in `claim`?",
        "criteria": VERDICT_CRITERIA,
    }


def _evidence_question(i: int) -> dict:
    return {
        "type": "noul",
        "instructions": (
            f"Does sentence `abstract.sentences.s{i}` report a result or finding that, on its own or together "
            "with neighbouring sentences, helps show whether the claim in `claim` is true or false?"
        ),
    }


def verify_question_chunks(n_sentences: int) -> list[dict]:
    chunks = []
    for start in range(0, max(n_sentences, 1), MAX_SENTENCE_QUESTIONS_PER_REQUEST):
        chunk = {"verdict": _verdict_question()} if start == 0 else {}
        for i in range(start, min(start + MAX_SENTENCE_QUESTIONS_PER_REQUEST, n_sentences)):
            chunk[f"evidence_s{i}"] = _evidence_question(i)
        chunks.append(chunk)
    return chunks
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `uv run pytest tests/test_questions.py -v`. Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/lib/questions.py tests/test_questions.py
git commit -m "Define v1 Jev questions: relevance Score, verdict Choice, per-sentence evidence Nouls

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: BM25 retrieval and retrieval metrics

**Files:**
- Create: `scripts/01_bm25_retrieve.py`, `scripts/lib/evalstats.py` (retrieval part), `tests/test_bm25.py`

**Interfaces:**
- Consumes: `lib.data.load_corpus`, `load_split`, `split_dir`, `write_jsonl`
- Produces:
  - `data/processed/scifact/{split}/bm25_top30.jsonl` with rows `{"claim_id": int, "doc_ids": [int]*30, "scores": [float]*30}`
  - `lib.evalstats.tokenize(text) -> list[str]`
  - `recall_at_k(ranked: dict[int, list[int]], gold: dict[int, set[int]], k: int) -> float`
  - `mrr(ranked, gold) -> float`

- [ ] **Step 1: Write the failing tests**

`tests/test_bm25.py`:

```python
import importlib

import pytest

from lib.evalstats import mrr, recall_at_k, tokenize

bm25_mod = importlib.import_module("01_bm25_retrieve")


def test_tokenize_lowercases_and_drops_stopwords():
    assert tokenize("The p53 protein IS mutated in Cancer.") == ["p53", "protein", "mutated", "cancer"]


def test_retrieve_ranks_matching_doc_first():
    from lib.data import Doc
    corpus = {1: Doc(1, "Aspirin and stroke", ("Aspirin lowers stroke risk.",)),
              2: Doc(2, "Coffee", ("Coffee and sleep.",)),
              3: Doc(3, "Tea", ("Tea polyphenols.",))}
    index = bm25_mod.build_index(corpus)
    ids, _ = bm25_mod.retrieve(index, "does aspirin reduce stroke", k=2)
    assert ids[0] == 1 and len(ids) == 2


def test_recall_and_mrr_micro_over_gold_pairs():
    ranked = {1: [5, 6, 7], 2: [9, 8, 4]}
    gold = {1: {6}, 2: {4, 10}}
    assert recall_at_k(ranked, gold, 1) == pytest.approx(0 / 3)
    assert recall_at_k(ranked, gold, 3) == pytest.approx(2 / 3)
    assert mrr(ranked, gold) == pytest.approx((1 / 2 + 1 / 3) / 2)
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `uv run pytest tests/test_bm25.py -v`. Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Create `scripts/lib/evalstats.py` with the retrieval section**

```python
"""Evaluation statistics: tokenisation, retrieval metrics, bootstrap, calibration."""

from __future__ import annotations

import re

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from lib.data import SEED


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in ENGLISH_STOP_WORDS]


def recall_at_k(ranked: dict[int, list[int]], gold: dict[int, set[int]], k: int) -> float:
    """Micro recall over (claim, gold abstract) pairs. Claims without gold are ignored."""
    hits = total = 0
    for cid, g in gold.items():
        if not g:
            continue
        top = set(ranked.get(cid, [])[:k])
        hits += len(g & top)
        total += len(g)
    return hits / total if total else float("nan")


def mrr(ranked: dict[int, list[int]], gold: dict[int, set[int]]) -> float:
    """Mean reciprocal rank of the first gold abstract per claim (0 if none were retrieved)."""
    rr = []
    for cid, g in gold.items():
        if not g:
            continue
        ranks = [i for i, d in enumerate(ranked.get(cid, []), start=1) if d in g]
        rr.append(1 / ranks[0] if ranks else 0.0)
    return float(np.mean(rr)) if rr else float("nan")
```

- [ ] **Step 4: Implement `scripts/01_bm25_retrieve.py`**

```python
"""Stage 1: BM25 top-k abstracts per claim (no API cost).

Usage: uv run python scripts/01_bm25_retrieve.py --split {train,dev}
"""

from __future__ import annotations

import argparse

import numpy as np
from rank_bm25 import BM25Okapi

from lib.data import load_corpus, load_split, split_dir, write_jsonl
from lib.evalstats import recall_at_k, tokenize

TOP_K = 30


def build_index(corpus):
    doc_ids = sorted(corpus)
    bm25 = BM25Okapi([tokenize(corpus[d].title + " " + " ".join(corpus[d].sentences)) for d in doc_ids])
    return bm25, doc_ids


def retrieve(index, query: str, k: int = TOP_K):
    bm25, doc_ids = index
    scores = bm25.get_scores(tokenize(query))
    order = np.argsort(-scores, kind="stable")[:k]
    return [doc_ids[i] for i in order], [float(scores[i]) for i in order]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "dev"], required=True)
    args = ap.parse_args()
    corpus, claims = load_corpus(), load_split(args.split)
    index = build_index(corpus)
    rows = []
    for c in claims:
        ids, scores = retrieve(index, c.text)
        rows.append({"claim_id": c.claim_id, "doc_ids": ids, "scores": scores})
    write_jsonl(split_dir(args.split) / "bm25_top30.jsonl", rows)
    ranked = {r["claim_id"]: r["doc_ids"] for r in rows}
    gold = {c.claim_id: set(c.evidence) for c in claims}
    print(" ".join(f"R@{k}={recall_at_k(ranked, gold, k):.3f}" for k in (1, 3, 10, 30)))


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the tests and confirm they pass**

Run: `uv run pytest tests/test_bm25.py -v`. Expected: 3 passed.

- [ ] **Step 6: Run retrieval on both splits ($0)**

Run: `uv run python scripts/01_bm25_retrieve.py --split train && uv run python scripts/01_bm25_retrieve.py --split dev`
Expected: one `R@1=… R@3=… R@10=… R@30=…` line per split. Record both lines in the commit message. If dev R@30 < 0.70, stop and report: the pipeline's ceiling would be too low for the plan's conclusions.

- [ ] **Step 7: Commit**

```bash
git add scripts/01_bm25_retrieve.py scripts/lib/evalstats.py tests/test_bm25.py data/processed/scifact/*/bm25_top30.jsonl
git commit -m "Add BM25 retrieval stage; train <paste R@k line>, dev <paste R@k line>

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Jev re-rank and verify stages

**Files:**
- Create: `scripts/02_jev_rerank.py`, `scripts/03_jev_verify.py`, `tests/test_rerank_verify.py`

**Interfaces:**
- Consumes: `JevRunner.ask`, `relevance_questions`, `verify_question_chunks`, `rerank_state`, `verify_state`, `bm25_top30.jsonl`
- Produces:
  - `02_jev_rerank.rerank_claim(runner, claim, corpus, bm25_row, split) -> dict`, written to `rerank.jsonl` as rows `{"claim_id", "doc_ids" (reranked), "relevance" (Score expectation per doc, aligned), "bm25_rank" (aligned), "question_version"}`
  - `03_jev_verify.select_pairs(rerank_rows, claims, top_n=3) -> list[tuple[int, int, bool]]`, giving `(claim_id, doc_id, is_oracle)`
  - `03_jev_verify.verify_pair(runner, claim, doc, split) -> dict`, written to `verify.jsonl` as rows `{"claim_id", "doc_id", "is_oracle", "rerank_position" (int or null), "verdict", "verdict_confidence", "verdict_probs", "evidence_probs" (list[float], length = number of sentences), "question_version"}`

- [ ] **Step 1: Write the failing tests** (a fake runner; no API)

`tests/test_rerank_verify.py`:

```python
import asyncio
import importlib

from lib.data import Claim, Doc

rerank_mod = importlib.import_module("02_jev_rerank")
verify_mod = importlib.import_module("03_jev_verify")


class FakeRunner:
    def __init__(self, relevance_by_doc=None):
        self.relevance_by_doc = relevance_by_doc or {}
        self.requests = []

    async def ask(self, state, questions, **kw):
        self.requests.append(questions)
        answers = {}
        for qid, q in questions.items():
            if qid == "relevance":
                answers[qid] = {"type": "score", "score": self.relevance_by_doc[state["abstract"]["title"]],
                                "confidence": 0.9, "probabilities": {"0": 0.1, "1": 0.2, "2": 0.3, "3": 0.4}}
            elif qid == "verdict":
                answers[qid] = {"type": "choice", "choice": "supports", "confidence": 0.8,
                                "probabilities": {"supports": 0.9, "contradicts": 0.05, "not_enough_info": 0.05}}
            else:
                answers[qid] = {"type": "noul", "noul": int(qid.removeprefix("evidence_s")) / 1000}
        return {"answers": answers, "input_tokens": 10, "cached": False}


def test_rerank_orders_by_expectation_then_bm25_rank():
    corpus = {i: Doc(i, f"d{i}", ("x",)) for i in (1, 2, 3)}
    runner = FakeRunner({"d1": 1.0, "d2": 2.5, "d3": 2.5})
    row = asyncio.run(rerank_mod.rerank_claim(runner, Claim(7, "c", {}), corpus,
                                              {"claim_id": 7, "doc_ids": [1, 3, 2], "scores": [3, 2, 1]}, "train"))
    assert row["doc_ids"] == [3, 2, 1]  # tie between 3 and 2 broken by BM25 rank (3 came first)
    assert row["bm25_rank"] == [1, 2, 0]


def test_select_pairs_adds_gold_outside_top3_as_oracle():
    rerank_rows = [{"claim_id": 1, "doc_ids": [10, 11, 12, 13, 14]}]
    claims = [Claim(1, "c", {13: ("SUPPORT", [[0]]), 99: ("CONTRADICT", [[1]])})]
    pairs = verify_mod.select_pairs(rerank_rows, claims, top_n=3)
    assert pairs == [(1, 10, False), (1, 11, False), (1, 12, False), (1, 13, True), (1, 99, True)]


def test_verify_merges_chunks_for_long_abstract():
    n = 95
    doc = Doc(5, "long", tuple(f"sentence {i}" for i in range(n)))
    runner = FakeRunner()
    row = asyncio.run(verify_mod.verify_pair(runner, Claim(1, "c", {}), doc, "train"))
    assert len(runner.requests) == 3  # 40 + 40 + 15
    assert row["evidence_probs"] == [i / 1000 for i in range(n)]
    assert row["verdict"] == "supports" and row["verdict_confidence"] == 0.8
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `uv run pytest tests/test_rerank_verify.py -v`. Expected: FAIL with `ModuleNotFoundError: No module named '02_jev_rerank'`.

- [ ] **Step 3: Implement `scripts/02_jev_rerank.py`**

```python
"""Stage 2: re-rank BM25 candidates with one Jev Score question per (claim, abstract).

Usage: zsh -ic 'uv run python scripts/02_jev_rerank.py --split {train,dev}'
"""

from __future__ import annotations

import argparse
import asyncio

from lib.budget import Budget
from lib.data import load_corpus, load_split, read_jsonl, rerank_state, split_dir, write_jsonl
from lib.jev_client import JevRunner, gather_or_stop, make_sdk_call, run_with_budget
from lib.questions import QUESTION_VERSION, relevance_questions


async def rerank_claim(runner, claim, corpus, bm25_row, split) -> dict:
    doc_ids = bm25_row["doc_ids"]
    results = await gather_or_stop([
        runner.ask(rerank_state(claim, corpus[d]), relevance_questions(),
                   stage="rerank", split=split, question_version=QUESTION_VERSION)
        for d in doc_ids
    ])
    expectations = [r["answers"]["relevance"]["score"] for r in results]
    order = sorted(range(len(doc_ids)), key=lambda i: (-expectations[i], i))
    return {
        "claim_id": claim.claim_id,
        "doc_ids": [doc_ids[i] for i in order],
        "relevance": [expectations[i] for i in order],
        "bm25_rank": order,
        "question_version": QUESTION_VERSION,
    }


async def main_async(split: str) -> None:
    from typesafe_sdk import AsyncTypeSafeClient

    corpus, claims = load_corpus(), {c.claim_id: c for c in load_split(split)}
    bm25_rows = read_jsonl(split_dir(split) / "bm25_top30.jsonl")
    async with AsyncTypeSafeClient() as client:
        runner = JevRunner(Budget(), make_sdk_call(client))
        rows = await gather_or_stop([rerank_claim(runner, claims[r["claim_id"]], corpus, r, split) for r in bm25_rows])
    write_jsonl(split_dir(split) / "rerank.jsonl", rows)
    print(f"reranked {len(rows)} claims; spent so far ${runner.budget.spent_usd:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "dev"], required=True)
    run_with_budget(main_async(ap.parse_args().split))
```

- [ ] **Step 4: Implement `scripts/03_jev_verify.py`**

```python
"""Stage 3: verdict + per-sentence evidence for the top-3 re-ranked abstracts, plus gold (oracle) pairs.

Usage: zsh -ic 'uv run python scripts/03_jev_verify.py --split {train,dev}'
"""

from __future__ import annotations

import argparse

from lib.budget import Budget
from lib.data import load_corpus, load_split, read_jsonl, split_dir, verify_state, write_jsonl
from lib.jev_client import JevRunner, gather_or_stop, make_sdk_call, run_with_budget
from lib.questions import QUESTION_VERSION, verify_question_chunks

TOP_N = 3


def select_pairs(rerank_rows, claims, top_n: int = TOP_N) -> list[tuple[int, int, bool]]:
    by_id = {c.claim_id: c for c in claims}
    pairs = []
    for r in rerank_rows:
        top = r["doc_ids"][:top_n]
        pairs += [(r["claim_id"], d, False) for d in top]
        pairs += [(r["claim_id"], d, True) for d in sorted(by_id[r["claim_id"]].evidence) if d not in top]
    return pairs


async def verify_pair(runner, claim, doc, split) -> dict:
    state = verify_state(claim, doc)
    results = await gather_or_stop([
        runner.ask(state, chunk, stage="verify", split=split, question_version=QUESTION_VERSION)
        for chunk in verify_question_chunks(len(doc.sentences))
    ])
    answers = {k: v for r in results for k, v in r["answers"].items()}
    verdict = answers["verdict"]
    return {
        "claim_id": claim.claim_id,
        "doc_id": doc.doc_id,
        "verdict": verdict["choice"],
        "verdict_confidence": verdict["confidence"],
        "verdict_probs": verdict["probabilities"],
        "evidence_probs": [answers[f"evidence_s{i}"]["noul"] for i in range(len(doc.sentences))],
        "question_version": QUESTION_VERSION,
    }


async def main_async(split: str) -> None:
    from typesafe_sdk import AsyncTypeSafeClient

    corpus, claims = load_corpus(), load_split(split)
    by_id = {c.claim_id: c for c in claims}
    rerank_rows = read_jsonl(split_dir(split) / "rerank.jsonl")
    positions = {(r["claim_id"], d): i for r in rerank_rows for i, d in enumerate(r["doc_ids"])}
    pairs = select_pairs(rerank_rows, claims)
    async with AsyncTypeSafeClient() as client:
        runner = JevRunner(Budget(), make_sdk_call(client))
        rows = await gather_or_stop([verify_pair(runner, by_id[c], corpus[d], split) for c, d, _ in pairs])
    for row, (c, d, is_oracle) in zip(rows, pairs):
        row["is_oracle"] = is_oracle
        row["rerank_position"] = positions.get((c, d))
    write_jsonl(split_dir(split) / "verify.jsonl", rows)
    print(f"verified {len(rows)} pairs ({sum(p[2] for p in pairs)} oracle); spent ${runner.budget.spent_usd:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "dev"], required=True)
    run_with_budget(main_async(ap.parse_args().split))
```

- [ ] **Step 5: Run the tests and confirm they pass**

Run: `uv run pytest tests/test_rerank_verify.py -v`. Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add scripts/02_jev_rerank.py scripts/03_jev_verify.py tests/test_rerank_verify.py
git commit -m "Add Jev re-rank (Score) and verify (Choice + chunked sentence Nouls) stages

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Pilot cost gate (first paid calls, under $0.01)

**Files:**
- Create: `scripts/00_pilot.py`

**Interfaces:**
- Consumes: `rerank_claim`-style calls via `JevRunner`, `verify_pair`, `train/bm25_top30.jsonl`
- Produces: stdout projection, and exit code 0 (proceed) or 3 (over budget)

- [ ] **Step 1: Implement `scripts/00_pilot.py`**

```python
"""Pilot: 5 rerank + 5 verify calls on train, then project the full plan's cost.

Usage: zsh -ic 'uv run python scripts/00_pilot.py'
"""

from __future__ import annotations

import importlib
import statistics
import sys

from lib.budget import BUDGET_CAP_USD, PRICE_PER_TOKEN_USD, Budget
from lib.data import load_corpus, load_split, read_jsonl, rerank_state, split_dir
from lib.jev_client import JevRunner, gather_or_stop, make_sdk_call, run_with_budget
from lib.questions import QUESTION_VERSION, relevance_questions

verify_mod = importlib.import_module("03_jev_verify")

# Planned call volumes (spec §6): dev once, plus train for at most 2 question versions.
N_RERANK = 300 * 30 + 2 * 100 * 30
N_VERIFY = (300 * 3 + 60) + 2 * (100 * 3 + 40)  # top-3 + estimated oracle pairs


async def main_async() -> None:
    from typesafe_sdk import AsyncTypeSafeClient

    corpus, claims = load_corpus(), load_split("train")
    claim = next(c for c in claims if c.evidence)
    bm25 = next(r for r in read_jsonl(split_dir("train") / "bm25_top30.jsonl") if r["claim_id"] == claim.claim_id)
    docs = [corpus[d] for d in bm25["doc_ids"][:5]]
    async with AsyncTypeSafeClient() as client:
        runner = JevRunner(Budget(), make_sdk_call(client))
        before = runner.budget.spent_usd
        rr = await gather_or_stop([runner.ask(rerank_state(claim, d), relevance_questions(), stage="pilot_rerank",
                                              split="train", question_version=QUESTION_VERSION) for d in docs])
        vr = await gather_or_stop([verify_mod.verify_pair(runner, claim, d, "train") for d in docs])
        spent_pilot = runner.budget.spent_usd - before
    rerank_tok = statistics.mean(r["input_tokens"] for r in rr)
    # verify_pair doesn't return tokens; recover them from the ledger delta.
    verify_tok = (spent_pilot / PRICE_PER_TOKEN_USD - sum(r["input_tokens"] for r in rr if not r["cached"])) / len(vr)
    projected = (N_RERANK * rerank_tok + N_VERIFY * verify_tok) * PRICE_PER_TOKEN_USD
    remaining = BUDGET_CAP_USD - runner.budget.spent_usd
    print(f"rerank tokens/call ≈ {rerank_tok:.0f}; verify tokens/pair ≈ {verify_tok:.0f}")
    print(f"projected full plan ${projected:.3f}; remaining under cap ${remaining:.3f}")
    print(f"sample verdict: {vr[0]['verdict']} (conf {vr[0]['verdict_confidence']:.2f}); "
          f"top relevance {max(r['answers']['relevance']['score'] for r in rr):.2f}")
    if projected > remaining:
        print("OVER BUDGET: use the fallback (TOP_K=20 in 01_bm25_retrieve.py, 1 train question version)")
        sys.exit(3)


if __name__ == "__main__":
    run_with_budget(main_async())
```

- [ ] **Step 2: Run the pilot (paid, about $0.001)**

Run: `zsh -ic 'uv run python scripts/00_pilot.py'`
Expected: the tokens-per-call lines and a projection ≤ remaining; exit 0. **If it exits 3:** set `TOP_K = 20` in `01_bm25_retrieve.py`, change `N_RERANK` to `300 * 20 + 1 * 100 * 20` and `N_VERIFY` to one train version, rerun Task 5 Step 6 and this step, and note the fallback in the methods doc. If it still exits 3, stop and report to the user.

- [ ] **Step 3: Commit**

```bash
git add scripts/00_pilot.py data/processed/scifact/spend_ledger.csv
git commit -m "Pilot Jev cost: <paste tokens/call lines>; projected <paste> under \$0.90 cap

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Assemble predictions and select thresholds

**Files:**
- Create: `scripts/04_assemble_predictions.py`, `tests/test_assemble.py`

**Interfaces:**
- Consumes: `verify.jsonl` rows (Task 6), `lib.official.claim_counts`, `metrics_from_counts`, `VERDICT_TO_SCIFACT`
- Produces:
  - `assemble(verify_rows, claim_ids, tau_verdict, tau_sentence) -> list[dict]`, one SciFact prediction line per claim id
  - `score(preds, claims) -> dict` (official metrics)
  - `select_thresholds(verify_rows, claims) -> dict {"tau_verdict", "tau_sentence", "train_metrics"}`
  - Files: `results/tables/thresholds.json` (train) and `data/processed/scifact/{split}/predictions.jsonl`

- [ ] **Step 1: Write the failing tests**

`tests/test_assemble.py`:

```python
import importlib

asm = importlib.import_module("04_assemble_predictions")
from lib.data import Claim


def row(cid, did, verdict, conf, probs, oracle=False):
    return {"claim_id": cid, "doc_id": did, "verdict": verdict, "verdict_confidence": conf,
            "evidence_probs": probs, "is_oracle": oracle}


ROWS = [
    row(1, 10, "supports", 0.9, [0.1, 0.8, 0.6, 0.95]),
    row(1, 11, "not_enough_info", 0.9, [0.9]),
    row(1, 12, "contradicts", 0.3, [0.9]),
    row(1, 13, "supports", 0.99, [0.99], oracle=True),
    row(2, 20, "contradicts", 0.7, [0.2, 0.1]),
]


def test_every_claim_gets_a_line():
    preds = asm.assemble(ROWS, [1, 2, 3], tau_verdict=0.5, tau_sentence=0.5)
    assert [p["id"] for p in preds] == [1, 2, 3]
    assert preds[2] == {"id": 3, "evidence": {}}


def test_filters_nei_low_confidence_and_oracle_and_orders_sentences():
    p1 = asm.assemble(ROWS, [1], tau_verdict=0.5, tau_sentence=0.5)[0]
    assert list(p1["evidence"]) == ["10"]
    assert p1["evidence"]["10"] == {"sentences": [3, 1, 2], "label": "SUPPORT"}


def test_falls_back_to_best_sentence_when_none_pass():
    p2 = asm.assemble(ROWS, [2], tau_verdict=0.5, tau_sentence=0.5)[0]
    assert p2["evidence"]["20"] == {"sentences": [0], "label": "CONTRADICT"}


def test_prediction_keys_are_str_ints():
    for p in asm.assemble(ROWS, [1, 2], 0.0, 0.0):
        assert all(isinstance(k, str) and k.isdigit() for k in p["evidence"])


def test_select_thresholds_prefers_the_grid_point_with_best_f1():
    claims = [Claim(1, "c", {10: ("SUPPORT", [[3]])}), Claim(2, "c", {})]
    out = asm.select_thresholds(ROWS, claims)
    assert out["train_metrics"]["abstract_rationalized"]["f1"] == 1.0
    assert out["tau_verdict"] > 0.7  # must exclude claim 2's wrong 0.7-confidence prediction
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `uv run pytest tests/test_assemble.py -v`. Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `scripts/04_assemble_predictions.py`**

```python
"""Stage 4: turn Jev answers into SciFact predictions; thresholds chosen on train only.

Usage: uv run python scripts/04_assemble_predictions.py --split {train,dev}
  train: grid-searches thresholds, writes results/tables/thresholds.json, then predictions
  dev:   reads thresholds.json (must exist) and writes predictions
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np

from lib.data import ROOT, load_split, read_jsonl, split_dir, write_jsonl
from lib.official import claim_counts, metrics_from_counts
from lib.questions import QUESTION_VERSION, VERDICT_TO_SCIFACT

THRESHOLDS_PATH = ROOT / "results" / "tables" / "thresholds.json"
GRID = [round(x, 2) for x in np.arange(0.0, 1.0, 0.05)]
SELECTION_METRIC = "abstract_rationalized"


def assemble(verify_rows, claim_ids, tau_verdict: float, tau_sentence: float) -> list[dict]:
    by_claim: dict[int, dict] = {cid: {} for cid in claim_ids}
    for r in verify_rows:
        if r["is_oracle"] or r["claim_id"] not in by_claim:
            continue
        if r["verdict"] not in VERDICT_TO_SCIFACT or r["verdict_confidence"] < tau_verdict:
            continue
        probs = r["evidence_probs"]
        chosen = sorted((i for i, p in enumerate(probs) if p >= tau_sentence), key=lambda i: (-probs[i], i))
        if not chosen:
            chosen = [int(np.argmax(probs))]
        by_claim[r["claim_id"]][str(int(r["doc_id"]))] = {"sentences": chosen, "label": VERDICT_TO_SCIFACT[r["verdict"]]}
    return [{"id": cid, "evidence": ev} for cid, ev in by_claim.items()]


def score(preds, claims) -> dict:
    gold = {c.claim_id: c.evidence for c in claims}
    total = Counter()
    for p in preds:
        total += claim_counts(p, gold[p["id"]])
    return metrics_from_counts(total)


def select_thresholds(verify_rows, claims) -> dict:
    ids = [c.claim_id for c in claims]
    best = None
    for tv in GRID:
        for ts in GRID:
            m = score(assemble(verify_rows, ids, tv, ts), claims)
            key = (m[SELECTION_METRIC]["f1"], m["sentence_label"]["f1"])
            if best is None or key > best[0]:
                best = (key, tv, ts, m)
    return {"tau_verdict": best[1], "tau_sentence": best[2], "train_metrics": best[3],
            "selection_metric": SELECTION_METRIC, "question_version": QUESTION_VERSION}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "dev"], required=True)
    split = ap.parse_args().split
    claims = load_split(split)
    rows = read_jsonl(split_dir(split) / "verify.jsonl")
    if split == "train":
        th = select_thresholds(rows, claims)
        THRESHOLDS_PATH.parent.mkdir(parents=True, exist_ok=True)
        THRESHOLDS_PATH.write_text(json.dumps(th, indent=2))
    else:
        th = json.loads(THRESHOLDS_PATH.read_text())
        if th["question_version"] != QUESTION_VERSION:
            raise SystemExit(f"thresholds were tuned on {th['question_version']}, questions are {QUESTION_VERSION}")
    preds = assemble(rows, [c.claim_id for c in claims], th["tau_verdict"], th["tau_sentence"])
    write_jsonl(split_dir(split) / "predictions.jsonl", preds)
    m = score(preds, claims)
    print(f"tau_verdict={th['tau_verdict']} tau_sentence={th['tau_sentence']} | "
          + " ".join(f"{k}.f1={v['f1']:.3f}" for k, v in m.items()))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests and confirm they pass**

Run: `uv run pytest tests/test_assemble.py -v`. Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/04_assemble_predictions.py tests/test_assemble.py
git commit -m "Add prediction assembly with train-only threshold selection

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: Development run on train, then freeze the questions (paid, ≤ $0.20)

**Files:**
- Modify (only if revising): `scripts/lib/questions.py`
- Create: `docs/scifact_question_development_log.md`

- [ ] **Step 1: Run v1 on train**

```bash
zsh -ic 'uv run python scripts/02_jev_rerank.py --split train' && \
zsh -ic 'uv run python scripts/03_jev_verify.py --split train' && \
uv run python scripts/04_assemble_predictions.py --split train
```

Expected: the three summary lines, and spend under about $0.12 so far (check `tail -1 data/processed/scifact/spend_ledger.csv`).

- [ ] **Step 2: Error analysis (code only, $0)**

```bash
uv run python - <<'EOF'
import collections, sys
sys.path.insert(0, "scripts")
from lib.data import load_split, read_jsonl, split_dir
claims = {c.claim_id: c for c in load_split("train")}
rows = read_jsonl(split_dir("train") / "verify.jsonl")
conf = collections.Counter()
for r in rows:
    gold = claims[r["claim_id"]].evidence.get(r["doc_id"])
    g = {"SUPPORT": "supports", "CONTRADICT": "contradicts"}[gold[0]] if gold else "not_enough_info"
    conf[(g, r["verdict"])] += 1
for k, v in sorted(conf.items()): print(k, v)
EOF
```

Write the confusion counts and 5 illustrative errors (claim text, verdict, gold) into `docs/scifact_question_development_log.md` under `## v1`.

- [ ] **Step 3: Decide whether to revise (at most one revision)**

Revise **only** for a systematic, explainable failure seen in Step 2, e.g. `contradicts` answered as `not_enough_info` for more than 30% of gold CONTRADICT pairs. That would indicate literal reading, which the docs flag as a known Jev weakness. To revise:
- edit the wording in `questions.py`;
- set `QUESTION_VERSION = "v2"`;
- rerun Step 1 (new cache keys, about $0.08);
- keep whichever version has the higher train `abstract_rationalized` F1, restoring its wording and version if needed, then rerun `04 --split train` so that `thresholds.json` matches.

Log the change and both train scores in the development log. No third version.

- [ ] **Step 4: Freeze**

```bash
git add scripts/lib/questions.py docs/scifact_question_development_log.md results/tables/thresholds.json data/processed/scifact/train data/processed/scifact/spend_ledger.csv
git commit -m "Freeze Jev questions <version> and thresholds after train development

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git tag questions-frozen
```

From here on, `questions.py` and `thresholds.json` must not change. Before Task 12, check with `git diff questions-frozen -- scripts/lib/questions.py results/tables/thresholds.json`, which must print nothing.

---

### Task 10: Evaluation statistics, baselines, and the published-results table

**Files:**
- Modify: `scripts/lib/evalstats.py` (append)
- Create: `scripts/lib/baselines.py`, `results/tables/published_scifact.csv`, `tests/test_evalstats.py`

**Interfaces:**
- Produces:
  - `bootstrap_ci(per_claim: list, stat_fn, n=1000, seed=SEED) -> tuple[float, float]`
  - `reliability(prob: np.ndarray, outcome: np.ndarray, n_bins=10) -> pd.DataFrame` with columns `bin_lo, bin_hi, n, mean_prob, frac_pos`
  - `ece(prob, outcome, n_bins=10) -> float`
  - `selective_curve(conf, correct) -> pd.DataFrame` with columns `threshold, coverage, accuracy`
  - `majority_baseline(bm25_rows) -> list[dict]` and `lexical_baseline(bm25_rows, corpus, claims) -> list[dict]`

- [ ] **Step 1: Write the failing tests**

`tests/test_evalstats.py`:

```python
import numpy as np
import pytest

from lib.baselines import lexical_baseline, majority_baseline
from lib.data import Claim, Doc
from lib.evalstats import bootstrap_ci, ece, reliability, selective_curve


def test_bootstrap_ci_brackets_mean_and_is_deterministic():
    vals = list(np.linspace(0, 1, 50))
    lo, hi = bootstrap_ci(vals, lambda xs: float(np.mean(xs)))
    assert lo < 0.5 < hi
    assert (lo, hi) == bootstrap_ci(vals, lambda xs: float(np.mean(xs)))


def test_ece_zero_when_perfectly_calibrated_and_positive_otherwise():
    p = np.array([0.25] * 4 + [0.75] * 4)
    y = np.array([1, 0, 0, 0, 1, 1, 1, 0])
    assert ece(p, y) == pytest.approx(0.0)
    assert ece(np.array([0.9] * 4), np.array([0, 0, 0, 0])) == pytest.approx(0.9)
    assert reliability(p, y)["n"].sum() == 8


def test_selective_curve_accuracy_rises_when_errors_are_low_confidence():
    conf = np.array([0.9, 0.8, 0.2, 0.1])
    correct = np.array([1, 1, 0, 0])
    curve = selective_curve(conf, correct)
    assert curve.iloc[0]["coverage"] == 1.0 and curve.iloc[0]["accuracy"] == 0.5
    assert curve["accuracy"].max() == 1.0


def test_baselines_predict_top1_support():
    corpus = {1: Doc(1, "t", ("alpha beta", "gamma", "claim words here", "delta"))}
    bm25 = [{"claim_id": 7, "doc_ids": [1], "scores": [1.0]}]
    assert majority_baseline(bm25) == [{"id": 7, "evidence": {"1": {"sentences": [0, 1, 2], "label": "SUPPORT"}}}]
    lex = lexical_baseline(bm25, corpus, [Claim(7, "claim words", {})])
    assert lex[0]["evidence"]["1"]["sentences"][0] == 2
```

- [ ] **Step 2: Run them and confirm they fail**

Run: `uv run pytest tests/test_evalstats.py -v`. Expected: FAIL with `ImportError`.

- [ ] **Step 3: Append to `scripts/lib/evalstats.py`**

```python
import pandas as pd


def bootstrap_ci(per_claim: list, stat_fn, n: int = 1000, seed: int = SEED, alpha: float = 0.05):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(per_claim))
    stats = [stat_fn([per_claim[i] for i in rng.choice(idx, size=len(idx), replace=True)]) for _ in range(n)]
    return float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2))


def reliability(prob, outcome, n_bins: int = 10) -> pd.DataFrame:
    prob, outcome = np.asarray(prob, float), np.asarray(outcome, float)
    edges = np.linspace(0, 1, n_bins + 1)
    bins = np.clip(np.digitize(prob, edges[1:-1], right=True), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = bins == b
        rows.append({"bin_lo": edges[b], "bin_hi": edges[b + 1], "n": int(m.sum()),
                     "mean_prob": float(prob[m].mean()) if m.any() else np.nan,
                     "frac_pos": float(outcome[m].mean()) if m.any() else np.nan})
    return pd.DataFrame(rows)


def ece(prob, outcome, n_bins: int = 10) -> float:
    t = reliability(prob, outcome, n_bins).dropna()
    return float((t["n"] / t["n"].sum() * (t["mean_prob"] - t["frac_pos"]).abs()).sum())


def selective_curve(conf, correct) -> pd.DataFrame:
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    rows = []
    for th in np.unique(np.concatenate([[0.0], conf])):
        m = conf >= th
        if m.any():
            rows.append({"threshold": float(th), "coverage": float(m.mean()), "accuracy": float(correct[m].mean())})
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Create `scripts/lib/baselines.py`**

```python
"""Non-Jev baselines, emitted in SciFact prediction format."""

from __future__ import annotations

from lib.evalstats import tokenize


def majority_baseline(bm25_rows) -> list[dict]:
    """BM25 top-1, always SUPPORT (the majority label), first three sentences as evidence."""
    return [{"id": r["claim_id"], "evidence": {str(r["doc_ids"][0]): {"sentences": [0, 1, 2], "label": "SUPPORT"}}}
            for r in bm25_rows]


def lexical_baseline(bm25_rows, corpus, claims) -> list[dict]:
    """BM25 top-1, SUPPORT, and the three sentences with the most token overlap with the claim."""
    text = {c.claim_id: set(tokenize(c.text)) for c in claims}
    out = []
    for r in bm25_rows:
        doc = corpus[r["doc_ids"][0]]
        overlap = [len(text[r["claim_id"]] & set(tokenize(s))) for s in doc.sentences]
        top = sorted(range(len(overlap)), key=lambda i: (-overlap[i], i))[:3]
        out.append({"id": r["claim_id"], "evidence": {str(doc.doc_id): {"sentences": top, "label": "SUPPORT"}}})
    return out
```

Note: the majority baseline's `[0, 1, 2]` references sentences that may not exist in a 1–2 sentence abstract. The official scorer only compares indices, so this is harmless.

- [ ] **Step 5: Run the tests and confirm they pass**

Run: `uv run pytest tests/test_evalstats.py -v`. Expected: 4 passed.

- [ ] **Step 6: Transcribe the published numbers**

Open the papers (arXiv:2004.14974, VeriSci; arXiv:2112.01640, MultiVerS). Find the tables that report the four official metrics (`sentence_selection`, `sentence_label`, `abstract_label_only`, `abstract_rationalized`; F1, and also P/R where reported). Write `results/tables/published_scifact.csv` with exactly these columns:

```
system,split,retrieval_setting,metric,precision,recall,f1,source
```

- `split`: `dev` or `test`, **exactly as stated** in the table caption.
- `retrieval_setting`: `open` or `oracle`.
- `source`: the arXiv ID plus the table number, e.g. `arXiv:2004.14974 Table 3`.
- Leave P or R empty if they are not reported. Never copy a number whose split is unclear; omit it and add a row whose `metric` is `NOTE`, with the reason in `source`.

Check with `uv run python -c "import pandas as pd; d=pd.read_csv('results/tables/published_scifact.csv'); assert d['source'].notna().all() and set(d['split'])<={'dev','test'}; print(d)"`.

- [ ] **Step 7: Commit**

```bash
git add scripts/lib/evalstats.py scripts/lib/baselines.py tests/test_evalstats.py results/tables/published_scifact.csv
git commit -m "Add bootstrap, calibration and selective-prediction stats, baselines, and sourced published SciFact results

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 11: Evaluation script and figures (smoke test on train)

**Files:**
- Create: `scripts/05_evaluate.py`

**Interfaces:**
- Consumes: everything above
- Produces: the tables and figures listed in spec §5.6 (with the split name in the filename, e.g. `official_metrics_dev.csv`)

- [ ] **Step 1: Implement `scripts/05_evaluate.py`**

```python
"""Stage 5: official metrics with bootstrap CIs, retrieval, oracle verification, calibration, figures.

Usage: uv run python scripts/05_evaluate.py --split {train,dev}
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from lib.baselines import lexical_baseline, majority_baseline
from lib.budget import LEDGER_PATH
from lib.data import ROOT, load_corpus, load_split, read_jsonl, split_dir
from lib.evalstats import bootstrap_ci, ece, mrr, recall_at_k, reliability, selective_curve
from lib.official import claim_counts, metrics_from_counts

TABLES, FIGS = ROOT / "results" / "tables", ROOT / "results" / "figures"
METRICS = ["abstract_label_only", "abstract_rationalized", "sentence_selection", "sentence_label"]
GOLD_TO_VERDICT = {"SUPPORT": "supports", "CONTRADICT": "contradicts"}
RED, BLUE, GREEN, PURPLE, ORANGE, GREY = "#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#e67e22", "#bdc3c7"

plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "axes.spines.top": False,
                     "axes.spines.right": False, "font.family": "sans-serif"})


def official_with_ci(preds, gold) -> pd.DataFrame:
    per_claim = [claim_counts(p, gold[p["id"]]) for p in preds]
    point = metrics_from_counts(sum(per_claim, Counter()))
    rows = []
    for m in METRICS:
        lo, hi = bootstrap_ci(per_claim, lambda xs, m=m: metrics_from_counts(sum(xs, Counter()))[m]["f1"])
        rows.append({"metric": m, **point[m], "f1_lo": lo, "f1_hi": hi})
    return pd.DataFrame(rows)


def verdict_table(verify_rows, claims) -> pd.DataFrame:
    """Gold pairs (claim, evidence doc), plus NEI claims paired with their re-ranked top-1."""
    by_id = {c.claim_id: c for c in claims}
    out = []
    for r in verify_rows:
        c = by_id[r["claim_id"]]
        if r["doc_id"] in c.evidence:
            gold = GOLD_TO_VERDICT[c.evidence[r["doc_id"]][0]]
        elif not c.evidence and r["rerank_position"] == 0:
            gold = "not_enough_info"
        else:
            continue
        out.append({"claim_id": r["claim_id"], "doc_id": r["doc_id"], "gold": gold, "pred": r["verdict"],
                    "confidence": r["verdict_confidence"], "p_max": max(r["verdict_probs"].values())})
    df = pd.DataFrame(out)
    df["correct"] = (df["gold"] == df["pred"]).astype(int)
    return df


def sentence_table(verify_rows, claims) -> pd.DataFrame:
    by_id = {c.claim_id: c for c in claims}
    out = []
    for r in verify_rows:
        ev = by_id[r["claim_id"]].evidence.get(r["doc_id"])
        if not ev:
            continue
        gold_idx = {i for rat in ev[1] for i in rat}
        out += [{"claim_id": r["claim_id"], "p": p, "gold": int(i in gold_idx)} for i, p in enumerate(r["evidence_probs"])]
    return pd.DataFrame(out)


def fig_retrieval(bm25_rank, jev_rank, gold, path):
    ks = list(range(1, 31))
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.plot(ks, [recall_at_k(bm25_rank, gold, k) for k in ks], color=GREY, lw=2.5, label="BM25")
    ax.plot(ks, [recall_at_k(jev_rank, gold, k) for k in ks], color=RED, lw=2.5, label="BM25 → Jev re-rank")
    ax.axvline(3, color="grey", ls="--", lw=1, alpha=0.6)
    ax.set_xlabel("k (abstracts kept)", fontsize=15, fontweight="bold")
    ax.set_ylabel("Recall@k of gold abstracts", fontsize=15, fontweight="bold")
    ax.set_title("Retrieval: does Jev re-ranking beat BM25?", fontsize=16, fontweight="bold", pad=12)
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=12, frameon=True, framealpha=0.9)
    plt.tight_layout(); fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)


def fig_reliability(table, title, xlabel, path, colour):
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.plot([0, 1], [0, 1], color="black", lw=1, alpha=0.4)
    t = table.dropna()
    ax.scatter(t["mean_prob"], t["frac_pos"], s=40 + 400 * t["n"] / t["n"].max(), c=colour, alpha=0.85,
               edgecolors="white", linewidths=1, zorder=3)
    ax.plot(t["mean_prob"], t["frac_pos"], color=colour, lw=1.5)
    for _, r in t.iterrows():
        ax.annotate(f"n={int(r['n'])}", (r["mean_prob"], r["frac_pos"]), textcoords="offset points",
                    xytext=(6, -12), fontsize=9, fontweight="bold", color="#333")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel(xlabel, fontsize=15, fontweight="bold")
    ax.set_ylabel("Observed frequency correct / positive", fontsize=15, fontweight="bold")
    ax.set_title(title, fontsize=16, fontweight="bold", pad=12)
    ax.tick_params(labelsize=12)
    plt.tight_layout(); fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)


def fig_selective(curve, path):
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.plot(curve["coverage"], curve["accuracy"], color=BLUE, lw=2.5)
    ax.set_xlim(1.02, 0)
    ax.set_xlabel("Coverage (fraction of verdicts kept, most confident first)", fontsize=15, fontweight="bold")
    ax.set_ylabel("Verdict accuracy", fontsize=15, fontweight="bold")
    ax.set_title("Can Jev's confidence drive abstention?", fontsize=16, fontweight="bold", pad=12)
    ax.tick_params(labelsize=12)
    plt.tight_layout(); fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)


def fig_metrics(df, path):
    systems = list(df["system"].unique())
    colours = dict(zip(systems, [RED, GREY, "#7f8c8d", PURPLE, GREEN, ORANGE]))
    fig, axes = plt.subplots(1, len(METRICS), figsize=(20, 8), sharey=True)
    for ax, m in zip(axes, METRICS):
        sub = df[df["metric"] == m].set_index("system").reindex(systems)
        y = np.arange(len(systems))
        err = np.vstack([sub["f1"] - sub["f1_lo"], sub["f1_hi"] - sub["f1"]])
        ax.barh(y, sub["f1"], color=[colours[s] for s in systems], edgecolor="white", height=0.7,
                xerr=np.nan_to_num(err), error_kw=dict(ecolor="#333", lw=1, capsize=3))
        for yi, v in zip(y, sub["f1"]):
            if not np.isnan(v):
                ax.text(v + 0.02, yi, f"{v:.2f}", va="center", fontsize=8.5, color="#333")
        ax.set_yticks(y, systems, fontsize=11)
        ax.set_xlim(0, 1)
        ax.set_title(m.replace("_", " "), fontsize=14, fontweight="bold", pad=10)
        ax.tick_params(labelsize=11)
    axes[0].invert_yaxis()
    fig.supxlabel("F1 (95% bootstrap CI where computed here)", fontsize=15, fontweight="bold")
    plt.tight_layout(); fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "dev"], required=True)
    split = ap.parse_args().split
    sd = split_dir(split)
    TABLES.mkdir(parents=True, exist_ok=True); FIGS.mkdir(parents=True, exist_ok=True)
    corpus, claims = load_corpus(), load_split(split)
    gold_ev = {c.claim_id: c.evidence for c in claims}
    bm25_rows, rerank_rows = read_jsonl(sd / "bm25_top30.jsonl"), read_jsonl(sd / "rerank.jsonl")
    verify_rows, preds = read_jsonl(sd / "verify.jsonl"), read_jsonl(sd / "predictions.jsonl")

    # Official metrics: Jev vs baselines.
    systems = {"Jev pipeline": preds, "B1 majority": majority_baseline(bm25_rows),
               "B2 lexical": lexical_baseline(bm25_rows, corpus, claims)}
    official = pd.concat([official_with_ci(p, gold_ev).assign(system=s) for s, p in systems.items()])
    official.to_csv(TABLES / f"official_metrics_{split}.csv", index=False)

    # Retrieval.
    gold_docs = {c.claim_id: set(c.evidence) for c in claims}
    bm25_rank = {r["claim_id"]: r["doc_ids"] for r in bm25_rows}
    jev_rank = {r["claim_id"]: r["doc_ids"] for r in rerank_rows}
    retr = pd.DataFrame([{"ranker": n, **{f"R@{k}": recall_at_k(rk, gold_docs, k) for k in (1, 3, 10, 30)},
                          "MRR": mrr(rk, gold_docs)} for n, rk in (("BM25", bm25_rank), ("Jev re-rank", jev_rank))])
    retr.to_csv(TABLES / f"retrieval_{split}.csv", index=False)
    fig_retrieval(bm25_rank, jev_rank, gold_docs, FIGS / f"retrieval_recall_at_k_{split}.png")

    # Verification in isolation, plus calibration.
    vt = verdict_table(verify_rows, claims)
    gp = vt[vt["gold"] != "not_enough_info"]
    st = sentence_table(verify_rows, claims)
    tau_s = json.loads((TABLES / "thresholds.json").read_text())["tau_sentence"]
    pred_s = (st["p"] >= tau_s).astype(int)
    tp = int(((pred_s == 1) & (st["gold"] == 1)).sum())
    prec = tp / max(int(pred_s.sum()), 1)
    rec = tp / max(int(st["gold"].sum()), 1)
    oracle = pd.DataFrame([{
        "n_gold_pairs": len(gp),
        "verdict_accuracy_gold_pairs": gp["correct"].mean(),
        "verdict_macro_f1_support_contradict": f1_score(gp["gold"], gp["pred"],
                                                        labels=["supports", "contradicts"], average="macro"),
        "n_nei_pairs": int((vt["gold"] == "not_enough_info").sum()),
        "nei_accuracy": vt.loc[vt["gold"] == "not_enough_info", "correct"].mean(),
        "evidence_precision": prec, "evidence_recall": rec,
        "evidence_f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "evidence_auroc": roc_auc_score(st["gold"], st["p"]),
        "tau_sentence": tau_s,
    }])
    oracle.to_csv(TABLES / f"verification_oracle_{split}.csv", index=False)
    pd.crosstab(vt["gold"], vt["pred"]).to_csv(TABLES / f"verdict_confusion_{split}.csv")

    rel_v, rel_e = reliability(vt["p_max"], vt["correct"]), reliability(st["p"], st["gold"])
    pd.concat([rel_v.assign(question="verdict (max prob vs correct)"),
               rel_e.assign(question="evidence noul (P vs gold)")]).to_csv(TABLES / f"calibration_{split}.csv", index=False)
    pd.DataFrame([{"question": "verdict", "ece": ece(vt["p_max"], vt["correct"])},
                  {"question": "evidence", "ece": ece(st["p"], st["gold"])}]).to_csv(TABLES / f"ece_{split}.csv", index=False)
    fig_reliability(rel_v, "Verdict calibration", "Top-choice probability", FIGS / f"reliability_verdict_{split}.png", BLUE)
    fig_reliability(rel_e, "Evidence-sentence calibration", "P(evidence)", FIGS / f"reliability_evidence_{split}.png", PURPLE)
    curve = selective_curve(vt["confidence"], vt["correct"])
    curve.to_csv(TABLES / f"selective_prediction_{split}.csv", index=False)
    fig_selective(curve, FIGS / f"selective_prediction_{split}.png")

    # Published systems, as context.
    pub = pd.read_csv(TABLES / "published_scifact.csv")
    pub = pub[pub["metric"].isin(METRICS)].assign(
        system=lambda d: d["system"] + " (" + d["split"] + ", " + d["retrieval_setting"] + ")", f1_lo=np.nan, f1_hi=np.nan)
    fig_metrics(pd.concat([official, pub[["system", "metric", "f1", "f1_lo", "f1_hi"]]]),
                FIGS / f"metrics_vs_baselines_{split}.png")

    # Cost.
    ledger = pd.read_csv(LEDGER_PATH)
    ledger.groupby(["stage", "split", "question_version"])[["input_tokens", "cost_usd"]].sum().reset_index() \
        .to_csv(TABLES / "cost_summary.csv", index=False)

    print(official.pivot(index="system", columns="metric", values="f1").round(3).to_string())
    print(retr.round(3).to_string(index=False))
    print(oracle.round(3).T.to_string())
    print(f"total spend ${ledger['cost_usd'].sum():.4f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-test on train ($0, all cached)**

Run: `uv run python scripts/05_evaluate.py --split train`
Expected: four printed blocks. All `*_train.*` tables and figures exist. Open each PNG with the Read tool and check the labels, legend and layout. The printed Jev `abstract_rationalized` F1 must equal the value in `thresholds.json`.

- [ ] **Step 3: Commit**

```bash
git add scripts/05_evaluate.py results/tables/*_train.csv results/figures/*_train.png
git commit -m "Add evaluation stage: official metrics with bootstrap CIs, retrieval, oracle verification, calibration, figures

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 12: Held-out dev run (once; paid, about $0.30)

- [ ] **Step 1: Confirm the freeze and the budget**

```bash
git diff --exit-code questions-frozen -- scripts/lib/questions.py results/tables/thresholds.json && echo FROZEN-OK
uv run python -c "import pandas as pd; print('spent', pd.read_csv('data/processed/scifact/spend_ledger.csv').cost_usd.sum())"
uv run pytest -q
```

Expected: `FROZEN-OK`, spend ≤ about $0.25, all tests pass. If spend + $0.30 > $0.90, stop and report instead of running.

- [ ] **Step 2: Run dev**

```bash
zsh -ic 'uv run python scripts/02_jev_rerank.py --split dev' && \
zsh -ic 'uv run python scripts/03_jev_verify.py --split dev' && \
uv run python scripts/04_assemble_predictions.py --split dev && \
uv run python scripts/05_evaluate.py --split dev
```

If a stage exits with code 2 (budget), **do not** relax the cap. Report to the user with the ledger total.

- [ ] **Step 3: Verify the outputs**

Open every `results/figures/*_dev.png` with the Read tool. Confirm that `results/tables/official_metrics_dev.csv` has three systems × four metrics with CIs, and that the total spend printed is ≤ $0.90.

Acceptance criterion 1 (a rerun is free): record the ledger total, rerun `02` and `03` for dev, and confirm the total is unchanged:

```bash
before=$(uv run python -c "import pandas as pd; print(pd.read_csv('data/processed/scifact/spend_ledger.csv').cost_usd.sum())")
zsh -ic 'uv run python scripts/02_jev_rerank.py --split dev' && zsh -ic 'uv run python scripts/03_jev_verify.py --split dev'
after=$(uv run python -c "import pandas as pd; print(pd.read_csv('data/processed/scifact/spend_ledger.csv').cost_usd.sum())")
[ "$before" = "$after" ] && echo RERUN-FREE-OK
```

- [ ] **Step 4: Commit**

```bash
git add data/processed/scifact/dev data/processed/scifact/spend_ledger.csv results/tables results/figures
git commit -m "Run frozen Jev pipeline once on held-out SciFact dev: <paste abstract_rationalized F1 and CI>, total spend <paste>

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 13: Methods write-up and project docs

**Files:**
- Create: `docs/scifact_jev_evaluation_methods.md`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Write the methods doc** with these sections, filled from the actual outputs (not the plan):
  1. **Aim.**
  2. **Data:** SciFact release, SHA-256, split sizes, the train sample seed, the licence.
  3. **Model:** `jev-1.13.0`, SDK version, dates of the calls from the ledger.
  4. **Pipeline:** BM25 parameters; the verbatim question wording at the frozen version, copied from `questions.py`; the chunking rule; the thresholds and how they were chosen.
  5. **Question development:** a summary of the development log, including whether v2 happened.
  6. **Metrics:** official definitions, the bootstrap procedure, the ECE definition and binning, what "confidence" means for Choice vs Noul.
  7. **Results:** tables copied from the CSVs, with CIs, and the figures linked.
  8. **Comparison with published systems:** with the split and setting stated for each row, and the explicit caveat for test-set rows.
  9. **Cost:** from `cost_summary.csv`.
  10. **Limitations:** single run; train sample of 100; thresholds tuned on train; BM25 ceiling; top-3 cap; Jev's documented weaknesses (literal reading, indirection); the published comparison not being head-to-head.
  11. **References:** those from spec §References.

- [ ] **Step 2: Update `CLAUDE.md`**, adding a `## SciFact evaluation` section that lists stages `00`–`05` and their commands, the budget ledger path and cap, the freeze tag, and the rule that dev is never rerun with changed questions.

- [ ] **Step 3: Final checks and commit**

```bash
uv run pytest -q
git add docs/scifact_jev_evaluation_methods.md CLAUDE.md
git commit -m "Document SciFact evaluation methods, results, and limitations

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
