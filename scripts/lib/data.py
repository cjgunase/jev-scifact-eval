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
