# Design: Evaluating Jev on SciFact scientific claim verification

- **Date:** 2026-09-29
- **Status:** Draft, awaiting review
- **Budget:** hard cap of **$0.90** TypeSafe spend (user limit: $1.00)

## 1. Purpose

Find out what Jev (TypeSafe System One, `jev-1.13.0`) can and cannot do on a real
biomedical evidence task, with labels from independent experts. The previous
synthetic test ([jev_sample_annotation_methods.md](../../jev_sample_annotation_methods.md))
leaked answers into its inputs (80 of 99 non-default labels appeared as keywords in the text). This
design fixes that by using an external benchmark, simple baselines, and a held-out split.

The goal is an honest assessment that would hold up in a methods section, not a showcase.

### Questions this evaluation answers

1. **Retrieval:** Does Jev re-ranking beat BM25 at putting the correct abstract near the top?
2. **Verification:** Given the right abstract, how accurately does Jev judge whether it supports or contradicts a claim?
3. **Evidence selection:** Can Jev pick out the evidence sentences that expert annotators marked?
4. **Confidence:** Does Jev's reported confidence track its accuracy, and can it drive abstention?
5. **Full pipeline:** How does the end-to-end system score on the official SciFact metrics, compared with baselines and published systems?

## 2. Dataset

SciFact (Wadden et al., 2020), from AI2's original release:
`https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz`

- A corpus of ~5.2k PubMed abstracts, pre-split into sentences.
- Expert-written claims, each with gold evidence: relevant abstract IDs, a label per abstract
  (`SUPPORT` / `CONTRADICT`), and evidence sentence sets. Claims without evidence count as NEI (not enough info).
- License (per the repository LICENSE.md): claims CC BY 4.0; abstracts ODC-By 1.0 (from S2ORC); code Apache-2.0. Raw files live in `data/raw/scifact/` (gitignored) and are never modified.
- Recorded at download (2026-09-29): tarball SHA-256 `11c62128…76be`; corpus 5,183 abstracts (median 8 sentences, max 367); train 809 claims (505 with evidence); dev 300 claims (188 with evidence).
- **Join-key caveat:** `doc_id` is an int in the corpus but a string key in `claims.evidence`. Every join must cast explicitly.

### Splits

| Split | Use | Size |
|---|---|---|
| Train | **Development only**: question wording, thresholds | Fixed random sample of 100 claims (`numpy` seed 20260929) |
| Dev | **Held-out evaluation**, run once | All dev claims (~300) |
| Test | Not used (labels are hidden) | none |

The held-out dev split is **frozen**. No question wording or threshold may change after the
first held-out run. Any held-out call made before freezing is logged and flagged in the report.

## 3. Pipeline

Each stage is one script that reads and writes files, so any stage can be re-run
on its own. Every stage takes `--split {train,dev}`.

```
claims + corpus
   │
   ▼  01_bm25_retrieve.py      (code, $0)
top-30 abstracts per claim
   │
   ▼  02_jev_rerank.py         (Jev: 1 Score per pair)
re-ranked top-30
   │
   ▼  03_jev_verify.py         (Jev: 1 Choice + N Nouls per pair, top-3 + gold-oracle pairs)
verdict + P(evidence) per sentence
   │
   ▼  04_assemble_predictions.py (code, thresholds from train)
SciFact-format predictions.jsonl
   │
   ▼  05_evaluate.py           (code: official metrics, baselines, calibration, figures)
results/tables, results/figures
```

### 3.1 `01_bm25_retrieve.py`
- BM25 (`rank_bm25`, Okapi, default k1 = 1.5, b = 0.75) over title + abstract. Tokenisation: lowercase, split on non-alphanumerics, remove English stopwords.
- Output: `data/processed/{split}/bm25_top30.jsonl` (claim_id, ranked doc_ids, BM25 scores).
- Also reports BM25 recall@{1,3,10,30} against the gold abstracts. This bounds everything downstream: a gold abstract outside the top 30 cannot be recovered.

### 3.2 `02_jev_rerank.py`
- One request per (claim, candidate abstract). State:
  `{"claim": <text>, "abstract": {"title": ..., "text": <full abstract>}}`
- One **Score** question with described levels (draft, finalised on train):
  - 0: The abstract is about a different topic from the claim.
  - 1: Same general topic, but it does not address what the claim asserts.
  - 2: Addresses the claim's subject, but its findings neither confirm nor refute the claim.
  - 3: Reports findings that directly confirm or refute the claim.
- Ranking key: the Score expectation, with ties broken by the BM25 rank.
- Output: `data/processed/{split}/rerank.jsonl`.

### 3.3 `03_jev_verify.py`
- Pairs judged: the top 3 after re-ranking for every claim, **plus** any gold abstract outside that top 3 (the "oracle" pairs, used only for the verification-in-isolation analysis).
- One request per pair, all questions in parallel. State:
  `{"claim": ..., "abstract": {"title": ..., "sentences": {"s0": ..., "s1": ...}}}`
- Questions:
  - `verdict`, a **Choice**: `supports`, `contradicts`, `not_enough_info`, with criteria written so that "same topic but no finding about the claim" maps to `not_enough_info`.
  - `evidence_s{i}`, one **Noul** per sentence: "Does sentence `abstract.sentences.s{i}` report a finding, on its own or together with neighbouring sentences, that bears on whether the claim is true?"
- Abstracts with more than 40 sentences (9 in the corpus, max 367) have their sentence Nouls split into
  chunks of at most 40 per request over the same state. The verdict is asked only in the first chunk.
- Output: `data/processed/{split}/verify.jsonl`.

### 3.4 `04_assemble_predictions.py`
Pure code. Two thresholds are chosen on train by maximising abstract-level Label+Rationale F1:
- `τ_verdict`: keep an abstract if verdict ≠ `not_enough_info` and verdict confidence ≥ `τ_verdict`.
- `τ_sentence`: evidence sentences are those with P(evidence) ≥ `τ_sentence`, ordered by probability, highest first. If none pass, take the single highest-probability sentence.
- Predicted labels map to SciFact's labels (`supports` → `SUPPORT`, `contradicts` → `CONTRADICT`).
- Output: `data/processed/{split}/predictions.jsonl` in the official submission format.
- The thresholds are written to `results/tables/thresholds.json` along with the train metrics that selected them.

### 3.5 `05_evaluate.py`
See §5.

## 4. Shared code (`scripts/lib/`)

| Module | Responsibility |
|---|---|
| `data.py` | Load the corpus, claims and splits; build states; **assert that no gold fields appear in any state** |
| `questions.py` | Every Jev question definition, with a `QUESTION_VERSION` string |
| `jev_client.py` | Cached, budget-guarded wrapper around `AsyncTypeSafeClient` |
| `budget.py` | Spend ledger and hard cap |
| `scifact_official/` | Official `metrics.py` and `data.py`, vendored unmodified from `allenai/scifact` @ `68b98a56`, with provenance in `__init__.py` |
| `baselines.py` | Baseline predictors |

### 4.1 Caching and budget guard
- Cache key: SHA-256 of (model ID, state, questions). Stored as one JSON file per key in `data/processed/jev_cache/` (gitignored), so a cached call costs $0.
- Before each uncached call, the guard estimates its tokens (bytes/4, rounded up, times a safety factor of 1.5). If ledger total + estimate would exceed **$0.90**, it raises `BudgetExceeded` and the stage stops cleanly.
- After each call, the actual `usage.input_tokens` is appended to `data/processed/spend_ledger.csv` (timestamp, stage, split, question version, tokens, cost in USD at $0.042/M, running total).
- Model pinned to `jev-1.13.0` rather than `jev-latest`, so that moving the alias cannot change results midway.
- Concurrency ≤ 16 requests; the SDK's default retry and backoff handle rate-limit responses (HTTP 429).

### 4.2 Pilot gate
Before any stage runs at scale, a `--pilot` flag runs 5 train pairs through stages 2 and 3,
then prints the measured tokens per call and the projected cost for the full plan. It stops without
scaling if the projection exceeds $0.90. In that case, fall back to re-ranking the top 20 (§7).

## 5. Evaluation

All metrics are reported on held-out dev, with 95% bootstrap CIs (1,000 resamples of claims, seed 20260929).

### 5.1 Official SciFact metrics
Abstract-level (Label-only; Label+Rationale) and sentence-level (Selection-only; Selection+Label):
precision, recall and F1, computed with the vendored official code.

### 5.2 Retrieval
Recall@{1,3,10} and MRR of the gold abstracts: BM25 alone vs BM25 + Jev re-ranking.

### 5.3 Verification in isolation (oracle abstracts)
On (claim, gold abstract) pairs only:
- Verdict accuracy, and macro-F1 over {supports, contradicts}
- A confusion matrix, which also covers NEI claims paired with their re-ranked top-1 abstract as negatives (already judged in the top 3, so no extra cost)
- Evidence-sentence precision, recall and F1 at `τ_sentence`, plus AUROC of P(evidence) against gold evidence sentences

### 5.4 Confidence
- A reliability diagram and ECE (10 equal-width bins) for the verdict Choice and for the evidence Nouls
- A selective-prediction curve: verdict accuracy vs coverage as the confidence threshold rises

### 5.5 Baselines

| Baseline | Retrieval | Label | Evidence sentences |
|---|---|---|---|
| B1: majority | BM25 top-1 | `SUPPORT` | first 3 sentences |
| B2: lexical | BM25 top-1 | `SUPPORT` | top 3 by token overlap with the claim |
| Published systems | from the literature | | |

Published numbers (VeriSci: Wadden et al., 2020; MultiVerS: Wadden et al., 2022) come from the
papers, and each row is labelled with its split and setting (open vs oracle retrieval). Where only
test-set numbers exist, the table says so explicitly; those rows are context, not a head-to-head comparison.

### 5.6 Outputs
- `results/tables/`: `official_metrics.csv`, `retrieval.csv`, `verification_oracle.csv`, `calibration.csv`, `baselines.csv`, `cost_summary.csv`, `thresholds.json`
- `results/figures/`: `retrieval_recall_at_k.png`, `reliability_verdict.png`, `reliability_evidence.png`, `selective_prediction.png`, `metrics_vs_baselines.png` (following the global plotting style)
- `docs/scifact_jev_evaluation_methods.md`: the methods write-up (model, questions, splits, thresholds, metrics, limitations, references)

## 6. Cost plan

Price: $0.042 per 1M input tokens (output is free). Estimates are confirmed by the pilot (§4.2).

| Item | Calls | Tokens per call (est.) | Tokens | Cost |
|---|---|---|---|---|
| Rerank, dev | 300 × 30 | ~500 | 4.5M | $0.19 |
| Verify, dev (top 3 + oracle) | ≤ 900 + ≤ 300 | ~2,000 | ≤ 2.4M | ≤ $0.10 |
| Development on train (100 claims, ≤ 2 question versions) | | | ≤ 4.8M | ≤ $0.20 |
| Pilot | 10 | | < 0.05M | < $0.01 |
| **Total** | | | **≈ 12M** | **≈ $0.50** (cap $0.90) |

## 7. Risks and fallbacks

| Risk | Mitigation |
|---|---|
| Actual tokens per call exceed the estimate | Pilot gate; fall back to re-ranking the top 20; reduce train development to 1 question version |
| BM25 recall@30 is low, capping the pipeline | Reported explicitly; the oracle analysis (§5.3) still measures verification |
| Jev's known weakness with literal reading (NEI vs contradict) | Criteria spell out the boundary cases; errors are analysed by confusion cell |
| Published baselines are not on a comparable split | Labelled as context only, never presented as head-to-head |
| Held-out leakage through iteration | Dev is frozen; the ledger records the question version per call; any pre-freeze dev call is flagged |

## 8. Testing

- Unit tests (`tests/`, pytest) cover:
  - the state builder never includes gold fields;
  - the budget guard raises before exceeding the cap;
  - the cache returns hits without calling the API;
  - `04_assemble_predictions` output validates against the SciFact format;
  - the vendored metrics reproduce the worked example in SciFact's `doc/evaluation.md` (abstract F1 = 1/2, sentence F1 = 2/9).
- Tests use a fake client; no paid API calls.

## 9. Out of scope

- The SciFact test split (hidden labels) and any leaderboard submission.
- LLM comparators (no API key; the user chose published systems instead).
- Other datasets (SYNERGY, NCBI-disease, feature extraction). These are candidates for later, reusing `scripts/lib/`.
- Tuning BM25.

## 10. Acceptance criteria

1. The full pipeline runs end to end on dev from cached inputs with `$0` of new spend on re-run.
2. Total spend in the ledger ≤ $0.90.
3. All tables and figures in §5.6 exist, and every metric has a 95% CI.
4. The methods doc exists, states all limitations, and matches the code (question version, thresholds, model ID).
5. All unit tests pass.

## References

- Wadden D. et al. (2020). Fact or Fiction: Verifying Scientific Claims. *EMNLP*. arXiv:2004.14974.
- Wadden D. et al. (2022). MultiVerS: Improving scientific claim verification with weak supervision and full-document context. *Findings of NAACL*. arXiv:2112.01640.
- Robertson S., Zaragoza H. (2009). The Probabilistic Relevance Framework: BM25 and Beyond. *Found. Trends Inf. Retr.* 3(4).
- TypeSafe documentation, https://docs.typesafe.ai (Score, Choice, Noul, Confidence, Re-ranking cookbook, Jev 1.13 jaggedness).
