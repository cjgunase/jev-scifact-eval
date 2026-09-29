# jev-test

Hands-on evaluation of TypeSafe's Jev model for bioinformatics metadata tasks.
Use the TypeSafe skill (`typesafe:typesafe-ai`) and read the live docs before changing API code.

- API key: `TYPESAFE_API_KEY` is exported in `~/.zshrc`, so run API scripts through `zsh -ic '...'`.
- Environment: `uv` (the venv runs Python 3.14). Run scripts with `uv run python scripts/<name>.py`. Tests: `uv run pytest -q`.

## Pipeline
1. `scripts/annotate_samples.py`: calls Jev once per sample and caches the answers in `data/processed/jev_answers.json`
2. `scripts/evaluate_annotations.py`: compares answers with the expected labels and writes `results/tables/*.csv` and `results/figures/jev_annotation_overview.png`

Methods: `docs/jev_sample_annotation_methods.md`

## SciFact evaluation

Evaluates Jev on SciFact claim verification. Spec: `docs/superpowers/specs/2026-09-29-scifact-jev-evaluation-design.md`; results and methods: `docs/scifact_jev_evaluation_methods.md`.

| Stage | Command | Cost |
|---|---|---|
| Data | `uv run python scripts/fetch_scifact.py` | free |
| 00 pilot | `zsh -ic 'uv run python scripts/00_pilot.py'` | paid |
| 01 BM25 | `uv run python scripts/01_bm25_retrieve.py --split {train,dev}` | free |
| 02 re-rank | `zsh -ic 'uv run python scripts/02_jev_rerank.py --split {train,dev}'` | paid (cached) |
| 03 verify | `zsh -ic 'uv run python scripts/03_jev_verify.py --split {train,dev}'` | paid (cached) |
| 04 assemble | `uv run python scripts/04_assemble_predictions.py --split {train,dev}` | free |
| 05 evaluate | `uv run python scripts/05_evaluate.py --split {train,dev}` | free |

- **Budget:** every paid call is logged in `data/processed/scifact/spend_ledger.csv`, and the hard cap is **$0.90** (`lib/budget.py`). Responses are cached in `data/processed/scifact/jev_cache/` (gitignored), so reruns cost $0.
- **Freeze:** questions and thresholds are frozen at git tag `questions-frozen`. Never rerun dev with changed questions or thresholds. Any new wording needs a new `QUESTION_VERSION` and a new train-only development round, and must be reported as a separate experiment.
- **Chaining:** when chaining stages through a pipe, use `set -o pipefail` so that a failed stage stops the chain.
