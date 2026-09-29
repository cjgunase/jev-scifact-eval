# jev-test

Hands-on evaluation of TypeSafe's Jev model for bioinformatics metadata tasks.
Use the TypeSafe skill (`typesafe:typesafe-ai`) and read the live docs before changing API code.

- API key: `TYPESAFE_API_KEY` is exported in `~/.zshrc`, so run API scripts through `zsh -ic '...'`.
- Environment: `uv` (Python 3.12). Run scripts with `uv run python scripts/<name>.py`.

## Pipeline
1. `scripts/annotate_samples.py`: calls Jev once per sample and caches the answers in `data/processed/jev_answers.json`
2. `scripts/evaluate_annotations.py`: compares answers with the expected labels and writes `results/tables/*.csv` and `results/figures/jev_annotation_overview.png`

Methods: `docs/jev_sample_annotation_methods.md`
