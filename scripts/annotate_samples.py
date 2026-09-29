"""Annotate GEO-style sample records with Jev (TypeSafe System One).

For each sample, one request asks five independent questions in parallel:
tissue, sample condition, assay (Choice) and cell-line / treated (Noul).
Raw API answers are cached so evaluation can be re-run without re-querying.

Usage:
    uv run python scripts/annotate_samples.py
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from typesafe_sdk import AsyncTypeSafeClient, Choice, Noul

ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH = ROOT / "data" / "raw" / "geo_style_samples.json"
OUTPUT_PATH = ROOT / "data" / "processed" / "jev_answers.json"

MODEL = "jev-latest"
MAX_CONCURRENT_REQUESTS = 8

# Fields that are shown to the model. `expected` and `ambiguous` are ground
# truth and must never leak into the state.
STATE_FIELDS = ("title", "source_name", "characteristics", "protocol")

QUESTIONS = {
    "tissue": Choice(
        instructions=(
            "Which organ or tissue of origin does this sequencing sample "
            "(`sample`) come from? For cell lines, organoids and derived cells, "
            "answer the organ the cells originate from or model."
        ),
        criteria={
            "liver": None,
            "lung": "Lung tissue or airway samples such as bronchoalveolar lavage",
            "breast": None,
            "colon": "Colon, rectum or large intestine",
            "brain": "Brain or central nervous system, including neurons and glia",
            "blood": "Blood, bone marrow, and immune or haematopoietic cells",
            "skin": "Skin, including melanocytes and keratinocytes",
            "kidney": None,
            "other_or_unclear": "Another organ, or the record does not say",
        },
    ),
    "condition": Choice(
        instructions="What is the disease status of the tissue or cells in `sample`?",
        criteria={
            "healthy_control": "Normal tissue or cells from a donor without the studied disease",
            "tumor": "Cancer tissue, cancer cells, or a cancer-derived cell line",
            "tumor_adjacent_normal": "Histologically normal tissue taken next to a tumour from a cancer patient",
            "non_cancer_disease": "Tissue or cells from a patient with a non-cancer disease",
            "unclear": "The record does not say enough to tell",
        },
    ),
    "assay": Choice(
        instructions="Which molecular assay produced the data for `sample`?",
        criteria={
            "bulk_rna_seq": "Bulk transcriptome sequencing (mRNA-seq, total RNA-seq, SMART-seq of a population)",
            "single_cell_rna_seq": "Single-cell or single-nucleus RNA sequencing, including CITE-seq",
            "dna_methylation": "Bisulfite sequencing or methylation arrays (450K, EPIC)",
            "chip_seq": "Protein-DNA profiling: ChIP-seq, CUT&RUN, CUT&Tag",
            "atac_seq": "Chromatin accessibility: ATAC-seq, DNase-seq",
            "wgs_wes": "Whole-genome or whole-exome DNA sequencing",
            "other": "A different assay (e.g. expression microarray) or not stated",
        },
    ),
    "cell_line": Noul(
        instructions=(
            "Are the cells in `sample` from an established or derived cell line "
            "(including iPSC-derived cells), rather than primary tissue, primary "
            "cells or organoids taken directly from a donor?"
        ),
    ),
    "treated": Noul(
        instructions=(
            "Was `sample` experimentally exposed to a drug, stimulus, or genetic "
            "perturbation (knockdown, CRISPR, overexpression) before profiling? "
            "Vehicle-only controls and patient-level clinical history do not count."
        ),
    ),
}


def build_state(record: dict) -> dict:
    return {"sample": {field: record[field] for field in STATE_FIELDS}}


async def annotate_one(client: AsyncTypeSafeClient, record: dict, sem: asyncio.Semaphore) -> dict:
    async with sem:
        response = await client.system_one(
            model=MODEL,
            state=build_state(record),
            questions=QUESTIONS,
        )
    answers = {}
    for qid, ans in response.answers.items():
        answers[qid] = ans.model_dump() if hasattr(ans, "model_dump") else dict(ans)
    return {
        "sample_id": record["sample_id"],
        "model": response.model,
        "answers": answers,
        "usage": response.usage.model_dump() if hasattr(response.usage, "model_dump") else None,
    }


async def main() -> None:
    records = json.loads(INPUT_PATH.read_text())
    sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    async with AsyncTypeSafeClient() as client:
        results = await asyncio.gather(*(annotate_one(client, r, sem) for r in records))
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"Annotated {len(results)} samples with {results[0]['model']} -> {OUTPUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    asyncio.run(main())
