"""Stage 2: re-rank BM25 candidates with one Jev Score question per (claim, abstract).

Usage: zsh -ic 'uv run python scripts/02_jev_rerank.py --split {train,dev}'
"""

from __future__ import annotations

import argparse

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
