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
