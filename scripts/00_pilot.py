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
