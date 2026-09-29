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
