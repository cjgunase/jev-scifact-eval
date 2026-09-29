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
