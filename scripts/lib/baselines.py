"""Non-Jev baselines, emitted in SciFact prediction format."""

from __future__ import annotations

from lib.evalstats import tokenize


def majority_baseline(bm25_rows) -> list[dict]:
    """BM25 top-1, always SUPPORT (the majority label), first three sentences as evidence."""
    return [{"id": r["claim_id"], "evidence": {str(r["doc_ids"][0]): {"sentences": [0, 1, 2], "label": "SUPPORT"}}}
            for r in bm25_rows]


def lexical_baseline(bm25_rows, corpus, claims) -> list[dict]:
    """BM25 top-1, SUPPORT, and the three sentences with the most token overlap with the claim."""
    text = {c.claim_id: set(tokenize(c.text)) for c in claims}
    out = []
    for r in bm25_rows:
        doc = corpus[r["doc_ids"][0]]
        overlap = [len(text[r["claim_id"]] & set(tokenize(s))) for s in doc.sentences]
        top = sorted(range(len(overlap)), key=lambda i: (-overlap[i], i))[:3]
        out.append({"id": r["claim_id"], "evidence": {str(doc.doc_id): {"sentences": top, "label": "SUPPORT"}}})
    return out
