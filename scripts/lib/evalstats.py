"""Evaluation statistics: tokenisation, retrieval metrics, bootstrap, calibration."""

from __future__ import annotations

import re

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from lib.data import SEED


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in ENGLISH_STOP_WORDS]


def recall_at_k(ranked: dict[int, list[int]], gold: dict[int, set[int]], k: int) -> float:
    """Micro recall over (claim, gold abstract) pairs. Claims without gold are ignored."""
    hits = total = 0
    for cid, g in gold.items():
        if not g:
            continue
        top = set(ranked.get(cid, [])[:k])
        hits += len(g & top)
        total += len(g)
    return hits / total if total else float("nan")


def mrr(ranked: dict[int, list[int]], gold: dict[int, set[int]]) -> float:
    """Mean reciprocal rank of the first gold abstract per claim (0 if none were retrieved)."""
    rr = []
    for cid, g in gold.items():
        if not g:
            continue
        ranks = [i for i, d in enumerate(ranked.get(cid, []), start=1) if d in g]
        rr.append(1 / ranks[0] if ranks else 0.0)
    return float(np.mean(rr)) if rr else float("nan")
