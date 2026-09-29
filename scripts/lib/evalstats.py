"""Evaluation statistics: tokenisation, retrieval metrics, bootstrap, calibration."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
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


def bootstrap_ci(per_claim: list, stat_fn, n: int = 1000, seed: int = SEED, alpha: float = 0.05):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(per_claim))
    stats = [stat_fn([per_claim[i] for i in rng.choice(idx, size=len(idx), replace=True)]) for _ in range(n)]
    return float(np.quantile(stats, alpha / 2)), float(np.quantile(stats, 1 - alpha / 2))


def reliability(prob, outcome, n_bins: int = 10) -> pd.DataFrame:
    prob, outcome = np.asarray(prob, float), np.asarray(outcome, float)
    edges = np.linspace(0, 1, n_bins + 1)
    bins = np.clip(np.digitize(prob, edges[1:-1], right=True), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = bins == b
        rows.append({"bin_lo": edges[b], "bin_hi": edges[b + 1], "n": int(m.sum()),
                     "mean_prob": float(prob[m].mean()) if m.any() else np.nan,
                     "frac_pos": float(outcome[m].mean()) if m.any() else np.nan})
    return pd.DataFrame(rows)


def ece(prob, outcome, n_bins: int = 10) -> float:
    t = reliability(prob, outcome, n_bins).dropna()
    return float((t["n"] / t["n"].sum() * (t["mean_prob"] - t["frac_pos"]).abs()).sum())


def selective_curve(conf, correct) -> pd.DataFrame:
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    rows = []
    for th in np.unique(np.concatenate([[0.0], conf])):
        m = conf >= th
        if m.any():
            rows.append({"threshold": float(th), "coverage": float(m.mean()), "accuracy": float(correct[m].mean())})
    return pd.DataFrame(rows)
