"""Confusion matrix and ROC curves for a general audience (held-out SciFact dev split).

- Confusion matrix: Jev's verdict vs the expert label, on the pairs used in docs §7.3.
- ROC 1: "Is this paper relevant to the claim?"  Jev relevance Score vs BM25 score, over the
  top-30 BM25 candidates of every claim that has evidence.
- ROC 2: "Is this sentence evidence?"  Jev evidence Noul vs word overlap with the claim, over
  all sentences of gold abstracts.

Usage: uv run python scripts/07_social_figures.py
"""

from __future__ import annotations

import importlib
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import roc_auc_score, roc_curve

from lib.data import ROOT, load_split, read_jsonl, split_dir
from lib.evalstats import tokenize

ev5 = importlib.import_module("05_evaluate")

FIGS, TABLES = ROOT / "results" / "figures", ROOT / "results" / "tables"
RED, GREY, DARK = "#e74c3c", "#7f8c8d", "#333"
ORDER = ["supports", "contradicts", "not_enough_info"]
ROW_LABELS = ["Supports", "Contradicts", "Unrelated paper\n(not enough info)"]
COL_LABELS = ["Supports", "Contradicts", "Not enough info"]

plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "axes.spines.top": False,
                     "axes.spines.right": False, "font.family": "sans-serif"})


def confusion_figure(vt: pd.DataFrame, path) -> None:
    counts = pd.crosstab(vt["gold"], vt["pred"]).reindex(index=ORDER, columns=ORDER, fill_value=0)
    pct = counts.div(counts.sum(axis=1), axis=0)
    cmap = LinearSegmentedColormap.from_list("rg", ["#c0392b", "#f39c12", "#f1c40f", "#2ecc71"])
    fig, ax = plt.subplots(figsize=(11, 8))
    im = ax.imshow(pct.values, cmap=cmap, vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{pct.values[i, j]:.0%}\n(n={counts.values[i, j]})", ha="center", va="center",
                    fontsize=14, fontweight="bold" if i == j else "normal", color=DARK)
    ax.set_xticks(range(3), COL_LABELS, fontsize=13)
    ax.set_yticks(range(3), [f"{l}\n[{n} pairs]" for l, n in zip(ROW_LABELS, counts.sum(axis=1))], fontsize=13)
    ax.set_xlabel("What Jev said", fontsize=15, fontweight="bold", labelpad=10)
    ax.set_ylabel("Expert answer", fontsize=15, fontweight="bold", labelpad=10)
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, format=matplotlib.ticker.PercentFormatter(1.0))
    cbar.set_label("Share of each row", fontsize=12, fontweight="bold")
    acc = (counts.values.diagonal().sum()) / counts.values.sum()
    ax.set_title(f"Does the paper support or contradict the claim?  Jev vs experts (overall {acc:.0%} agree)",
                 fontsize=15, fontweight="bold", pad=18, y=1.08)
    fig.text(0.02, 0.01, "SciFact held-out dev set. Read across a row: of the pairs experts labelled that way, "
             "how Jev answered. Diagonal = agreement.", fontsize=10.5, color="#555")
    plt.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return counts


def roc_panel(ax, y, scores: dict, title: str, subtitle: str) -> dict:
    ax.plot([0, 1], [0, 1], color="black", lw=0.8, alpha=0.35, ls="--")
    aucs = {}
    for (name, s), colour in zip(scores.items(), [RED, GREY]):
        fpr, tpr, _ = roc_curve(y, s)
        aucs[name] = roc_auc_score(y, s)
        ax.plot(fpr, tpr, color=colour, lw=2.8, label=f"{name}  (AUC {aucs[name]:.2f})")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
    ax.set_xlabel("False alarms (share of irrelevant items flagged)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Hits (share of relevant items found)", fontsize=13, fontweight="bold")
    ax.set_title(f"{title}\n", fontsize=15, fontweight="bold", loc="left")
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=11, color="#555", va="bottom")
    ax.tick_params(labelsize=11)
    ax.legend(fontsize=12, frameon=True, framealpha=0.9, loc="lower right")
    return aucs


def main() -> None:
    claims = load_split("dev")
    by_id = {c.claim_id: c for c in claims}
    sd = split_dir("dev")
    verify, rerank, bm25 = (read_jsonl(sd / f) for f in ("verify.jsonl", "rerank.jsonl", "bm25_top30.jsonl"))

    counts = confusion_figure(ev5.verdict_table(verify, claims), FIGS / "confusion_matrix_dev.png")

    # ROC 1: abstract relevance among BM25 top-30 candidates (claims with evidence).
    bm = {r["claim_id"]: dict(zip(r["doc_ids"], r["scores"])) for r in bm25}
    yA, jA, bA = [], [], []
    for r in rerank:
        c = by_id[r["claim_id"]]
        if not c.evidence:
            continue
        for d, s in zip(r["doc_ids"], r["relevance"]):
            yA.append(int(d in c.evidence)); jA.append(s); bA.append(bm[r["claim_id"]][d])

    # ROC 2: evidence sentences within gold abstracts.
    st = ev5.sentence_table(verify, claims)
    from lib.data import load_corpus
    corpus = load_corpus()
    overlap = []
    for r in verify:
        ev = by_id[r["claim_id"]].evidence.get(r["doc_id"])
        if not ev:
            continue
        ct = set(tokenize(by_id[r["claim_id"]].text))
        overlap += [len(ct & set(tokenize(s))) for s in corpus[r["doc_id"]].sentences]
    assert len(overlap) == len(st)

    fig, axes = plt.subplots(1, 2, figsize=(19, 8))
    auc1 = roc_panel(axes[0], yA, {"Jev": jA, "Keyword search (BM25)": bA},
                     "1. Is this paper relevant to the claim?",
                     f"{len(yA):,} candidate papers ({sum(yA)} relevant) for {sum(bool(c.evidence) for c in claims)} claims")
    auc2 = roc_panel(axes[1], st["gold"], {"Jev": st["p"], "Word overlap with claim": overlap},
                     "2. Is this sentence the evidence?",
                     f"{len(st):,} sentences in relevant papers ({int(st['gold'].sum())} marked as evidence by experts)")
    fig.suptitle("Jev vs keyword methods: the curve closer to the top-left corner is better",
                 fontsize=17, fontweight="bold", x=0.01, ha="left")
    fig.text(0.01, -0.02, "ROC curve: every point is one possible cut-off. AUC 1.0 = perfect ranking, 0.5 = coin flip. "
             "SciFact held-out dev set.", fontsize=10.5, color="#555")
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIGS / "roc_curves_dev.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    facts = {"confusion_counts": counts.to_dict(), "roc_abstract_auc": auc1, "roc_sentence_auc": auc2}
    (TABLES / "social_figure_facts_dev.json").write_text(json.dumps(facts, indent=2))
    print(json.dumps({k: v for k, v in facts.items() if k != "confusion_counts"}, indent=2))


if __name__ == "__main__":
    main()
