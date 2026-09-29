"""Plain-language, one-image summary of the held-out SciFact results (for talks / social posts).

Every number is read from results/tables/*_dev.csv, so the figure stays in sync with the analysis.

Usage: uv run python scripts/06_summary_figure.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lib.budget import LEDGER_PATH
from lib.data import ROOT

TABLES, FIGS = ROOT / "results" / "tables", ROOT / "results" / "figures"
RED, GREY, PURPLE, DARK = "#e74c3c", "#bdc3c7", "#9b59b6", "#333"

plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "axes.spines.top": False,
                     "axes.spines.right": False, "font.family": "sans-serif"})


def load_numbers() -> dict:
    retr = pd.read_csv(TABLES / "retrieval_dev.csv").set_index(["ranker", "metric"])
    ver = pd.read_csv(TABLES / "verification_oracle_dev.csv").set_index("metric")
    off = pd.read_csv(TABLES / "official_metrics_dev.csv").set_index(["system", "metric"])
    pub = pd.read_csv(TABLES / "published_scifact.csv").set_index(["system", "split", "metric"])
    sel = pd.read_csv(TABLES / "selective_prediction_dev.csv")
    half = sel.iloc[(sel["coverage"] - 0.5).abs().argmin()]
    ledger = pd.read_csv(LEDGER_PATH)

    def ci(df, key, v="value"):
        r = df.loc[key]
        return float(r[v]), float(r["lo"] if "lo" in r else r["f1_lo"]), float(r["hi"] if "hi" in r else r["f1_hi"])

    return {
        "r1_bm25": ci(retr, ("BM25", "R@1")),
        "r1_jev": ci(retr, ("Jev re-rank", "R@1")),
        "verdict": ci(ver, "verdict_accuracy_gold_pairs"),
        "nei": ci(ver, "nei_accuracy"),
        "f1_jev": ci(off, ("Jev pipeline", "abstract_rationalized"), v="f1"),
        "f1_lex": ci(off, ("B2 lexical", "abstract_rationalized"), v="f1"),
        "f1_verisci": float(pub.loc[("VeriSci", "dev", "abstract_rationalized"), "f1"]),
        "acc_all": float(sel.iloc[0]["accuracy"]),
        "acc_half": float(half["accuracy"]),
        "cov_half": float(half["coverage"]),
        "n_calls": len(ledger),
        "cost": float(ledger["cost_usd"].sum()),
    }


def bars(ax, labels, values, colours, title, subtitle, cis=None, pct=True):
    y = np.arange(len(labels))[::-1]
    ax.barh(y, values, color=colours, edgecolor="white", height=0.6)
    if cis is not None:
        lo = [v - c[0] if c else 0 for v, c in zip(values, cis)]
        hi = [c[1] - v if c else 0 for v, c in zip(values, cis)]
        ax.errorbar(values, y, xerr=[lo, hi], fmt="none", ecolor=DARK, lw=1, capsize=3)
    for yi, v, c in zip(y, values, cis or [None] * len(values)):
        x = (c[1] if c else v) + 0.02
        ax.text(x, yi, f"{v:.0%}" if pct else f"{v:.2f}", va="center", fontsize=13, fontweight="bold", color=DARK)
    ax.set_yticks(y, labels, fontsize=12)
    ax.set_xlim(0, 1.12)
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title(f"{title}\n", fontsize=15, fontweight="bold", loc="left", pad=6)
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=11, color="#555", va="bottom")


def main() -> None:
    n = load_numbers()
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    fig.suptitle("Can a small, fast AI model check scientific claims?", fontsize=20, fontweight="bold", x=0.02, ha="left")
    fig.text(0.02, 0.935, "TypeSafe Jev (jev-1.13, no fine-tuning) on SciFact: 300 expert-labelled biomedical claims "
             "vs 5,183 PubMed abstracts, held out and run once", fontsize=12, color="#555", ha="left")

    bars(axes[0, 0], ["Keyword search", "Keyword search\n+ Jev re-ranking"],
         [n["r1_bm25"][0], n["r1_jev"][0]], [GREY, RED],
         "1. Finding the right paper", "Correct abstract ranked #1",
         cis=[n["r1_bm25"][1:], n["r1_jev"][1:]])
    bars(axes[0, 1], ["Right paper:\nsupports or contradicts?", "Unrelated paper:\nsays 'not enough info'?"],
         [n["verdict"][0], n["nei"][0]], [RED, "#e67e22"],
         "2. Reading the paper", "How often Jev's verdict is correct",
         cis=[n["verdict"][1:], n["nei"][1:]])
    bars(axes[1, 0], ["Keyword baseline", "VeriSci (2020, fine-tuned)", "Jev pipeline"],
         [n["f1_lex"][0], n["f1_verisci"], n["f1_jev"][0]], [GREY, PURPLE, RED],
         "3. Whole task, official SciFact score", "F1: right paper + right verdict + right evidence (0–1)",
         cis=[n["f1_lex"][1:], None, n["f1_jev"][1:]], pct=False)
    bars(axes[1, 1], ["All answers", f"Most confident {n['cov_half']:.0%}"],
         [n["acc_all"], n["acc_half"]], [GREY, RED],
         "4. Does its confidence mean something?", "Verdict accuracy when keeping only confident answers")

    fig.text(0.02, 0.015,
             f"Lines = 95% bootstrap CI.  Cost: ${n['cost']:.2f} for {n['n_calls']:,} API calls.  "
             "Newer fine-tuned models score higher (e.g. MultiVerS, 2022).\n"
             "Jev's per-sentence evidence probabilities were overconfident, so treat them as a ranking, not literal probabilities.",
             fontsize=10.5, color="#555", ha="left")
    plt.tight_layout(rect=(0, 0.05, 1, 0.93), h_pad=3)
    out = FIGS / "summary_for_social.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
