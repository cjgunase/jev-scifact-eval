"""Compare cached Jev answers with hand-assigned labels and plot the results.

Inputs:  data/raw/geo_style_samples.json, data/processed/jev_answers.json
Outputs: results/tables/annotation_long.csv   one row per sample x question
         results/tables/accuracy_summary.csv  accuracy by question and ambiguity
         results/figures/jev_annotation_overview.png

Usage:
    uv run python scripts/evaluate_annotations.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
SAMPLES_PATH = ROOT / "data" / "raw" / "geo_style_samples.json"
ANSWERS_PATH = ROOT / "data" / "processed" / "jev_answers.json"
TABLES_DIR = ROOT / "results" / "tables"
FIGURES_DIR = ROOT / "results" / "figures"

# A Noul probability at or above this is read as "yes". 0.5 is the neutral
# decision point for a calibrated binary probability; no tuning was done.
NOUL_THRESHOLD = 0.5
QUESTION_ORDER = ["tissue", "condition", "assay", "cell_line", "treated"]

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.family": "sans-serif",
})
RED, BLUE, GREY, DARK = "#e74c3c", "#3498db", "#bdc3c7", "#333"


def build_long_table(samples: list[dict], answers: list[dict]) -> pd.DataFrame:
    by_id = {a["sample_id"]: a["answers"] for a in answers}
    missing = {s["sample_id"] for s in samples} ^ set(by_id)
    if missing:
        raise ValueError(f"Sample IDs not matched between inputs: {sorted(missing)}")

    rows = []
    for s in samples:
        for q in QUESTION_ORDER:
            expected = s["expected"][q]
            ans = by_id[s["sample_id"]][q]
            if ans["type"] == "choice":
                predicted = ans["choice"]
                p_expected = ans["probabilities"][expected]
                confidence = ans["confidence"]
            else:
                p_yes = ans["noul"]
                predicted = p_yes >= NOUL_THRESHOLD
                p_expected = p_yes if expected else 1 - p_yes
                # Nouls have no separate confidence; distance from 0.5 rescaled to [0, 1].
                confidence = abs(p_yes - 0.5) * 2
            rows.append({
                "sample_id": s["sample_id"],
                "title": s["title"],
                "ambiguous": s["ambiguous"],
                "question": q,
                "expected": expected,
                "predicted": predicted,
                "correct": predicted == expected,
                "p_expected": round(p_expected, 4),
                "confidence": round(confidence, 4),
            })
    return pd.DataFrame(rows)


def summarise(long_df: pd.DataFrame) -> pd.DataFrame:
    grp = long_df.assign(set=np.where(long_df["ambiguous"], "ambiguous", "clear"))
    summary = (
        grp.groupby(["question", "set"])["correct"].agg(n="size", n_correct="sum").reset_index()
    )
    overall = grp.groupby("question")["correct"].agg(n="size", n_correct="sum").reset_index()
    overall["set"] = "all"
    summary = pd.concat([summary, overall], ignore_index=True)
    summary["accuracy"] = (summary["n_correct"] / summary["n"]).round(3)
    summary["question"] = pd.Categorical(summary["question"], QUESTION_ORDER, ordered=True)
    return summary.sort_values(["question", "set"]).reset_index(drop=True)


def plot_overview(long_df: pd.DataFrame, out_path: Path) -> None:
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(20, 9), gridspec_kw={"width_ratios": [1.15, 1]})

    # Panel A: probability Jev assigned to the expected label, per sample x question.
    ax = ax_a
    mat = long_df.pivot(index="sample_id", columns="question", values="p_expected")[QUESTION_ORDER]
    titles = long_df.drop_duplicates("sample_id").set_index("sample_id")
    labels = [f"{sid}  {titles.loc[sid, 'title']}" for sid in mat.index]
    cmap = LinearSegmentedColormap.from_list("rg", ["#c0392b", "#f39c12", "#f1c40f", "#2ecc71"])
    im = ax.imshow(mat.values, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(len(QUESTION_ORDER)), QUESTION_ORDER, fontsize=11, rotation=0)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    for tick, sid in zip(ax.get_yticklabels(), mat.index):
        if titles.loc[sid, "ambiguous"]:
            tick.set_color(BLUE)
            tick.set_fontweight("bold")
    wrong = long_df.pivot(index="sample_id", columns="question", values="correct")[QUESTION_ORDER]
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                    color=DARK, fontweight="bold" if not wrong.values[i, j] else "normal")
            if not wrong.values[i, j]:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="black", lw=1.6))
    ax.spines[["left", "bottom"]].set_visible(False)
    ax.tick_params(length=0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("P(expected label)", fontsize=12, fontweight="bold")
    ax.set_title("A. Probability Jev gave to the expected label\n"
                 "(boxed = wrong top answer; blue sample = deliberately ambiguous)",
                 fontsize=15, fontweight="bold", pad=12)

    # Panel B: confidence vs correctness for Choice questions.
    ax = ax_b
    choice_df = long_df[long_df["question"].isin(["tissue", "condition", "assay"])]
    rng = np.random.default_rng(0)  # jitter only; does not affect any statistic
    for k, (flag, colour, name) in enumerate([(True, BLUE, "correct"), (False, RED, "wrong")]):
        sub = choice_df[choice_df["correct"] == flag]
        x = sub["question"].map({"tissue": 0, "condition": 1, "assay": 2}) + rng.uniform(-0.18, 0.18, len(sub))
        ax.scatter(x, sub["confidence"], s=70 if not flag else 45, alpha=0.85,
                   c=colour, edgecolors="white" if flag else "black", linewidths=0.8,
                   label=f"{name} (n={len(sub)})", zorder=3)
    for _, r in choice_df[~choice_df["correct"]].iterrows():
        xi = {"tissue": 0, "condition": 1, "assay": 2}[r["question"]]
        ax.annotate(f"{r['sample_id']}: {r['predicted']}\n(expected {r['expected']})",
                    xy=(xi, r["confidence"]), xytext=(xi + 0.28, r["confidence"] + 0.03),
                    fontsize=9, fontweight="bold", color=DARK,
                    arrowprops=dict(arrowstyle="->", color=RED, lw=1))
    ax.set_xticks([0, 1, 2], ["tissue", "condition", "assay"], fontsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_xlim(-0.5, 2.9)
    ax.set_ylim(-0.03, 1.08)
    ax.set_xlabel("Choice question", fontsize=15, fontweight="bold")
    ax.set_ylabel("Jev confidence", fontsize=15, fontweight="bold")
    ax.legend(fontsize=11, frameon=True, framealpha=0.9, loc="lower left")
    ax.set_title("B. Are the wrong answers the low-confidence ones?",
                 fontsize=15, fontweight="bold", pad=12)

    plt.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    samples = json.loads(SAMPLES_PATH.read_text())
    answers = json.loads(ANSWERS_PATH.read_text())
    long_df = build_long_table(samples, answers)
    summary = summarise(long_df)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    long_df.to_csv(TABLES_DIR / "annotation_long.csv", index=False)
    summary.to_csv(TABLES_DIR / "accuracy_summary.csv", index=False)
    plot_overview(long_df, FIGURES_DIR / "jev_annotation_overview.png")

    print(summary.pivot(index="question", columns="set", values="accuracy").to_string())
    print("\nDisagreements with expected labels:")
    cols = ["sample_id", "title", "question", "expected", "predicted", "p_expected", "confidence"]
    print(long_df.loc[~long_df["correct"], cols].to_string(index=False))


if __name__ == "__main__":
    main()
