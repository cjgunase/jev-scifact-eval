"""Stage 5: official metrics with bootstrap CIs, retrieval, oracle verification, calibration, figures.

Usage: uv run python scripts/05_evaluate.py --split {train,dev}
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from lib.baselines import lexical_baseline, majority_baseline
from lib.budget import LEDGER_PATH
from lib.data import ROOT, load_corpus, load_split, read_jsonl, split_dir
from lib.evalstats import bootstrap_ci, ece, mrr, recall_at_k, reliability, selective_curve
from lib.official import claim_counts, metrics_from_counts

TABLES, FIGS = ROOT / "results" / "tables", ROOT / "results" / "figures"
METRICS = ["abstract_label_only", "abstract_rationalized", "sentence_selection", "sentence_label"]
GOLD_TO_VERDICT = {"SUPPORT": "supports", "CONTRADICT": "contradicts"}
RED, BLUE, GREEN, PURPLE, ORANGE, GREY = "#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#e67e22", "#bdc3c7"

plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "axes.spines.top": False,
                     "axes.spines.right": False, "font.family": "sans-serif"})


def official_with_ci(preds, gold) -> pd.DataFrame:
    per_claim = [claim_counts(p, gold[p["id"]]) for p in preds]
    point = metrics_from_counts(sum(per_claim, Counter()))
    rows = []
    for m in METRICS:
        lo, hi = bootstrap_ci(per_claim, lambda xs, m=m: metrics_from_counts(sum(xs, Counter()))[m]["f1"])
        rows.append({"metric": m, **point[m], "f1_lo": lo, "f1_hi": hi})
    return pd.DataFrame(rows)


def verdict_table(verify_rows, claims) -> pd.DataFrame:
    """Gold pairs (claim, evidence doc), plus NEI claims paired with their re-ranked top-1."""
    by_id = {c.claim_id: c for c in claims}
    out = []
    for r in verify_rows:
        c = by_id[r["claim_id"]]
        if r["doc_id"] in c.evidence:
            gold = GOLD_TO_VERDICT[c.evidence[r["doc_id"]][0]]
        elif not c.evidence and r["rerank_position"] == 0:
            gold = "not_enough_info"
        else:
            continue
        out.append({"claim_id": r["claim_id"], "doc_id": r["doc_id"], "gold": gold, "pred": r["verdict"],
                    "confidence": r["verdict_confidence"], "p_max": max(r["verdict_probs"].values())})
    df = pd.DataFrame(out)
    df["correct"] = (df["gold"] == df["pred"]).astype(int)
    return df


def sentence_table(verify_rows, claims) -> pd.DataFrame:
    by_id = {c.claim_id: c for c in claims}
    out = []
    for r in verify_rows:
        ev = by_id[r["claim_id"]].evidence.get(r["doc_id"])
        if not ev:
            continue
        gold_idx = {i for rat in ev[1] for i in rat}
        out += [{"claim_id": r["claim_id"], "p": p, "gold": int(i in gold_idx)} for i, p in enumerate(r["evidence_probs"])]
    return pd.DataFrame(out)


def fig_retrieval(bm25_rank, jev_rank, gold, path):
    ks = list(range(1, 31))
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.plot(ks, [recall_at_k(bm25_rank, gold, k) for k in ks], color=GREY, lw=2.5, label="BM25")
    ax.plot(ks, [recall_at_k(jev_rank, gold, k) for k in ks], color=RED, lw=2.5, label="BM25 → Jev re-rank")
    ax.axvline(3, color="grey", ls="--", lw=1, alpha=0.6)
    ax.set_xlabel("k (abstracts kept)", fontsize=15, fontweight="bold")
    ax.set_ylabel("Recall@k of gold abstracts", fontsize=15, fontweight="bold")
    ax.set_title("Retrieval: does Jev re-ranking beat BM25?", fontsize=16, fontweight="bold", pad=12)
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=12, frameon=True, framealpha=0.9)
    plt.tight_layout(); fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)


def fig_reliability(table, title, xlabel, path, colour):
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.plot([0, 1], [0, 1], color="black", lw=1, alpha=0.4)
    t = table.dropna()
    ax.scatter(t["mean_prob"], t["frac_pos"], s=40 + 400 * t["n"] / t["n"].max(), c=colour, alpha=0.85,
               edgecolors="white", linewidths=1, zorder=3)
    ax.plot(t["mean_prob"], t["frac_pos"], color=colour, lw=1.5)
    for _, r in t.iterrows():
        ax.annotate(f"n={int(r['n'])}", (r["mean_prob"], r["frac_pos"]), textcoords="offset points",
                    xytext=(6, -12), fontsize=9, fontweight="bold", color="#333")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel(xlabel, fontsize=15, fontweight="bold")
    ax.set_ylabel("Observed frequency correct / positive", fontsize=15, fontweight="bold")
    ax.set_title(title, fontsize=16, fontweight="bold", pad=12)
    ax.tick_params(labelsize=12)
    plt.tight_layout(); fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)


def fig_selective(curve, path):
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.plot(curve["coverage"], curve["accuracy"], color=BLUE, lw=2.5)
    ax.set_xlim(1.02, 0)
    ax.set_xlabel("Coverage (fraction of verdicts kept, most confident first)", fontsize=15, fontweight="bold")
    ax.set_ylabel("Verdict accuracy", fontsize=15, fontweight="bold")
    ax.set_title("Can Jev's confidence drive abstention?", fontsize=16, fontweight="bold", pad=12)
    ax.tick_params(labelsize=12)
    plt.tight_layout(); fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)


def fig_metrics(df, path):
    systems = list(df["system"].unique())
    colours = dict(zip(systems, [RED, GREY, "#7f8c8d", PURPLE, GREEN, ORANGE]))
    fig, axes = plt.subplots(1, len(METRICS), figsize=(20, 8), sharey=True)
    for ax, m in zip(axes, METRICS):
        sub = df[df["metric"] == m].set_index("system").reindex(systems)
        y = np.arange(len(systems))
        err = np.vstack([sub["f1"] - sub["f1_lo"], sub["f1_hi"] - sub["f1"]])
        ax.barh(y, sub["f1"], color=[colours[s] for s in systems], edgecolor="white", height=0.7,
                xerr=np.nan_to_num(err), error_kw=dict(ecolor="#333", lw=1, capsize=3))
        for yi, v, hi in zip(y, sub["f1"], sub["f1_hi"]):
            if not np.isnan(v):
                x = (hi if not np.isnan(hi) else v) + 0.02  # label clears the CI whisker
                ax.text(x, yi, f"{v:.2f}", va="center", fontsize=8.5, color="#333")
        ax.set_yticks(y, systems, fontsize=11)
        ax.set_xlim(0, 1)
        ax.set_title(m.replace("_", " "), fontsize=14, fontweight="bold", pad=10)
        ax.tick_params(labelsize=11)
    axes[0].invert_yaxis()
    fig.supxlabel("F1 (95% bootstrap CI where computed here)", fontsize=15, fontweight="bold")
    plt.tight_layout(); fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "dev"], required=True)
    split = ap.parse_args().split
    sd = split_dir(split)
    TABLES.mkdir(parents=True, exist_ok=True); FIGS.mkdir(parents=True, exist_ok=True)
    corpus, claims = load_corpus(), load_split(split)
    gold_ev = {c.claim_id: c.evidence for c in claims}
    bm25_rows, rerank_rows = read_jsonl(sd / "bm25_top30.jsonl"), read_jsonl(sd / "rerank.jsonl")
    verify_rows, preds = read_jsonl(sd / "verify.jsonl"), read_jsonl(sd / "predictions.jsonl")

    # Official metrics: Jev vs baselines.
    systems = {"Jev pipeline": preds, "B1 majority": majority_baseline(bm25_rows),
               "B2 lexical": lexical_baseline(bm25_rows, corpus, claims)}
    official = pd.concat([official_with_ci(p, gold_ev).assign(system=s) for s, p in systems.items()])
    official.to_csv(TABLES / f"official_metrics_{split}.csv", index=False)

    # Retrieval.
    gold_docs = {c.claim_id: set(c.evidence) for c in claims}
    bm25_rank = {r["claim_id"]: r["doc_ids"] for r in bm25_rows}
    jev_rank = {r["claim_id"]: r["doc_ids"] for r in rerank_rows}
    retr = pd.DataFrame([{"ranker": n, **{f"R@{k}": recall_at_k(rk, gold_docs, k) for k in (1, 3, 10, 30)},
                          "MRR": mrr(rk, gold_docs)} for n, rk in (("BM25", bm25_rank), ("Jev re-rank", jev_rank))])
    retr.to_csv(TABLES / f"retrieval_{split}.csv", index=False)
    fig_retrieval(bm25_rank, jev_rank, gold_docs, FIGS / f"retrieval_recall_at_k_{split}.png")

    # Verification in isolation, plus calibration.
    vt = verdict_table(verify_rows, claims)
    gp = vt[vt["gold"] != "not_enough_info"]
    st = sentence_table(verify_rows, claims)
    tau_s = json.loads((TABLES / "thresholds.json").read_text())["tau_sentence"]
    pred_s = (st["p"] >= tau_s).astype(int)
    tp = int(((pred_s == 1) & (st["gold"] == 1)).sum())
    prec = tp / max(int(pred_s.sum()), 1)
    rec = tp / max(int(st["gold"].sum()), 1)
    oracle = pd.DataFrame([{
        "n_gold_pairs": len(gp),
        "verdict_accuracy_gold_pairs": gp["correct"].mean(),
        "verdict_macro_f1_support_contradict": f1_score(gp["gold"], gp["pred"],
                                                        labels=["supports", "contradicts"], average="macro"),
        "n_nei_pairs": int((vt["gold"] == "not_enough_info").sum()),
        "nei_accuracy": vt.loc[vt["gold"] == "not_enough_info", "correct"].mean(),
        "evidence_precision": prec, "evidence_recall": rec,
        "evidence_f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "evidence_auroc": roc_auc_score(st["gold"], st["p"]),
        "tau_sentence": tau_s,
    }])
    oracle.to_csv(TABLES / f"verification_oracle_{split}.csv", index=False)
    pd.crosstab(vt["gold"], vt["pred"]).to_csv(TABLES / f"verdict_confusion_{split}.csv")

    rel_v, rel_e = reliability(vt["p_max"], vt["correct"]), reliability(st["p"], st["gold"])
    pd.concat([rel_v.assign(question="verdict (max prob vs correct)"),
               rel_e.assign(question="evidence noul (P vs gold)")]).to_csv(TABLES / f"calibration_{split}.csv", index=False)
    pd.DataFrame([{"question": "verdict", "ece": ece(vt["p_max"], vt["correct"])},
                  {"question": "evidence", "ece": ece(st["p"], st["gold"])}]).to_csv(TABLES / f"ece_{split}.csv", index=False)
    fig_reliability(rel_v, "Verdict calibration", "Top-choice probability", FIGS / f"reliability_verdict_{split}.png", BLUE)
    fig_reliability(rel_e, "Evidence-sentence calibration", "P(evidence)", FIGS / f"reliability_evidence_{split}.png", PURPLE)
    curve = selective_curve(vt["confidence"], vt["correct"])
    curve.to_csv(TABLES / f"selective_prediction_{split}.csv", index=False)
    fig_selective(curve, FIGS / f"selective_prediction_{split}.png")

    # Published systems, as context.
    pub = pd.read_csv(TABLES / "published_scifact.csv")
    pub = pub[pub["metric"].isin(METRICS)].assign(
        system=lambda d: d["system"] + " (" + d["split"] + ", " + d["retrieval_setting"] + ")", f1_lo=np.nan, f1_hi=np.nan)
    fig_metrics(pd.concat([official, pub[["system", "metric", "f1", "f1_lo", "f1_hi"]]]),
                FIGS / f"metrics_vs_baselines_{split}.png")

    # Cost.
    ledger = pd.read_csv(LEDGER_PATH)
    ledger.groupby(["stage", "split", "question_version"])[["input_tokens", "cost_usd"]].sum().reset_index() \
        .to_csv(TABLES / "cost_summary.csv", index=False)

    print(official.pivot(index="system", columns="metric", values="f1").round(3).to_string())
    print(retr.round(3).to_string(index=False))
    print(oracle.round(3).T.to_string())
    print(f"total spend ${ledger['cost_usd'].sum():.4f}")


if __name__ == "__main__":
    main()
