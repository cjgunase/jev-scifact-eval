"""Stage 4: turn Jev answers into SciFact predictions; thresholds chosen on train only.

Usage: uv run python scripts/04_assemble_predictions.py --split {train,dev}
  train: grid-searches thresholds, writes results/tables/thresholds.json, then predictions
  dev:   reads thresholds.json (must exist) and writes predictions
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

import numpy as np

from lib.data import ROOT, load_split, read_jsonl, split_dir, write_jsonl
from lib.official import claim_counts, metrics_from_counts
from lib.questions import QUESTION_VERSION, VERDICT_TO_SCIFACT

THRESHOLDS_PATH = ROOT / "results" / "tables" / "thresholds.json"
GRID = [round(x, 2) for x in np.arange(0.0, 1.0, 0.05)]
SELECTION_METRIC = "abstract_rationalized"


def assemble(verify_rows, claim_ids, tau_verdict: float, tau_sentence: float) -> list[dict]:
    by_claim: dict[int, dict] = {cid: {} for cid in claim_ids}
    for r in verify_rows:
        if r["is_oracle"] or r["claim_id"] not in by_claim:
            continue
        if r["verdict"] not in VERDICT_TO_SCIFACT or r["verdict_confidence"] < tau_verdict:
            continue
        probs = r["evidence_probs"]
        chosen = sorted((i for i, p in enumerate(probs) if p >= tau_sentence), key=lambda i: (-probs[i], i))
        if not chosen:
            chosen = [int(np.argmax(probs))]
        by_claim[r["claim_id"]][str(int(r["doc_id"]))] = {"sentences": chosen, "label": VERDICT_TO_SCIFACT[r["verdict"]]}
    return [{"id": cid, "evidence": ev} for cid, ev in by_claim.items()]


def score(preds, claims) -> dict:
    gold = {c.claim_id: c.evidence for c in claims}
    total = Counter()
    for p in preds:
        total += claim_counts(p, gold[p["id"]])
    return metrics_from_counts(total)


def select_thresholds(verify_rows, claims) -> dict:
    ids = [c.claim_id for c in claims]
    best = None
    for tv in GRID:
        for ts in GRID:
            m = score(assemble(verify_rows, ids, tv, ts), claims)
            key = (m[SELECTION_METRIC]["f1"], m["sentence_label"]["f1"])
            if best is None or key > best[0]:
                best = (key, tv, ts, m)
    return {"tau_verdict": best[1], "tau_sentence": best[2], "train_metrics": best[3],
            "selection_metric": SELECTION_METRIC, "question_version": QUESTION_VERSION}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "dev"], required=True)
    split = ap.parse_args().split
    claims = load_split(split)
    rows = read_jsonl(split_dir(split) / "verify.jsonl")
    if split == "train":
        th = select_thresholds(rows, claims)
        THRESHOLDS_PATH.parent.mkdir(parents=True, exist_ok=True)
        THRESHOLDS_PATH.write_text(json.dumps(th, indent=2))
    else:
        th = json.loads(THRESHOLDS_PATH.read_text())
        if th["question_version"] != QUESTION_VERSION:
            raise SystemExit(f"thresholds were tuned on {th['question_version']}, questions are {QUESTION_VERSION}")
    preds = assemble(rows, [c.claim_id for c in claims], th["tau_verdict"], th["tau_sentence"])
    write_jsonl(split_dir(split) / "predictions.jsonl", preds)
    m = score(preds, claims)
    print(f"tau_verdict={th['tau_verdict']} tau_sentence={th['tau_sentence']} | "
          + " ".join(f"{k}.f1={v['f1']:.3f}" for k, v in m.items()))


if __name__ == "__main__":
    main()
