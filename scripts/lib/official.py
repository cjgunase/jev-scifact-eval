"""Adapter from our plain-dict predictions and gold data onto the vendored official SciFact metrics."""

from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

from lib.scifact_official.data import EvidenceAbstract, PredictedAbstract, make_label
from lib.scifact_official.metrics import compute_f1, update_counts_abstract, update_counts_sentence


def _gold_obj(gold_evidence: dict[int, tuple[str, list[list[int]]]]) -> SimpleNamespace:
    return SimpleNamespace(evidence={
        int(doc_id): EvidenceAbstract(int(doc_id), make_label(label), rationales)
        for doc_id, (label, rationales) in gold_evidence.items()
    })


def _pred_obj(pred: dict) -> SimpleNamespace:
    return SimpleNamespace(predictions={
        int(doc_id): PredictedAbstract(int(doc_id), make_label(p["label"], allow_NEI=False), list(p["sentences"]))
        for doc_id, p in pred["evidence"].items()
    })


def claim_counts(pred: dict, gold_evidence: dict[int, tuple[str, list[list[int]]]]) -> Counter:
    """Official counts for one claim, with keys prefixed `_abstract` / `_sentence`."""
    gold, p = _gold_obj(gold_evidence), _pred_obj(pred)
    a = update_counts_abstract(p, gold, Counter())
    s = update_counts_sentence(p, gold, Counter())
    out = Counter({f"{k}_abstract": v for k, v in a.items()})
    out.update({f"{k}_sentence": v for k, v in s.items()})
    # Make sure the zero-valued keys exist so that downstream code can read them.
    for k in ("relevant_abstract", "retrieved_abstract", "relevant_sentence", "retrieved_sentence"):
        out.setdefault(k, 0)
    return out


def metrics_from_counts(total: Counter) -> dict[str, dict[str, float]]:
    a = Counter({k.removesuffix("_abstract"): v for k, v in total.items() if k.endswith("_abstract")})
    s = Counter({k.removesuffix("_sentence"): v for k, v in total.items() if k.endswith("_sentence")})
    return {
        "sentence_selection": compute_f1(s, "selection"),
        "sentence_label": compute_f1(s, "label"),
        "abstract_label_only": compute_f1(a, "label_only"),
        "abstract_rationalized": compute_f1(a, "rationalized"),
    }
