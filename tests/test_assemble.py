import importlib

asm = importlib.import_module("04_assemble_predictions")
from lib.data import Claim


def row(cid, did, verdict, conf, probs, oracle=False):
    return {"claim_id": cid, "doc_id": did, "verdict": verdict, "verdict_confidence": conf,
            "evidence_probs": probs, "is_oracle": oracle}


ROWS = [
    row(1, 10, "supports", 0.9, [0.1, 0.8, 0.6, 0.95]),
    row(1, 11, "not_enough_info", 0.9, [0.9]),
    row(1, 12, "contradicts", 0.3, [0.9]),
    row(1, 13, "supports", 0.99, [0.99], oracle=True),
    row(2, 20, "contradicts", 0.7, [0.2, 0.1]),
]


def test_every_claim_gets_a_line():
    preds = asm.assemble(ROWS, [1, 2, 3], tau_verdict=0.5, tau_sentence=0.5)
    assert [p["id"] for p in preds] == [1, 2, 3]
    assert preds[2] == {"id": 3, "evidence": {}}


def test_filters_nei_low_confidence_and_oracle_and_orders_sentences():
    p1 = asm.assemble(ROWS, [1], tau_verdict=0.5, tau_sentence=0.5)[0]
    assert list(p1["evidence"]) == ["10"]
    assert p1["evidence"]["10"] == {"sentences": [3, 1, 2], "label": "SUPPORT"}


def test_falls_back_to_best_sentence_when_none_pass():
    p2 = asm.assemble(ROWS, [2], tau_verdict=0.5, tau_sentence=0.5)[0]
    assert p2["evidence"]["20"] == {"sentences": [0], "label": "CONTRADICT"}


def test_prediction_keys_are_str_ints():
    for p in asm.assemble(ROWS, [1, 2], 0.0, 0.0):
        assert all(isinstance(k, str) and k.isdigit() for k in p["evidence"])


def test_select_thresholds_prefers_the_grid_point_with_best_f1():
    claims = [Claim(1, "c", {10: ("SUPPORT", [[3]])}), Claim(2, "c", {})]
    out = asm.select_thresholds(ROWS, claims)
    assert out["train_metrics"]["abstract_rationalized"]["f1"] == 1.0
    assert out["tau_verdict"] > 0.7  # must exclude claim 2's wrong 0.7-confidence prediction
