from collections import Counter

import pytest

from lib.official import claim_counts, metrics_from_counts

GOLD_52 = {11: ("SUPPORT", [[0, 1], [11]]), 15: ("SUPPORT", [[4]])}
PRED_52 = {
    "id": 52,
    "evidence": {
        "11": {"sentences": [1, 11, 13], "label": "SUPPORT"},
        "16": {"sentences": [18, 20], "label": "CONTRADICT"},
    },
}


def test_worked_example_matches_official_doc():
    m = metrics_from_counts(claim_counts(PRED_52, GOLD_52))
    assert m["abstract_rationalized"]["f1"] == pytest.approx(1 / 2)
    assert m["sentence_label"]["f1"] == pytest.approx(2 / 9)


def test_empty_prediction_scores_zero_but_counts_relevant():
    counts = claim_counts({"id": 52, "evidence": {}}, GOLD_52)
    assert counts["relevant_abstract"] == 2
    assert metrics_from_counts(counts)["abstract_label_only"]["f1"] == 0


def test_counts_add_across_claims():
    total = Counter()
    for _ in range(3):
        total += claim_counts(PRED_52, GOLD_52)
    assert metrics_from_counts(total)["abstract_rationalized"]["f1"] == pytest.approx(1 / 2)
