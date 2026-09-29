import importlib

import pytest

from lib.evalstats import mrr, recall_at_k, tokenize

bm25_mod = importlib.import_module("01_bm25_retrieve")


def test_tokenize_lowercases_and_drops_stopwords():
    assert tokenize("The p53 protein IS mutated in Cancer.") == ["p53", "protein", "mutated", "cancer"]


def test_retrieve_ranks_matching_doc_first():
    from lib.data import Doc
    corpus = {1: Doc(1, "Aspirin and stroke", ("Aspirin lowers stroke risk.",)),
              2: Doc(2, "Coffee", ("Coffee and sleep.",)),
              3: Doc(3, "Tea", ("Tea polyphenols.",))}
    index = bm25_mod.build_index(corpus)
    ids, _ = bm25_mod.retrieve(index, "does aspirin reduce stroke", k=2)
    assert ids[0] == 1 and len(ids) == 2


def test_recall_and_mrr_micro_over_gold_pairs():
    ranked = {1: [5, 6, 7], 2: [9, 8, 4]}
    gold = {1: {6}, 2: {4, 10}}
    assert recall_at_k(ranked, gold, 1) == pytest.approx(0 / 3)
    assert recall_at_k(ranked, gold, 3) == pytest.approx(2 / 3)
    assert mrr(ranked, gold) == pytest.approx((1 / 2 + 1 / 3) / 2)
