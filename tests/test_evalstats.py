import numpy as np
import pytest

from lib.baselines import lexical_baseline, majority_baseline
from lib.data import Claim, Doc
from lib.evalstats import bootstrap_ci, ece, reliability, selective_curve


def test_bootstrap_ci_brackets_mean_and_is_deterministic():
    vals = list(np.linspace(0, 1, 50))
    lo, hi = bootstrap_ci(vals, lambda xs: float(np.mean(xs)))
    assert lo < 0.5 < hi
    assert (lo, hi) == bootstrap_ci(vals, lambda xs: float(np.mean(xs)))


def test_ece_zero_when_perfectly_calibrated_and_positive_otherwise():
    p = np.array([0.25] * 4 + [0.75] * 4)
    y = np.array([1, 0, 0, 0, 1, 1, 1, 0])
    assert ece(p, y) == pytest.approx(0.0)
    assert ece(np.array([0.9] * 4), np.array([0, 0, 0, 0])) == pytest.approx(0.9)
    assert reliability(p, y)["n"].sum() == 8


def test_selective_curve_accuracy_rises_when_errors_are_low_confidence():
    conf = np.array([0.9, 0.8, 0.2, 0.1])
    correct = np.array([1, 1, 0, 0])
    curve = selective_curve(conf, correct)
    assert curve.iloc[0]["coverage"] == 1.0 and curve.iloc[0]["accuracy"] == 0.5
    assert curve["accuracy"].max() == 1.0


def test_baselines_predict_top1_support():
    corpus = {1: Doc(1, "t", ("alpha beta", "gamma", "claim words here", "delta"))}
    bm25 = [{"claim_id": 7, "doc_ids": [1], "scores": [1.0]}]
    assert majority_baseline(bm25) == [{"id": 7, "evidence": {"1": {"sentences": [0, 1, 2], "label": "SUPPORT"}}}]
    lex = lexical_baseline(bm25, corpus, [Claim(7, "claim words", {})])
    assert lex[0]["evidence"]["1"]["sentences"][0] == 2


def test_claim_bootstrap_resamples_whole_claims():
    import pandas as pd

    from lib.evalstats import claim_bootstrap_ci

    # Every claim has the same per-claim mean, so any resample of whole claims gives exactly 0.5;
    # a row-level resample would not.
    df = pd.DataFrame({"claim_id": [1, 1, 2, 2, 3, 3], "x": [0, 1, 1, 0, 0, 1]})
    lo, hi = claim_bootstrap_ci(df, lambda d: float(d["x"].mean()))
    assert lo == pytest.approx(0.5) and hi == pytest.approx(0.5)


def test_claim_bootstrap_ci_brackets_point_estimate():
    import pandas as pd

    from lib.evalstats import claim_bootstrap_ci

    rng = np.random.default_rng(0)
    df = pd.DataFrame({"claim_id": np.repeat(np.arange(40), 3), "x": rng.integers(0, 2, 120)})
    lo, hi = claim_bootstrap_ci(df, lambda d: float(d["x"].mean()))
    assert lo < df["x"].mean() < hi
