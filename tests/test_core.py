"""Unit tests for the CP primitives."""
import numpy as np
import pytest

from geocp_rs.core import aps_scores, conformal_quantile, weighted_quantile
from geocp_rs.metrics import coverage_and_size, interval_score


def test_aps_scores_shape_and_range():
    rng = np.random.RandomState(0)
    probs = rng.dirichlet(np.ones(5), size=10)
    scores = aps_scores(probs, rng=rng)
    assert scores.shape == (10, 5)
    # APS scores are in [0, 1 + eps]
    assert scores.min() >= -1e-9
    assert scores.max() <= 1.0 + 1e-9


def test_aps_scores_at_labels():
    rng = np.random.RandomState(0)
    probs = rng.dirichlet(np.ones(4), size=7)
    labels = np.array([0, 1, 2, 3, 0, 1, 2])
    s_full = aps_scores(probs, rng=np.random.RandomState(0))
    s_lab = aps_scores(probs, labels, rng=np.random.RandomState(0))
    assert s_lab.shape == (7,)
    # Values must match the full matrix at the label positions
    for i in range(7):
        assert abs(s_lab[i] - s_full[i, labels[i]]) < 1e-12


def test_conformal_quantile_finite_sample():
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    # ceil((10+1)*(1-0.1))/10 = ceil(9.9)/10 = 10/10 = 1.0 -> max score
    q = conformal_quantile(scores, alpha=0.1)
    assert q == 1.0
    # For smaller alpha (larger target quantile), bigger q
    assert conformal_quantile(scores, alpha=0.5) < conformal_quantile(scores, alpha=0.1)


def test_weighted_quantile_uniform_matches_unweighted():
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    weights = np.ones_like(scores)
    wq = weighted_quantile(scores, weights, alpha=0.2)
    # Should be close to the unweighted (1 - 0.2) quantile
    assert 0.3 <= wq <= 0.5


def test_weighted_quantile_degenerate_returns_nan():
    scores = np.array([0.1, 0.2, 0.3])
    weights = np.zeros_like(scores)
    wq = weighted_quantile(scores, weights, alpha=0.1)
    assert np.isnan(wq)


def test_coverage_and_size_empty_hits_zero():
    pred_sets = [[0, 1], [1], [2, 3]]
    labels = np.array([0, 1, 3])
    cov, sz = coverage_and_size(pred_sets, labels)
    assert cov == 1.0
    assert sz == pytest.approx((2 + 1 + 2) / 3)


def test_interval_score_penalizes_miss():
    pred_sets = [[0], [0]]
    labels = np.array([0, 1])  # second one is a miss
    alpha = 0.1
    is_val = interval_score(pred_sets, labels, alpha)
    # mean( (1 + 0) + (1 + (2/0.1)*1) ) / 2 = (1 + 21) / 2 = 11
    assert is_val == pytest.approx(11.0)
