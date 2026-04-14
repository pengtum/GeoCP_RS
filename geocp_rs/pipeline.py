"""End-to-end SACP+GeoCP pipeline (Algorithm 1 of the paper).

Takes softmax probabilities from any trained classifier plus pixel
coordinates, and returns SACP-smoothed / GeoCP-thresholded prediction sets
with marginal coverage 1 - alpha.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial.distance import cdist

from .core import aps_scores
from .sacp import sacp_smooth
from .geocp import geocp_local_threshold


def run_sacp_geocp(probs_cal: np.ndarray,
                   probs_test: np.ndarray,
                   y_cal: np.ndarray,
                   coords_cal_rowcol: np.ndarray,
                   coords_test_rowcol: np.ndarray,
                   grid_shape: tuple[int, int],
                   cal_flat_idx: np.ndarray,
                   test_flat_idx: np.ndarray,
                   alpha: float = 0.1,
                   lmd: float = 0.5,
                   bandwidth: float | None = None,
                   rng: np.random.RandomState | None = None) -> dict:
    """Run the full SACP+GeoCP procedure.

    Parameters
    ----------
    probs_cal, probs_test : (n_cal, K), (n_test, K) softmax outputs.
    y_cal : (n_cal,) true class indices for calibration.
    coords_cal_rowcol, coords_test_rowcol : (n_cal, 2), (n_test, 2)
        Integer pixel (row, col) coordinates on the image grid.
    grid_shape : (h, w) of the parent image.
    cal_flat_idx, test_flat_idx : (n_cal,), (n_test,) flattened indices
        (row * w + col) into the h*w grid for calibration and test pixels.
    alpha : target miscoverage. Default 0.1.
    lmd : SACP smoothing strength. Default 0.5.
    bandwidth : GeoCP Gaussian bandwidth in pixel units. If None, defaults
        to the median pairwise cal-cal distance.
    rng : optional numpy RandomState for the APS tie-breaks.

    Returns
    -------
    dict with keys
        pred_sets : list of list[int] — prediction sets per test pixel.
        local_q   : (n_test,) — per-pixel GeoCP thresholds used.
        bandwidth : float — the bandwidth that was actually used.
        lmd       : float — the SACP smoothing strength.
        alpha     : float — target miscoverage.
    """
    rng = rng or np.random.RandomState(0)
    h, w = grid_shape
    K = probs_cal.shape[1]
    n_cal = len(cal_flat_idx)
    n_test = len(test_flat_idx)

    # Step 1: APS scores
    cal_all = aps_scores(probs_cal, rng=rng)
    test_all = aps_scores(probs_test, rng=rng)

    # Step 2: place scores on the h*w grid
    score_map = np.zeros((h * w, K))
    for e, i in enumerate(cal_flat_idx):
        score_map[i] = cal_all[e]
    for e, i in enumerate(test_flat_idx):
        score_map[i] = test_all[e]
    valid_idx = np.concatenate([cal_flat_idx, test_flat_idx])

    # Step 3: SACP 8-neighbor smoothing (uniform lambda)
    fused = sacp_smooth(score_map, h, w, valid_idx, lmd=lmd)

    # Step 4: extract smoothed scores at true cal labels and all test classes
    cal_true_smoothed = np.array(
        [fused[cal_flat_idx[e], int(y_cal[e])] for e in range(n_cal)])
    test_all_smoothed = np.array(
        [fused[test_flat_idx[e]] for e in range(n_test)])

    # Step 5: default bandwidth = median cal-cal distance
    if bandwidth is None:
        d_cc = cdist(coords_cal_rowcol, coords_cal_rowcol)
        bandwidth = float(np.median(d_cc[d_cc > 0]))

    # Step 6: per-pixel GeoCP quantile + prediction set
    pred_sets: list[list[int]] = []
    local_q = np.empty(n_test)
    for j in range(n_test):
        q_j = geocp_local_threshold(
            cal_true_smoothed, coords_cal_rowcol,
            coords_test_rowcol[j], bandwidth, alpha)
        local_q[j] = q_j
        pred_sets.append(np.where(test_all_smoothed[j] < q_j)[0].tolist())

    return {
        "pred_sets": pred_sets,
        "local_q": local_q,
        "bandwidth": float(bandwidth),
        "lmd": float(lmd),
        "alpha": float(alpha),
    }
