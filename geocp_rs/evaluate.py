"""High-level evaluator that runs all CP variants on a trained model's outputs.

Given per-pixel softmax probabilities from the 3D-CNN (or any classifier) plus
pixel coordinates and labels, this module computes:

1. Standard split CP (no spatial info).
2. SACP at lambda in {0.3, 0.5, 0.7} (spatial score smoothing).
3. SACP+GeoCP with a 5-fold-CV selected bandwidth on the calibration set
   (spatial score smoothing AND per-pixel geographic local quantile).

The bandwidth grid, lambda choices, and alpha are documented arguments.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.model_selection import KFold

from .core import aps_scores, conformal_quantile, weighted_quantile
from .sacp import sacp_smooth
from .geocp import geocp_local_threshold
from .metrics import coverage_and_size, interval_score


BW_GRID_DEFAULT = (3, 5, 7, 10, 15, 20, 30, 50, 100)
LMD_GRID_DEFAULT = (0.3, 0.5, 0.7)


def evaluate_all_cp_methods(probs_cal: np.ndarray,
                             probs_test: np.ndarray,
                             y_cal: np.ndarray,
                             y_test: np.ndarray,
                             cal_flat_idx: np.ndarray,
                             test_flat_idx: np.ndarray,
                             h: int,
                             w: int,
                             alpha: float = 0.1,
                             lmd_grid: tuple = LMD_GRID_DEFAULT,
                             bw_grid: tuple = BW_GRID_DEFAULT,
                             cv_folds: int = 5,
                             rng_seed: int = 42) -> dict:
    """Evaluate Standard CP, SACP, and SACP+GeoCP on a single (dataset, seed).

    All hyperparameter selection is done via CV on the calibration set. No
    test labels are accessed during selection.

    Returns a dict of per-method results:
        {
          'standard_cp'  : {'cov': ..., 'size': ..., 'is': ...,
                             'pred_sets': ..., 'q': ...},
          'sacp_0.3'     : {...},
          'sacp_0.5'     : {...},
          'sacp_0.7'     : {...},
          'sacp_geocp'   : {'cov': ..., 'size': ..., 'is': ...,
                             'pred_sets': ..., 'q_per_pixel': ...,
                             'bw': selected_bw, 'cv_is_mean': {bw: is}},
          'bandwidth_selected' : int,
        }
    """
    rng = np.random.RandomState(rng_seed)
    K = probs_cal.shape[1]
    N = h * w
    n_cal = len(cal_flat_idx)
    n_test = len(test_flat_idx)

    # APS scores (calibration: full + at-true-label; test: full)
    cal_all = aps_scores(probs_cal, rng=rng)
    test_all = aps_scores(probs_test, rng=rng)
    cal_true = aps_scores(probs_cal, y_cal, rng=rng)

    coords_cal = np.stack([cal_flat_idx // w, cal_flat_idx % w], axis=1).astype(float)
    coords_test = np.stack([test_flat_idx // w, test_flat_idx % w], axis=1).astype(float)

    results: dict = {}

    # ---- 1) Standard CP ----
    q0 = conformal_quantile(cal_true, alpha)
    ps_std = [np.where(test_all[i] < q0)[0].tolist() for i in range(n_test)]
    cov_std, sz_std = coverage_and_size(ps_std, y_test)
    results["standard_cp"] = {
        "cov": cov_std, "size": sz_std,
        "is": interval_score(ps_std, y_test, alpha),
        "q": float(q0), "pred_sets": ps_std,
    }

    # ---- 2) SACP at multiple lambdas ----
    score_map = np.zeros((N, K))
    for e, i in enumerate(cal_flat_idx):
        score_map[i] = cal_all[e]
    for e, i in enumerate(test_flat_idx):
        score_map[i] = test_all[e]
    valid_idx = np.concatenate([cal_flat_idx, test_flat_idx])

    fused_by_lmd: dict[float, np.ndarray] = {}
    for lmd in lmd_grid:
        fused = sacp_smooth(score_map, h, w, valid_idx, lmd=lmd)
        fused_by_lmd[lmd] = fused
        fcu = np.array([fused[cal_flat_idx[e], int(y_cal[e])] for e in range(n_cal)])
        ftu = np.array([fused[test_flat_idx[e]] for e in range(n_test)])
        q = conformal_quantile(fcu, alpha)
        ps = [np.where(ftu[i] < q)[0].tolist() for i in range(n_test)]
        cov, sz = coverage_and_size(ps, y_test)
        results[f"sacp_{lmd}"] = {
            "cov": cov, "size": sz,
            "is": interval_score(ps, y_test, alpha),
            "q": float(q), "pred_sets": ps,
            "lmd": float(lmd),
        }

    # ---- 3) SACP+GeoCP with 5-fold CV bandwidth selection (lmd = 0.5) ----
    fused05 = fused_by_lmd[0.5]
    fcu05 = np.array([fused05[cal_flat_idx[e], int(y_cal[e])] for e in range(n_cal)])
    ftu05 = np.array([fused05[test_flat_idx[e]] for e in range(n_test)])

    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=rng_seed)
    cv_is = {bw: [] for bw in bw_grid}
    for f_tr_local, f_val_local in kf.split(np.arange(n_cal)):
        f_ca_gi = cal_flat_idx[f_tr_local]
        f_va_gi = cal_flat_idx[f_val_local]
        f_fcu = np.array([fused05[f_ca_gi[e], int(y_cal[f_tr_local[e]])]
                          for e in range(len(f_tr_local))])
        f_ftu = np.array([fused05[f_va_gi[e]] for e in range(len(f_val_local))])
        f_yv = y_cal[f_val_local]
        f_coords_ca = np.stack([f_ca_gi // w, f_ca_gi % w], axis=1).astype(float)
        f_coords_va = np.stack([f_va_gi // w, f_va_gi % w], axis=1).astype(float)
        for bw in bw_grid:
            ps = []
            for i in range(len(f_val_local)):
                q = geocp_local_threshold(f_fcu, f_coords_ca, f_coords_va[i],
                                           float(bw), alpha)
                ps.append(np.where(f_ftu[i] < q)[0].tolist())
            cv_is[bw].append(interval_score(ps, f_yv, alpha))
    best_bw = int(min(bw_grid, key=lambda b: np.mean(cv_is[b])))

    # Final eval with the CV-selected bandwidth
    q_per_pixel = np.empty(n_test)
    ps_gc: list[list[int]] = []
    for j in range(n_test):
        q_j = geocp_local_threshold(fcu05, coords_cal, coords_test[j],
                                     float(best_bw), alpha)
        q_per_pixel[j] = q_j
        ps_gc.append(np.where(ftu05[j] < q_j)[0].tolist())
    cov_gc, sz_gc = coverage_and_size(ps_gc, y_test)
    results["sacp_geocp"] = {
        "cov": cov_gc, "size": sz_gc,
        "is": interval_score(ps_gc, y_test, alpha),
        "bw": int(best_bw),
        "q_per_pixel": q_per_pixel.tolist(),
        "pred_sets": ps_gc,
        "cv_is_mean": {int(bw): float(np.mean(cv_is[bw])) for bw in bw_grid},
    }
    results["bandwidth_selected"] = int(best_bw)

    return results
