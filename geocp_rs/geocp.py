"""GeoCP: geographic-kernel local quantile.

Implements a per-pixel conformal threshold by weighting calibration scores
with a Gaussian kernel on pixel coordinates. Because the weights depend
only on coordinates (not on labels), the weighted split-CP guarantee of
Tibshirani, Barber, Candes, Ramdas (2019) applies directly, and the
marginal coverage is preserved at level 1 - alpha.

Reference
---------
Lou, X., Luo, P., Meng, L. (2024).
    GeoConformal Prediction: A Model-Agnostic Framework of Measuring the
    Uncertainty of Spatial Prediction. arXiv:2412.08661.
Tibshirani, R.J., Barber, R.F., Candes, E.J., Ramdas, A. (2019).
    Conformal Prediction Under Covariate Shift. NeurIPS.
"""
from __future__ import annotations
import numpy as np

from .core import weighted_quantile


def geocp_local_threshold(cal_scores_true: np.ndarray,
                          coords_cal: np.ndarray,
                          coord_test: np.ndarray,
                          bandwidth: float,
                          alpha: float) -> float:
    """Per-test-pixel GeoCP quantile with a Gaussian geographic kernel.

    w_ji = exp(-||c_j - c_i||^2 / (2 h^2))
    q_j  = WeightedQuantile({cal_scores_true_i}, {w_ji}, 1 - alpha)

    The implementation uses a log-sum-exp stabilization so that even for
    test pixels very far from any calibration pixel (block OOD), at least
    one weight remains numerically nonzero and the weighted quantile is
    well-defined.

    Parameters
    ----------
    cal_scores_true : (n_cal,) array
        Non-conformity scores at the *true* calibration labels (typically
        the SACP-smoothed versions produced by the pipeline).
    coords_cal : (n_cal, 2) array
        Row/column coordinates of calibration pixels.
    coord_test : (2,) array
        Row/column coordinate of the single test pixel whose threshold we
        want to compute.
    bandwidth : float
        Gaussian bandwidth in pixel units.
    alpha : float
        Target miscoverage.

    Returns
    -------
    float : the per-pixel conformal threshold q_j.
    """
    diff = coords_cal - coord_test[None, :]
    d2 = np.sum(diff * diff, axis=1)
    log_w = -0.5 * d2 / (bandwidth ** 2 + 1e-12)
    log_w -= log_w.max()   # numerical stabilization
    w = np.exp(log_w)
    return weighted_quantile(cal_scores_true, w, alpha)
