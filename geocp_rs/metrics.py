"""Evaluation metrics for conformal prediction sets."""
from __future__ import annotations
import numpy as np


def coverage_and_size(pred_sets: list[list[int]],
                      true_labels: np.ndarray) -> tuple[float, float]:
    """Empirical coverage and mean prediction-set size.

    coverage    = fraction of test points whose true label is in their set.
    mean_size   = average cardinality of the prediction sets.
    """
    cov = float(np.mean([int(true_labels[i]) in s for i, s in enumerate(pred_sets)]))
    sz = float(np.mean([len(s) for s in pred_sets]))
    return cov, sz


def interval_score(pred_sets: list[list[int]],
                   true_labels: np.ndarray,
                   alpha: float = 0.1) -> float:
    """Set-valued Gneiting--Raftery Interval Score. Lower is better.

    IS(C, y; alpha) = |C|  +  (2 / alpha) * 1{ y not in C }

    This is a proper scoring rule: it rewards small sets, penalizes
    miscoverage linearly with penalty 2/alpha, and a method minimizing
    it is targeting the correct (1 - alpha) quantile asymptotically.

    Reference
    ---------
    Gneiting, T. and Raftery, A.E. (2007). Strictly Proper Scoring Rules,
    Prediction, and Estimation. JASA, 102(477), 359--378.
    """
    total = 0.0
    for i, s in enumerate(pred_sets):
        miss = 0.0 if int(true_labels[i]) in s else 1.0
        total += len(s) + (2.0 / alpha) * miss
    return float(total / len(pred_sets))
