"""Conformal-prediction primitives: APS scoring and (weighted) quantiles.

These are the base building blocks that SACP and GeoCP compose on top of.
Pure NumPy, no dependencies beyond the standard scientific stack.

References
----------
Romano, Y., Sesia, M., Candes, E.J. (2020).
    Classification with Valid and Adaptive Coverage. NeurIPS.
Vovk, V., Gammerman, A., Shafer, G. (2005).
    Algorithmic Learning in a Random World. Springer.
"""
from __future__ import annotations
import numpy as np


def aps_scores(probs: np.ndarray,
               labels: np.ndarray | None = None,
               rng: np.random.RandomState | None = None) -> np.ndarray:
    """Adaptive Prediction Sets (APS) non-conformity score.

    For sample *i* and candidate class *y*:
        S(i, y) = sum_{k: pi_{i,k} > pi_{i,y}} pi_{i,k}  +  U * pi_{i,y}
    with U ~ Unif(0, 1) injected for randomized tie-breaking. Small scores
    mean the true class is confidently near the top of the ranking.

    Parameters
    ----------
    probs : array, shape (n, K)
        Softmax probabilities from the underlying classifier.
    labels : array, shape (n,) or None
        If provided, returns scalar scores at the true labels (n,).
        Otherwise returns the full (n, K) score matrix.
    rng : numpy RandomState or None
        RNG used for the uniform draws. Default: numpy's global state.

    Returns
    -------
    scores : (n,) if labels given, else (n, K).
    """
    rng = rng or np.random
    n, K = probs.shape
    sort_idx = np.argsort(-probs, axis=1)
    sorted_p = np.take_along_axis(probs, sort_idx, axis=1)
    cumsum = np.cumsum(sorted_p, axis=1)
    U = rng.uniform(0, 1, size=(n, K))
    scores_sorted = cumsum - sorted_p * U
    scores = np.empty_like(scores_sorted)
    for i in range(n):
        scores[i, sort_idx[i]] = scores_sorted[i]
    if labels is not None:
        return np.array([scores[i, int(labels[i])] for i in range(n)])
    return scores


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample-corrected upper (1 - alpha) quantile for split CP.

    Uses the Vovk correction: ceil((n+1)(1-alpha)) / n.
    Returns a python float for JSON-friendliness.
    """
    n = len(scores)
    return float(np.quantile(scores, np.ceil((n + 1) * (1 - alpha)) / n))


def weighted_quantile(scores: np.ndarray,
                      weights: np.ndarray,
                      alpha: float) -> float:
    """Weighted (1 - alpha) empirical quantile.

    Sort the scores, cumulative-sum the normalized weights, and return the
    smallest score whose cumulative weight first reaches 1 - alpha.

    Returns NaN if the weights are all zero or non-finite (callers should
    handle this gracefully; the numerically-stable wrapper in
    :func:`geocp_rs.geocp.geocp_local_threshold` prevents this in practice).
    """
    order = np.argsort(scores)
    s = scores[order]
    w = weights[order]
    cum = np.cumsum(w)
    if cum[-1] <= 0 or not np.isfinite(cum[-1]):
        return float("nan")
    cum = cum / cum[-1]
    idx = np.searchsorted(cum, 1 - alpha)
    return float(s[min(idx, len(s) - 1)])
