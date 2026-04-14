"""
geocp_rs — Spatially-adaptive conformal prediction for remote-sensing
hyperspectral image classification.

This package implements SACP+GeoCP, a post-hoc, classifier-agnostic conformal
prediction procedure that composes SACP score smoothing (Liu et al., 2024)
with GeoCP local geographic quantiles (Lou, Luo, Meng, 2024) to produce
reliable, spatially-adaptive prediction sets for HSI classification.

Top-level imports:
    from geocp_rs import run_sacp_geocp
    from geocp_rs import aps_scores, conformal_quantile, weighted_quantile
    from geocp_rs import sacp_smooth, geocp_local_threshold
    from geocp_rs import interval_score, coverage_and_size
"""
from .core import aps_scores, conformal_quantile, weighted_quantile
from .sacp import sacp_smooth
from .geocp import geocp_local_threshold
from .metrics import interval_score, coverage_and_size
from .pipeline import run_sacp_geocp

__version__ = "0.1.0"

__all__ = [
    "aps_scores",
    "conformal_quantile",
    "weighted_quantile",
    "sacp_smooth",
    "geocp_local_threshold",
    "interval_score",
    "coverage_and_size",
    "run_sacp_geocp",
    "__version__",
]
