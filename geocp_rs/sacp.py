"""SACP: spatial-aware smoothing of APS non-conformity scores.

Implements an 8-neighbor average, applied identically to calibration and test
pixels so that marginal CP coverage is preserved (Liu et al., 2024).

Reference
---------
Liu, K., Sun, T., Zeng, H., Zhang, Y., Pun, C.-M., Vong, C.-M. (2024).
Spatial-Aware Conformal Prediction for Trustworthy Hyperspectral
Image Classification. arXiv:2409.01236.
"""
from __future__ import annotations
import numpy as np


def sacp_smooth(score_map: np.ndarray,
                h: int,
                w: int,
                valid_idx: np.ndarray,
                lmd: float = 0.5) -> np.ndarray:
    """8-neighbor local averaging of a per-pixel score map (vectorized).

    For each valid pixel *i*:
        tilde_s(i, y) = (1 - lmd) * s(i, y)
                      + lmd * mean_{j in N(i) valid} s(j, y)
    where N(i) is the 8-neighborhood on the image grid.

    Parameters
    ----------
    score_map : array, shape (h*w, K)
        Flattened per-pixel score matrix. Positions outside ``valid_idx``
        may contain any value (they are masked out by ``valid_idx``).
    h, w : int
        Image height and width (so that ``h * w == score_map.shape[0]``).
    valid_idx : array of int
        Flattened indices of pixels that carry valid APS scores
        (usually the union of calibration and test pixels).
    lmd : float
        Smoothing strength lambda in [0, 1]. 0 = no smoothing, 1 = fully
        replace each pixel by the neighbor mean.

    Returns
    -------
    array, shape (h*w, K) : the smoothed score matrix. Pixels outside
    ``valid_idx`` are left untouched.
    """
    N, K = h * w, score_map.shape[1]
    sm = score_map.reshape(h, w, K).astype(np.float64)

    vm = np.zeros((h, w), dtype=bool)
    vm.reshape(-1)[valid_idx] = True

    sm_pad = np.pad(sm, ((1, 1), (1, 1), (0, 0)),
                    mode="constant", constant_values=0)
    vm_pad = np.pad(vm, ((1, 1), (1, 1)),
                    mode="constant", constant_values=False)

    neighbor_sum = np.zeros_like(sm)
    neighbor_cnt = np.zeros((h, w))
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            s_shift = sm_pad[1 + dr:h + 1 + dr, 1 + dc:w + 1 + dc, :]
            v_shift = vm_pad[1 + dr:h + 1 + dr, 1 + dc:w + 1 + dc]
            neighbor_sum += s_shift * v_shift[:, :, None]
            neighbor_cnt += v_shift

    new = sm.copy()
    has = (neighbor_cnt > 0) & vm
    avg = np.zeros_like(sm)
    avg[has] = neighbor_sum[has] / neighbor_cnt[has][:, None]
    new[has] = (1 - lmd) * sm[has] + lmd * avg[has]
    return new.reshape(N, K)
