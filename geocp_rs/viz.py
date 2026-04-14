"""Plotting helpers: qualitative 5×4 grid + summary figures."""
from __future__ import annotations

import os
import pickle
from typing import Iterable

import numpy as np

try:
    import matplotlib.pyplot as plt
    MPL_AVAILABLE = True
except ImportError:
    MPL_AVAILABLE = False


def _bbox_from_gt(gt: np.ndarray, pad: int = 3) -> tuple[int, int, int, int]:
    rs, cs = np.where(gt > 0)
    if len(rs) == 0:
        return 0, gt.shape[0], 0, gt.shape[1]
    r0 = max(0, rs.min() - pad); r1 = min(gt.shape[0], rs.max() + 1 + pad)
    c0 = max(0, cs.min() - pad); c1 = min(gt.shape[1], cs.max() + 1 + pad)
    return r0, r1, c0, c1


def plot_qualitative_grid(checkpoint_dir: str,
                          dataset_keys: Iterable[str],
                          seed: int,
                          save_path: str,
                          data_dir: str | None = None) -> None:
    """Render a 5×4 qualitative grid (GT / Coverage / Size / Local q).

    Each row is a dataset; columns are:
      0. Ground truth classes
      1. Coverage of SACP+GeoCP (green = covered, red = miss)
      2. SACP+GeoCP prediction-set size
      3. GeoCP local threshold q(i)

    The GT map is loaded from ``data_dir/{key}/*.mat`` using
    :mod:`geocp_rs.datasets`. The test-point data comes from the per-seed
    pickle at ``checkpoint_dir/{key}_seed{seed}.pkl``.
    """
    if not MPL_AVAILABLE:
        raise ImportError("matplotlib is required for plotting.")
    from .datasets import DATASETS, load_dataset  # local import avoids cycle

    rows = []
    for key in dataset_keys:
        p = os.path.join(checkpoint_dir, f"{key}_seed{seed}.pkl")
        if not os.path.exists(p):
            continue
        with open(p, "rb") as f:
            R = pickle.load(f)
        if data_dir is not None:
            hsi, gt, _, _ = load_dataset(key, data_dir, normalize=False)
            R["gt"] = gt
        rows.append(R)

    if not rows:
        raise FileNotFoundError(
            f"No checkpoints found in {checkpoint_dir} for seed={seed}.")

    n = len(rows)
    fig, axes = plt.subplots(n, 4, figsize=(16, 3.4 * n))
    if n == 1:
        axes = axes[None, :]

    for row, R in enumerate(rows):
        h, w = R["h"], R["w"]
        gt = R.get("gt")
        n_cls = R["n_classes"]
        te_gi = np.array(R["te_gi"])
        y_te = np.array(R["y_te"])
        ps_gc = R["sacp_geocp"]["pred_sets"]
        q_pp = np.array(R["sacp_geocp"]["q_per_pixel"])

        rotate = (h > 2 * w)
        _rot = (np.rot90 if rotate else (lambda a: a))
        r0, r1, c0, c1 = _bbox_from_gt(gt) if gt is not None else (0, h, 0, w)
        te_rc = [(int(gi // w), int(gi % w)) for gi in te_gi]
        area_per_lab = (h * w) / max(1, int((gt > 0).sum()))
        radius = 1 if area_per_lab < 3 else (2 if area_per_lab < 10 else 3)

        # Col 0: GT
        ax = axes[row, 0]
        if gt is not None:
            cmap = plt.get_cmap("tab20", max(n_cls, 20))
            gt_m = np.ma.masked_where(gt == 0, gt.astype(float))[r0:r1, c0:c1]
            ax.imshow(_rot(gt_m), cmap=cmap, vmin=1, vmax=n_cls,
                      interpolation="nearest")
        ax.set_title(f"{R.get('nice_name', key)} — GT ({n_cls} cls, "
                     f"acc={R['accuracy']:.2f})", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

        # Col 1: Coverage
        ax = axes[row, 1]
        rgb = np.ones((h, w, 3))
        if gt is not None:
            gt_bg = (gt > 0).astype(float)[:, :, None]
            rgb = rgb * (0.15 + 0.85 * gt_bg) + 0.85 * (1 - gt_bg)
        for i, (r, c) in enumerate(te_rc):
            is_cov = int(y_te[i]) in ps_gc[i]
            ra, rb = max(0, r - radius), min(h, r + radius + 1)
            ca, cb = max(0, c - radius), min(w, c + radius + 1)
            rgb[ra:rb, ca:cb] = [0.2, 0.75, 0.25] if is_cov else [0.95, 0.15, 0.15]
        ax.imshow(_rot(rgb[r0:r1, c0:c1]), interpolation="nearest")
        ax.set_title(f"Coverage = {R['sacp_geocp']['cov']:.3f}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

        # Col 2: Pred-set size
        ax = axes[row, 2]
        sz_map = np.full((h, w), np.nan)
        for i, (r, c) in enumerate(te_rc):
            sz = len(ps_gc[i])
            ra, rb = max(0, r - radius), min(h, r + radius + 1)
            ca, cb = max(0, c - radius), min(w, c + radius + 1)
            sz_map[ra:rb, ca:cb] = sz
        vmax = min(n_cls, max(3, int(np.nanmax(sz_map))
                              if not np.all(np.isnan(sz_map)) else 3))
        im = ax.imshow(_rot(sz_map[r0:r1, c0:c1]), cmap="viridis",
                       vmin=1, vmax=vmax, interpolation="nearest")
        ax.set_title(f"Pred-set Size (mean={R['sacp_geocp']['size']:.2f})",
                     fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

        # Col 3: Local q(i)
        ax = axes[row, 3]
        q_map = np.full((h, w), np.nan)
        for i, (r, c) in enumerate(te_rc):
            q = q_pp[i]
            ra, rb = max(0, r - radius), min(h, r + radius + 1)
            ca, cb = max(0, c - radius), min(w, c + radius + 1)
            q_map[ra:rb, ca:cb] = q
        im = ax.imshow(_rot(q_map[r0:r1, c0:c1]), cmap="plasma",
                       interpolation="nearest")
        ax.set_title(f"Local q(i) (bw={R['sacp_geocp']['bw']})", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
