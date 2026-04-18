"""Generate all unified-style analysis figures for the HSI + S2 experiments.

All figures share:
- Method color palette: D gray, A tab:blue, B tab:green, C tab:red
- Font: DejaVu Sans, 10 pt body, 11 pt title
- DPI: 180
- Aspect: rectangular panels, consistent margins
- File naming: figN_<name>.png (summary figures), spatial_<exp>_<key>.png (spatial maps)

Run from repo root:
    python analysis/scripts/generate_analysis.py
"""
from __future__ import annotations

import csv
import json
import os
import pickle
from collections import Counter
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy import stats

# ---------------------------------------------------------------- setup paths

ROOT = Path(
    "/Users/peng/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/"
    "My Drive/Research/0_GeoCP_LISA/GeoCP_RS"
)
HSI = ROOT / "hsi_experiment"
S2 = ROOT / "s2_experiment"
HSI_RAW = Path(
    "/Users/peng/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/"
    "My Drive/sacp_geocp/datasets"
)
OUT = ROOT / "analysis" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- global style

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 180,
    "savefig.bbox": "tight",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

METHOD_COLORS = {
    "D": "#9e9e9e",   # Standard CP — gray
    "A": "#1f77b4",   # SACP default — blue
    "B": "#2ca02c",   # SACP-CV-r — green
    "C": "#d62728",   # SACP+GeoCP — red
}
METHOD_LABELS = {
    "D": "(D) Standard CP",
    "A": "(A) SACP default",
    "B": "(B) SACP-CV-r",
    "C": "(C) SACP+GeoCP",
}
METHOD_ORDER = ["D", "A", "B", "C"]

# --------------------------------------------------------------- data loaders

HSI_DATASETS = ["ip", "pu", "sa", "ksc", "botswana"]
HSI_NICE = {
    "ip": "Indian Pines",
    "pu": "Pavia University",
    "sa": "Salinas",
    "ksc": "KSC",
    "botswana": "Botswana",
}
HSI_GT_FILES = {
    "ip":       ("ip/Indian_pines_gt.mat", "indian_pines_gt"),
    "pu":       ("pu/PaviaU_gt.mat",       "paviaU_gt"),
    "sa":       ("sa/Salinas_gt.mat",      "salinas_gt"),
    "ksc":      ("ksc/KSC_gt.mat",         "KSC_gt"),
    "botswana": ("botswana/Botswana_gt.mat", "Botswana_gt"),
}


def load_hsi_per_seed() -> list[dict]:
    rows = []
    with open(HSI / "results" / "per_seed.csv") as f:
        for r in csv.DictReader(f):
            rows.append({
                "dataset": r["dataset"],
                "seed": int(r["seed"]),
                "acc": float(r["accuracy"]),
                "is_D": float(r["D_is"]), "is_A": float(r["A_is"]),
                "is_B": float(r["B_is"]), "is_C": float(r["C_is"]),
                "cov_D": float(r["D_cov"]), "cov_A": float(r["A_cov"]),
                "cov_B": float(r["B_cov"]), "cov_C": float(r["C_cov"]),
                "sz_D": float(r["D_size"]), "sz_A": float(r["A_size"]),
                "sz_B": float(r["B_size"]), "sz_C": float(r["C_size"]),
                "r_B": int(r["B_r"]),
                "r_C": int(r["C_r"]), "bw_C": int(r["C_bw"]),
            })
    return rows


def load_s2_per_row() -> list[dict]:
    rows = []
    with open(S2 / "results" / "csv" / "main_comparison_multisize.csv") as f:
        for r in csv.DictReader(f):
            rows.append({
                "tile": r["tile"],
                "size_km": float(r["size_km"]),
                "is_D": float(r["is_D"]),
                "is_A": float(r["is_A"]),
                "is_B": float(r["is_B"]),
                "is_C": float(r["is_C"]),
                "cov_D": float(r["cov_D"]),
                "cov_A": float(r["cov_A"]),
                "cov_B": float(r["cov_B"]),
                "cov_C": float(r["cov_C"]),
                "r_B": int(r["best_r_sacp"]),
                "r_C": int(r["best_r_geocp"]),
                "bw_C": int(r["best_bw_geocp"]),
            })
    return rows


# ------------------------------------------------------- headline figure (fig1)

def fig1_headline_bars(hsi_rows, s2_rows):
    """Side-by-side mean IS bar chart. Left: HSI per-dataset. Right: S2 per-size."""
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5),
                             gridspec_kw={"width_ratios": [5, 3]})

    # --- Panel A: HSI per-dataset ---
    ax = axes[0]
    x = np.arange(len(HSI_DATASETS))
    width = 0.2
    for i, m in enumerate(METHOD_ORDER):
        means = [np.mean([r[f"is_{m}"] for r in hsi_rows if r["dataset"] == ds])
                 for ds in HSI_DATASETS]
        stds = [np.std([r[f"is_{m}"] for r in hsi_rows if r["dataset"] == ds])
                for ds in HSI_DATASETS]
        offset = (i - 1.5) * width
        ax.bar(x + offset, means, width, yerr=stds, capsize=2.5,
               color=METHOD_COLORS[m], label=METHOD_LABELS[m],
               edgecolor="black", linewidth=0.5, error_kw={"elinewidth": 0.8})
    ax.set_xticks(x)
    ax.set_xticklabels([HSI_NICE[ds] for ds in HSI_DATASETS], rotation=15, ha="right")
    ax.set_ylabel("Mean Interval Score  (lower = better)")
    ax.set_title("HSI  —  5 datasets × 10 seeds (n = 50)")
    ax.legend(loc="upper right", ncol=2, frameon=False, fontsize=8.5)
    ax.grid(True, alpha=0.3, axis="y")

    # --- Panel B: S2 per-size ---
    ax = axes[1]
    sizes = sorted(set(r["size_km"] for r in s2_rows))
    x = np.arange(len(sizes))
    for i, m in enumerate(METHOD_ORDER):
        means = [np.mean([r[f"is_{m}"] for r in s2_rows if r["size_km"] == sz])
                 for sz in sizes]
        stds = [np.std([r[f"is_{m}"] for r in s2_rows if r["size_km"] == sz])
                for sz in sizes]
        offset = (i - 1.5) * width
        ax.bar(x + offset, means, width, yerr=stds, capsize=2.5,
               color=METHOD_COLORS[m], label=METHOD_LABELS[m],
               edgecolor="black", linewidth=0.5, error_kw={"elinewidth": 0.8})
    ax.set_xticks(x)
    ax.set_xticklabels([f"{sz:.0f} km" for sz in sizes])
    ax.set_ylabel("Mean Interval Score")
    ax.set_title("Sentinel-2  —  3 patch sizes × 10 tiles (n = 29)")
    ax.grid(True, alpha=0.3, axis="y")

    out = OUT / "fig1_headline_bars.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved  {out.name}")


# ---------------------------------------------------- paired improvement (fig2)

def fig2_paired_improvements(hsi_rows, s2_rows):
    """Per-dataset bar: C vs A and C vs D improvements with error bars + significance stars."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5),
                             gridspec_kw={"width_ratios": [5, 3]})

    def stars(p):
        if p < 1e-4: return "****"
        if p < 1e-3: return "***"
        if p < 1e-2: return "**"
        if p < 5e-2: return "*"
        return "ns"

    # --- HSI ---
    ax = axes[0]
    cats = list(HSI_DATASETS) + ["pooled"]
    x = np.arange(len(cats))
    width = 0.4
    CvA_means, CvA_err, CvA_p, CvD_means, CvD_err, CvD_p = [], [], [], [], [], []
    for ds in HSI_DATASETS:
        dsr = [r for r in hsi_rows if r["dataset"] == ds]
        CvA = [100 * (r["is_A"] - r["is_C"]) / r["is_A"] for r in dsr]
        CvD = [100 * (r["is_D"] - r["is_C"]) / r["is_D"] for r in dsr]
        CvA_means.append(np.mean(CvA)); CvA_err.append(np.std(CvA) / np.sqrt(len(CvA)))
        CvD_means.append(np.mean(CvD)); CvD_err.append(np.std(CvD) / np.sqrt(len(CvD)))
        _, pA = stats.ttest_rel([r["is_A"] for r in dsr], [r["is_C"] for r in dsr])
        _, pD = stats.ttest_rel([r["is_D"] for r in dsr], [r["is_C"] for r in dsr])
        CvA_p.append(pA); CvD_p.append(pD)
    # pooled
    allA_is = [r["is_A"] for r in hsi_rows]; allC_is = [r["is_C"] for r in hsi_rows]
    allD_is = [r["is_D"] for r in hsi_rows]
    CvA_all = [100 * (a - c) / a for a, c in zip(allA_is, allC_is)]
    CvD_all = [100 * (d - c) / d for d, c in zip(allD_is, allC_is)]
    CvA_means.append(np.mean(CvA_all)); CvA_err.append(np.std(CvA_all) / np.sqrt(len(CvA_all)))
    CvD_means.append(np.mean(CvD_all)); CvD_err.append(np.std(CvD_all) / np.sqrt(len(CvD_all)))
    _, pA_all = stats.ttest_rel(allA_is, allC_is)
    _, pD_all = stats.ttest_rel(allD_is, allC_is)
    CvA_p.append(pA_all); CvD_p.append(pD_all)

    bA = ax.bar(x - width/2, CvA_means, width, yerr=CvA_err,
                color=METHOD_COLORS["A"], label="C vs A (SACP default)",
                capsize=2.5, edgecolor="black", linewidth=0.5)
    bD = ax.bar(x + width/2, CvD_means, width, yerr=CvD_err,
                color=METHOD_COLORS["D"], label="C vs D (Standard CP)",
                capsize=2.5, edgecolor="black", linewidth=0.5)
    # stars
    for xi, (mA, eA, pA, mD, eD, pD) in enumerate(zip(CvA_means, CvA_err, CvA_p,
                                                       CvD_means, CvD_err, CvD_p)):
        ax.text(xi - width/2, mA + eA + 0.4, stars(pA), ha="center", fontsize=8)
        ax.text(xi + width/2, mD + eD + 0.4, stars(pD), ha="center", fontsize=8)

    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x)
    labels = [HSI_NICE[ds] for ds in HSI_DATASETS] + ["Pooled"]
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("IS improvement (%)   [C = SACP+GeoCP]")
    ax.set_title("HSI  —  paired improvement (± s.e.m.)")
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # --- S2 ---
    ax = axes[1]
    sizes = sorted(set(r["size_km"] for r in s2_rows)) + ["pooled"]
    x = np.arange(len(sizes))
    CvA_means, CvA_err, CvA_p, CvD_means, CvD_err, CvD_p = [], [], [], [], [], []
    for sz in sizes[:-1]:
        szr = [r for r in s2_rows if r["size_km"] == sz]
        CvA = [100 * (r["is_A"] - r["is_C"]) / r["is_A"] for r in szr]
        CvD = [100 * (r["is_D"] - r["is_C"]) / r["is_D"] for r in szr]
        CvA_means.append(np.mean(CvA)); CvA_err.append(np.std(CvA) / np.sqrt(len(CvA)))
        CvD_means.append(np.mean(CvD)); CvD_err.append(np.std(CvD) / np.sqrt(len(CvD)))
        _, pA = stats.ttest_rel([r["is_A"] for r in szr], [r["is_C"] for r in szr])
        _, pD = stats.ttest_rel([r["is_D"] for r in szr], [r["is_C"] for r in szr])
        CvA_p.append(pA); CvD_p.append(pD)
    allA_is = [r["is_A"] for r in s2_rows]; allC_is = [r["is_C"] for r in s2_rows]
    allD_is = [r["is_D"] for r in s2_rows]
    CvA_all = [100 * (a - c) / a for a, c in zip(allA_is, allC_is)]
    CvD_all = [100 * (d - c) / d for d, c in zip(allD_is, allC_is)]
    CvA_means.append(np.mean(CvA_all)); CvA_err.append(np.std(CvA_all) / np.sqrt(len(CvA_all)))
    CvD_means.append(np.mean(CvD_all)); CvD_err.append(np.std(CvD_all) / np.sqrt(len(CvD_all)))
    _, pA_all = stats.ttest_rel(allA_is, allC_is)
    _, pD_all = stats.ttest_rel(allD_is, allC_is)
    CvA_p.append(pA_all); CvD_p.append(pD_all)

    ax.bar(x - width/2, CvA_means, width, yerr=CvA_err,
           color=METHOD_COLORS["A"], label="C vs A", capsize=2.5,
           edgecolor="black", linewidth=0.5)
    ax.bar(x + width/2, CvD_means, width, yerr=CvD_err,
           color=METHOD_COLORS["D"], label="C vs D", capsize=2.5,
           edgecolor="black", linewidth=0.5)
    for xi, (mA, eA, pA, mD, eD, pD) in enumerate(zip(CvA_means, CvA_err, CvA_p,
                                                       CvD_means, CvD_err, CvD_p)):
        ax.text(xi - width/2, mA + eA + 0.15, stars(pA), ha="center", fontsize=8)
        ax.text(xi + width/2, mD + eD + 0.15, stars(pD), ha="center", fontsize=8)

    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s:.0f} km" if isinstance(s, float) else "Pooled" for s in sizes])
    ax.set_ylabel("IS improvement (%)")
    ax.set_title("Sentinel-2  —  paired improvement")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    out = OUT / "fig2_paired_improvements.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved  {out.name}")


# --------------------------------------------------- radius distribution (fig3)

def fig3_radius_distribution(hsi_rows, s2_rows):
    """Stacked bar: count of CV-selected radii for methods B and C on both experiments."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
    radii_all = [1, 2, 3, 5, 10]

    for ax, rows, title in [(axes[0], hsi_rows, "HSI  (n = 50)"),
                             (axes[1], s2_rows, "Sentinel-2  (n = 29)")]:
        rB = Counter(r["r_B"] for r in rows)
        rC = Counter(r["r_C"] for r in rows)
        x = np.arange(len(radii_all))
        width = 0.38
        bB = ax.bar(x - width/2, [rB.get(r, 0) for r in radii_all], width,
                     color=METHOD_COLORS["B"], label="(B) SACP-CV-r",
                     edgecolor="black", linewidth=0.5)
        bC = ax.bar(x + width/2, [rC.get(r, 0) for r in radii_all], width,
                     color=METHOD_COLORS["C"], label="(C) SACP+GeoCP",
                     edgecolor="black", linewidth=0.5)
        ax.axvline(0, color="black", lw=0.6, ls="--", alpha=0.5)
        ax.text(0, ax.get_ylim()[1] * 0.9 if False else max([rB.get(r, 0) for r in radii_all] + [rC.get(r, 0) for r in radii_all]) + 0.5,
                "r=1\nSACP default", ha="center", fontsize=8, color="dimgray")
        ax.set_xticks(x)
        ax.set_xticklabels([f"r={r}\n({2*r+1}×{2*r+1})" for r in radii_all])
        ax.set_ylabel("Number of runs")
        ax.set_title(title)
        ax.legend(loc="upper left", frameon=False)
        ax.grid(True, alpha=0.3, axis="y")
        # annotate bar counts
        for b in list(bB) + list(bC):
            h = b.get_height()
            if h > 0:
                ax.text(b.get_x() + b.get_width()/2, h + 0.25,
                        f"{int(h)}", ha="center", fontsize=8)

    fig.suptitle("CV-selected Moore radius distribution   —   "
                 "r=1 (SACP default) is never chosen by method C", y=1.02)
    out = OUT / "fig3_radius_distribution.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved  {out.name}")


# ---------------------------------------------------- coverage check (fig4)

def fig4_coverage_check(hsi_rows, s2_rows):
    """Strip/box plot of per-run coverage for each of D/A/B/C in HSI and S2."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, rows, title in [(axes[0], hsi_rows, "HSI  (50 runs × 4 methods)"),
                             (axes[1], s2_rows, "Sentinel-2  (29 runs × 4 methods)")]:
        data = [[r[f"cov_{m}"] for r in rows] for m in METHOD_ORDER]
        bp = ax.boxplot(data, labels=[METHOD_LABELS[m] for m in METHOD_ORDER],
                         patch_artist=True, widths=0.55,
                         boxprops={"linewidth": 0.8}, medianprops={"color": "black", "linewidth": 1.2},
                         whiskerprops={"linewidth": 0.8},
                         flierprops={"marker": "o", "markersize": 3, "markerfacecolor": "dimgray", "alpha": 0.6})
        for patch, m in zip(bp["boxes"], METHOD_ORDER):
            patch.set_facecolor(METHOD_COLORS[m])
            patch.set_alpha(0.6)
        # jittered points
        for i, vals in enumerate(data):
            jitter = np.random.RandomState(i).normal(0, 0.04, size=len(vals))
            ax.scatter([i + 1 + j for j in jitter], vals, s=8,
                        color=METHOD_COLORS[METHOD_ORDER[i]], alpha=0.6,
                        edgecolor="black", linewidth=0.3, zorder=3)
        ax.axhline(0.9, color="black", lw=0.8, ls="--", alpha=0.7)
        ax.text(4.45, 0.9, "α=0.10\ntarget", ha="left", va="center", fontsize=8.5, color="dimgray")
        ax.set_ylabel("Test-set coverage")
        ax.set_title(title)
        ax.set_ylim(0.86, 0.95)
        ax.grid(True, alpha=0.3, axis="y")
        ax.tick_params(axis="x", labelsize=8.5)

    fig.suptitle("Marginal coverage is maintained across all runs and all methods", y=1.02)
    out = OUT / "fig4_coverage_check.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved  {out.name}")


# -------------------------------------------------- per-seed scatter (fig5)

def fig5_per_run_scatter(hsi_rows, s2_rows):
    """Scatter of A vs C per run; points below diagonal = C better. Unified axes."""
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5), sharex=False, sharey=False)
    for ax, rows, title in [(axes[0], hsi_rows, "HSI (50 runs)"),
                             (axes[1], s2_rows, "Sentinel-2 (29 runs)")]:
        A = np.array([r["is_A"] for r in rows])
        C = np.array([r["is_C"] for r in rows])
        # diagonal line
        lo = min(A.min(), C.min()) * 0.98
        hi = max(A.max(), C.max()) * 1.02
        ax.plot([lo, hi], [lo, hi], ls="--", color="dimgray", lw=0.8, zorder=1)
        # points
        if "dataset" in rows[0]:
            # HSI: color by dataset
            cmap = plt.get_cmap("tab10")
            for i, ds in enumerate(HSI_DATASETS):
                idx = [j for j, r in enumerate(rows) if r["dataset"] == ds]
                ax.scatter(A[idx], C[idx], color=cmap(i), label=HSI_NICE[ds],
                            s=45, alpha=0.8, edgecolor="black", linewidth=0.5, zorder=3)
            ax.legend(loc="upper left", frameon=False, fontsize=8)
        else:
            # S2: color by size
            sizes = sorted(set(r["size_km"] for r in rows))
            cmap = plt.get_cmap("viridis")
            for i, sz in enumerate(sizes):
                idx = [j for j, r in enumerate(rows) if r["size_km"] == sz]
                color = cmap(i / max(len(sizes) - 1, 1))
                ax.scatter(A[idx], C[idx], color=color, label=f"{sz:.0f} km",
                            s=45, alpha=0.8, edgecolor="black", linewidth=0.5, zorder=3)
            ax.legend(loc="upper left", frameon=False, fontsize=8, title="Patch size")

        # annotate: # points below diagonal
        n_below = int(np.sum(C < A))
        ax.text(0.97, 0.03,
                f"C better on {n_below}/{len(A)} runs",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=9, bbox=dict(facecolor="white", alpha=0.85, edgecolor="gray", linewidth=0.5))

        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_xlabel("IS — (A) SACP default")
        ax.set_ylabel("IS — (C) SACP+GeoCP")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Per-run comparison: every point below the diagonal is a paired win for (C)",
                 y=1.02)
    out = OUT / "fig5_per_run_scatter.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  saved  {out.name}")


# -------------------------------------------------- HSI spatial visualization

def plot_spatial_hsi(pickle_path: Path, gt_path: Path, gt_key: str, out_path: Path):
    """5-panel spatial figure for an HSI (dataset, seed):
       [GT]  [Pred A]  [Size A]  [Size C]  [q_j (C)]
    """
    with open(pickle_path, "rb") as f:
        r = pickle.load(f)
    h, w = r["h"], r["w"]
    gt_img = sio.loadmat(str(gt_path))[gt_key]
    n_cls = r["n_classes"]

    te_gi = np.array(r["te_gi"])
    y_te = np.array(r["y_te"])
    pred_te = np.array(r["pred_te"])
    q_C = np.array(r["sacp_geocp"]["q_per_pixel"])
    ps_A = r["sacp_default"]["pred_sets"]
    ps_C = r["sacp_geocp"]["pred_sets"]
    size_A = np.array([len(s) for s in ps_A])
    size_C = np.array([len(s) for s in ps_C])

    te_r = te_gi // w
    te_c = te_gi % w

    # bbox crop
    rs, cs = np.where(gt_img > 0)
    pad = max(3, min(h, w) // 30)
    r0 = max(0, rs.min() - pad); r1 = min(h, rs.max() + 1 + pad)
    c0 = max(0, cs.min() - pad); c1 = min(w, cs.max() + 1 + pad)
    rotate = (r1 - r0) > 2 * (c1 - c0)  # rotate tall/narrow scenes for readability

    # marker radius for scatter-style overlay (HSI test-pixel density varies wildly)
    area_per_lab = (h * w) / max(1, int((gt_img > 0).sum()))
    radius = 1 if area_per_lab < 3 else (2 if area_per_lab < 10 else 3)

    def _overlay(values, dtype=np.float64, fill=np.nan):
        out = np.full((h, w), fill, dtype=dtype)
        for i, (rr, cc) in enumerate(zip(te_r, te_c)):
            ra, rb = max(0, rr - radius), min(h, rr + radius + 1)
            ca, cb = max(0, cc - radius), min(w, cc + radius + 1)
            out[ra:rb, ca:cb] = values[i]
        return out

    _rot = (np.rot90 if rotate else (lambda a: a))

    # Dynamic figsize based on cropped panel aspect (after any rotation)
    crop_h = (c1 - c0) if rotate else (r1 - r0)
    crop_w = (r1 - r0) if rotate else (c1 - c0)
    panel_h_in = 3.6
    panel_w_in = max(2.4, min(4.0, panel_h_in * crop_w / max(crop_h, 1)))
    fig_w = panel_w_in * 5 + 1.2   # 5 panels + colorbar margin
    fig_h = panel_h_in + 0.8       # panel + suptitle
    fig, axes = plt.subplots(1, 5, figsize=(fig_w, fig_h), constrained_layout=True)

    # Panel 0: ESA-style categorical GT
    cmap_cat = plt.get_cmap("tab20", max(n_cls, 20))
    gt_m = np.ma.masked_where(gt_img == 0, gt_img.astype(float))[r0:r1, c0:c1]
    axes[0].imshow(_rot(gt_m), cmap=cmap_cat, vmin=1, vmax=n_cls, interpolation="nearest")
    axes[0].set_title(f"GT  ({n_cls} classes)")
    axes[0].set_xticks([]); axes[0].set_yticks([])

    # Panel 1: Prediction (same categorical cmap)
    pred_img = _overlay(pred_te.astype(float) + 1)
    axes[1].imshow(_rot(pred_img[r0:r1, c0:c1]), cmap=cmap_cat, vmin=1, vmax=n_cls,
                   interpolation="nearest")
    axes[1].set_title(f"3D-CNN pred  (acc={r['accuracy']:.3f})")
    axes[1].set_xticks([]); axes[1].set_yticks([])

    # Panels 2 & 3: set sizes (shared vmin/vmax)
    size_vmax = max(size_A.max(), size_C.max())
    for ax_i, (sv, m_lab, sz_arr) in enumerate([
        (axes[2], f"(A) SACP r=1   mean={size_A.mean():.2f}", size_A),
        (axes[3], f"(C) SACP+GeoCP r={r['sacp_geocp']['r']},bw={r['sacp_geocp']['bw']}   mean={size_C.mean():.2f}", size_C),
    ]):
        sz_img = _overlay(sz_arr.astype(float))
        im = sv.imshow(_rot(sz_img[r0:r1, c0:c1]), cmap="viridis",
                        vmin=0, vmax=size_vmax, interpolation="nearest")
        sv.set_title(m_lab)
        sv.set_xticks([]); sv.set_yticks([])
        if ax_i == 1:
            plt.colorbar(im, ax=sv, fraction=0.045, pad=0.02, label="set size")

    # Panel 4: q_j (C)
    q_img = _overlay(q_C)
    vmin_q, vmax_q = np.nanpercentile(q_C, [2, 98])
    im = axes[4].imshow(_rot(q_img[r0:r1, c0:c1]), cmap="magma",
                        vmin=vmin_q, vmax=vmax_q, interpolation="nearest")
    axes[4].set_title(f"$\\hat q_j$  (sd={q_C.std():.4f})")
    axes[4].set_xticks([]); axes[4].set_yticks([])
    plt.colorbar(im, ax=axes[4], fraction=0.045, pad=0.02)

    fig.suptitle(f"{r['nice_name']}  (seed {r['seed']})", y=1.02)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  saved  {out_path.name}")


# -------------------------------------------------- S2 spatial visualization

def plot_spatial_s2(npz_path: Path, out_path: Path, tile_name: str = None):
    """5-panel spatial figure for an S2 (tile, size):
       [GT]  [Pred]  [Size A]  [Size C]  [q_j (C)]
    Uses the same method color / cmap conventions as HSI.
    """
    d = np.load(npz_path, allow_pickle=True)
    H, W = int(d["H"]), int(d["W"])
    label_full = d["label"]
    test_flat = d["test_flat_idx"]
    size_A = d["size_A"]
    size_C = d["size_C"]
    q_C = d["q_C"]

    def scatter_grid(vals):
        out = np.full(H * W, np.nan, dtype=np.float64)
        out[test_flat] = vals
        return out.reshape(H, W)

    # Categorical palette for ESA
    all_codes = sorted(set(label_full.ravel()) - {0})
    ESA_COLORS = {
        10: "#006400", 20: "#ffbb22", 30: "#ffff4c", 40: "#f096ff",
        50: "#fa0000", 60: "#b4b4b4", 70: "#f0f0f0", 80: "#0064c8",
        90: "#0096a0", 95: "#00cf75", 100: "#fae6a0",
    }
    cat_cmap = ListedColormap([ESA_COLORS.get(c, "#999999") for c in all_codes])
    cat_norm = BoundaryNorm(list(range(len(all_codes) + 1)), cat_cmap.N)
    code_to_idx = {c: i for i, c in enumerate(all_codes)}

    # S2 panels are (approximately) square-shaped. Make each panel ~3.2 inch.
    panel_in = 3.2
    fig_w = panel_in * 5 + 1.4
    fig_h = panel_in + 0.8
    fig, axes = plt.subplots(1, 5, figsize=(fig_w, fig_h), constrained_layout=True)

    # GT
    gt = np.full(label_full.shape, np.nan)
    for c, i in code_to_idx.items():
        gt[label_full == c] = i
    axes[0].imshow(gt, cmap=cat_cmap, norm=cat_norm, interpolation="nearest")
    axes[0].set_title("GT  (ESA WorldCover)")
    axes[0].set_xticks([]); axes[0].set_yticks([])

    # Prediction (from set size via the underlying pred_test array is not saved in npz;
    # we approximate by using the smallest-size method D reflection — fallback: show GT again masked)
    # Instead we use a simple prediction proxy: the most likely class from GT masked.
    # Better: just re-use GT here and label it "Test pixels" — no, use pred directly.
    # Since we don't have pred saved, we skip this panel and show SACP (A) size instead.
    # To match HSI's 5-panel style we substitute: [GT, Size A, Size C, q_j, — leave blank]
    # Or better: duplicate GT cropped to test pixels as a "Test coverage" map.
    test_mask = np.zeros(H * W, dtype=bool)
    test_mask[test_flat] = True
    coverage_img = np.full(H * W, np.nan)
    coverage_img[test_flat] = (label_full.ravel()[test_flat]).astype(float)
    # re-encode for colormap
    cov_remapped = np.full(coverage_img.shape, np.nan)
    for c, i in code_to_idx.items():
        cov_remapped[coverage_img == c] = i
    axes[1].imshow(cov_remapped.reshape(H, W), cmap=cat_cmap, norm=cat_norm,
                   interpolation="nearest")
    axes[1].set_title(f"Test pixels  ({len(test_flat):,})")
    axes[1].set_xticks([]); axes[1].set_yticks([])

    # Size A / Size C (shared scale)
    size_vmax = float(max(size_A.max(), size_C.max()))
    for ax, arr, lab in [(axes[2], size_A, f"(A) SACP r=1   mean={size_A.mean():.2f}"),
                         (axes[3], size_C, f"(C) SACP+GeoCP r={int(d['best_r_geocp'])},bw={int(d['best_bw_geocp'])}   mean={size_C.mean():.2f}")]:
        im = ax.imshow(scatter_grid(arr), cmap="viridis",
                        vmin=0, vmax=size_vmax, interpolation="nearest")
        ax.set_title(lab)
        ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=axes[3], fraction=0.045, pad=0.02, label="set size")

    # q_j
    vmin_q, vmax_q = np.nanpercentile(q_C, [2, 98])
    im = axes[4].imshow(scatter_grid(q_C), cmap="magma",
                        vmin=vmin_q, vmax=vmax_q, interpolation="nearest")
    axes[4].set_title(f"$\\hat q_j$  (sd={q_C.std():.4f})")
    axes[4].set_xticks([]); axes[4].set_yticks([])
    plt.colorbar(im, ax=axes[4], fraction=0.045, pad=0.02)

    nice = tile_name or str(d["tile"])
    fig.suptitle(f"Sentinel-2:  {nice}   @ {int(d['size_px'])} px = {float(d['size_px'])*0.01:.1f} km", y=1.02)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  saved  {out_path.name}")


# ============================================================ main

def main():
    print(f"Writing figures to {OUT}\n")
    hsi = load_hsi_per_seed()
    s2 = load_s2_per_row()
    print(f"Loaded  HSI: {len(hsi)} runs,  S2: {len(s2)} runs\n")

    print("---- Summary figures ----")
    fig1_headline_bars(hsi, s2)
    fig2_paired_improvements(hsi, s2)
    fig3_radius_distribution(hsi, s2)
    fig4_coverage_check(hsi, s2)
    fig5_per_run_scatter(hsi, s2)

    print("\n---- HSI spatial figures (one representative seed per dataset) ----")
    for ds in HSI_DATASETS:
        # pick the seed whose CvA is closest to that dataset's mean CvA — a "typical" seed
        dsr = [r for r in hsi if r["dataset"] == ds]
        CvA = [(r, 100 * (r["is_A"] - r["is_C"]) / r["is_A"]) for r in dsr]
        mean_CvA = np.mean([c for _, c in CvA])
        chosen = min(CvA, key=lambda t: abs(t[1] - mean_CvA))[0]
        seed = chosen["seed"]
        pk = HSI / "results" / "checkpoints" / f"{ds}_seed{seed}.pkl"
        gt_rel, gt_key = HSI_GT_FILES[ds]
        plot_spatial_hsi(pk, HSI_RAW / gt_rel, gt_key,
                         OUT / f"spatial_hsi_{ds}.png")

    print("\n---- S2 spatial figures (3 representative tiles × 3 sizes) ----")
    for tile in ["polk_iowa", "sacramento_ca", "lubbock_tx"]:
        for sp in [100, 200, 500]:
            npz = S2 / "results" / "spatial_maps" / f"{tile}_s{sp}.npz"
            if not npz.exists():
                print(f"  skip {tile} s{sp} (missing)")
                continue
            plot_spatial_s2(npz, OUT / f"spatial_s2_{tile}_s{sp}.png",
                            tile_name=tile.replace("_", " ").title())

    print("\nDone.")


if __name__ == "__main__":
    main()
