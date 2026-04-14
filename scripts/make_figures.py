"""Regenerate the paper figures from ``results/per_seed.csv``.

Output PDFs and PNGs are deterministic given the same input CSV.

Usage:
    python -m scripts.make_figures --in ./results --out ./figures
    geocp-rs-figures --in ./results --out ./figures
"""
from __future__ import annotations
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


DATASETS = ["ip", "pu", "sa", "ksc", "botswana"]
NICE = {"ip": "Indian Pines", "pu": "Pavia U.", "sa": "Salinas",
        "ksc": "KSC", "botswana": "Botswana"}
COLORS = {"ip": "#1f77b4", "pu": "#ff7f0e", "sa": "#2ca02c",
          "ksc": "#d62728", "botswana": "#9467bd"}


def fig_is_bars(df: pd.DataFrame, out_dir: str) -> None:
    methods = ["std_cp_is", "sacp03_is", "sacp05_is", "sacp07_is", "geocp_is"]
    labels = ["Standard\nCP", "SACP\n$\\lambda$=0.3", "SACP\n$\\lambda$=0.5",
              "SACP\n$\\lambda$=0.7", "SACP+GeoCP\n(CV bw)"]
    cols = ["#aaaaaa", "#bcd3e8", "#7fb2d9", "#3e87c2", "#c94040"]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    x = np.arange(len(DATASETS))
    w = 0.16
    for i, (m, lbl, clr) in enumerate(zip(methods, labels, cols)):
        means = [df[df.dataset == ds][m].mean() for ds in DATASETS]
        stds = [df[df.dataset == ds][m].std() for ds in DATASETS]
        ax.bar(x + (i - 2) * w, means, w, yerr=stds, label=lbl, color=clr,
               edgecolor="black", lw=0.6, capsize=2, ecolor="dimgray",
               error_kw={"lw": 0.8})
    ax.set_xticks(x)
    ax.set_xticklabels([NICE[d] for d in DATASETS], fontsize=11)
    ax.set_ylabel("Interval Score (lower is better)", fontsize=12)
    ax.set_title("SACP+GeoCP achieves lowest IS on all 5 HSI datasets "
                 "(10 seeds, error bars = std)", fontsize=11)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.12),
              fontsize=9, frameon=False)
    ax.grid(axis="y", alpha=0.3)
    stars = {"ip": "***", "pu": "*", "sa": "*", "ksc": "", "botswana": ""}
    for i, ds in enumerate(DATASETS):
        if stars[ds]:
            ymax = df[df.dataset == ds][methods].max().max()
            ax.annotate(stars[ds], (x[i] + 2 * w, ymax * 1.02),
                        ha="center", fontsize=13,
                        color="darkred", fontweight="bold")
    plt.tight_layout()
    _savefig(fig, os.path.join(out_dir, "fig_is_bars"))


def fig_acc_vs_imp(df: pd.DataFrame, out_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    accs, imps = [], []
    for ds in DATASETS:
        sub = df[df.dataset == ds]
        acc = sub.accuracy.mean()
        imp = (sub.sacp05_is.mean() - sub.geocp_is.mean()) / sub.sacp05_is.mean() * 100
        accs.append(acc); imps.append(imp)
        ax.errorbar(acc, imp, xerr=sub.accuracy.std(),
                    yerr=(((sub.sacp05_is - sub.geocp_is) / sub.sacp05_is * 100).std()),
                    fmt="o", ms=11, color=COLORS[ds], ecolor="gray",
                    elinewidth=1, capsize=3, zorder=3)
        ax.annotate(NICE[ds], (acc, imp), textcoords="offset points",
                    xytext=(10, 6), fontsize=10)
    r, p = stats.pearsonr(accs, imps)
    slope, intercept = np.polyfit(accs, imps, 1)
    xs = np.linspace(0.65, 0.97, 100)
    ax.plot(xs, slope * xs + intercept, "--", color="gray", alpha=0.6,
            lw=1.5, zorder=1, label=f"linear fit (r={r:.2f}, p={p:.3f})")
    ax.axhline(0, color="k", lw=0.5, alpha=0.3)
    ax.set_xlabel("Classifier accuracy (3D-CNN)", fontsize=12)
    ax.set_ylabel("SACP+GeoCP improvement ($\\Delta$ IS %)", fontsize=12)
    ax.set_title("Improvement from SACP+GeoCP scales inversely with accuracy",
                 fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.set_xlim(0.64, 0.98)
    plt.tight_layout()
    _savefig(fig, os.path.join(out_dir, "fig_acc_vs_improvement"))


def fig_cov_vs_size(df: pd.DataFrame, out_dir: str) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.4), sharey=False)
    for ax, ds in zip(axes, DATASETS):
        sub = df[df.dataset == ds]
        ax.errorbar(sub.std_cp_size.mean(), sub.std_cp_cov.mean(),
                    xerr=sub.std_cp_size.std(), yerr=sub.std_cp_cov.std(),
                    fmt="s", ms=10, color="gray", label="Standard CP",
                    capsize=3)
        for lmd, col in [(0.3, "#bcd3e8"), (0.5, "#7fb2d9"), (0.7, "#3e87c2")]:
            sz = sub[f"sacp{int(lmd*10):02d}_size"]
            cv = sub[f"sacp{int(lmd*10):02d}_cov"]
            ax.errorbar(sz.mean(), cv.mean(), xerr=sz.std(), yerr=cv.std(),
                        fmt="o", ms=9, color=col,
                        label=f"SACP($\\lambda$={lmd})", capsize=3)
        ax.errorbar(sub.geocp_size.mean(), sub.geocp_cov.mean(),
                    xerr=sub.geocp_size.std(), yerr=sub.geocp_cov.std(),
                    fmt="*", ms=18, color="#c94040", label="SACP+GeoCP",
                    capsize=3, markeredgecolor="black", markeredgewidth=0.5)
        ax.axhline(0.9, color="k", ls=":", lw=0.8, alpha=0.5)
        ax.set_title(NICE[ds], fontsize=11)
        ax.set_xlabel("Avg. pred-set size", fontsize=10)
        if ds == DATASETS[0]:
            ax.set_ylabel("Coverage", fontsize=11)
        ax.grid(alpha=0.3)
    axes[-1].legend(loc="lower right", fontsize=7, framealpha=0.9)
    plt.suptitle("Coverage vs. set size: SACP+GeoCP in the upper-left corner",
                 fontsize=11, y=1.03)
    plt.tight_layout()
    _savefig(fig, os.path.join(out_dir, "fig_coverage_vs_size"))


def fig_bandwidth_selection(df: pd.DataFrame, out_dir: str) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(15, 3), sharey=True)
    for ax, ds in zip(axes, DATASETS):
        bws = df[df.dataset == ds].geocp_bw.values
        bw_list = [3, 5, 7, 10, 15, 20, 30, 50, 100]
        counts = [int((bws == b).sum()) for b in bw_list]
        ax.bar(range(len(bw_list)), counts, color=COLORS[ds],
               edgecolor="black", lw=0.6)
        ax.set_xticks(range(len(bw_list)))
        ax.set_xticklabels(bw_list, fontsize=8)
        ax.set_title(f"{NICE[ds]}\n(median={int(np.median(bws))})", fontsize=10)
        ax.set_xlabel("bandwidth (pixels)", fontsize=9)
        if ds == DATASETS[0]:
            ax.set_ylabel("# seeds (out of 10)", fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    plt.suptitle("5-fold CV selects bandwidth aligned with each dataset's spatial scale",
                 fontsize=11, y=1.03)
    plt.tight_layout()
    _savefig(fig, os.path.join(out_dir, "fig_bandwidth_selection"))


def _savefig(fig, base: str) -> None:
    fig.savefig(base + ".pdf", bbox_inches="tight")
    fig.savefig(base + ".png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {base}.pdf + .png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_dir", default="./results",
                    help="Directory containing per_seed.csv")
    ap.add_argument("--out", default="./figures",
                    help="Output directory for generated figures")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    df = pd.read_csv(os.path.join(args.in_dir, "per_seed.csv"))
    fig_is_bars(df, args.out)
    fig_acc_vs_imp(df, args.out)
    fig_cov_vs_size(df, args.out)
    fig_bandwidth_selection(df, args.out)


if __name__ == "__main__":
    main()
