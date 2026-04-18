"""
α-sweep + SSCV analyses for HSI experiment (50 pkl from hsi_alpha_sweep).

Reads pkls from Drive-synced local path; writes figures + CSVs next to the
existing analysis/figures directory.
"""
import pickle, os
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as sstats

DRIVE_CKPT = Path("/Users/peng/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/"
                  "My Drive/hsi_alpha_sweep/checkpoints")
ROOT = Path(__file__).resolve().parents[2]
OUT_FIG = ROOT / "analysis" / "figures"
OUT = ROOT / "analysis"

COLORS = {"D": "#9e9e9e", "A": "#1f77b4", "B": "#2ca02c", "C": "#d62728"}
METHODS = ("D", "A", "B", "C")
DATASETS = ["ip", "pu", "sa", "ksc", "botswana"]
NICE = {"ip": "Indian Pines", "pu": "Pavia Univ.", "sa": "Salinas",
        "ksc": "KSC", "botswana": "Botswana"}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})


def load_all():
    rows = []
    for p in sorted(DRIVE_CKPT.glob("*.pkl")):
        with open(p, "rb") as f:
            d = pickle.load(f)
        for a_str, block in d["per_alpha"].items():
            for m in METHODS:
                b = block[m]
                rows.append({"dataset": d["dataset"], "seed": d["seed"],
                             "alpha": float(a_str), "method": m,
                             "is": b["is"], "cov": b["cov"], "size": b["size"],
                             "sscv_pct": b["sscv_pct"]})
    return pd.DataFrame(rows)


def pooled_paired_tests(df):
    """For each α, compute (A-C)/A % and (D-C)/D % paired tests pooled n=50."""
    out = []
    for a in sorted(df.alpha.unique()):
        for ref in ("A", "D"):
            sub_ref = df[(df.alpha == a) & (df.method == ref)].sort_values(["dataset", "seed"]).reset_index(drop=True)
            sub_C   = df[(df.alpha == a) & (df.method == "C")].sort_values(["dataset", "seed"]).reset_index(drop=True)
            impr = (sub_ref["is"].values - sub_C["is"].values) / sub_ref["is"].values * 100
            t, p = sstats.ttest_1samp(impr, 0.0)
            out.append({"alpha": a, "comparison": f"C vs {ref}",
                        "mean_pct": impr.mean(), "sem_val": impr.std(ddof=1)/np.sqrt(len(impr)),
                        "t": t, "p": p, "n": len(impr)})
    return pd.DataFrame(out)


def plot_alpha_sweep_is(df):
    """Panel per α, bars of mean IS per method per dataset."""
    alphas = sorted(df.alpha.unique())
    fig, axes = plt.subplots(1, len(alphas), figsize=(14, 3.8), sharey=True)
    xs = np.arange(len(DATASETS))
    w = 0.2
    for ax, a in zip(axes, alphas):
        for i, m in enumerate(METHODS):
            means, errs = [], []
            for ds in DATASETS:
                sub = df[(df.dataset == ds) & (df.method == m) & (df.alpha == a)]
                means.append(sub["is"].mean())
                errs.append(sub["is"].std(ddof=1) / np.sqrt(len(sub)))
            ax.bar(xs + (i - 1.5) * w, means, w, yerr=errs, capsize=2,
                   color=COLORS[m], label=f"({m})", edgecolor="black", linewidth=0.4)
        ax.set_xticks(xs)
        ax.set_xticklabels([NICE[d] for d in DATASETS], rotation=20, ha="right")
        ax.set_title(f"α = {a:.2f}")
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].set_ylabel("Interval Score (mean ± s.e.m., lower better)")
    axes[-1].legend(loc="upper right", frameon=False, fontsize=9)
    fig.suptitle("Method IS across α ∈ {0.05, 0.10, 0.15} — CV-selected (r, bw) fixed at α=0.10",
                 y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "fig8_alpha_sweep_is.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_alpha_improvement(paired_df):
    """Paired improvement of C vs A and C vs D across α."""
    fig, ax = plt.subplots(figsize=(7, 3.8))
    alphas = sorted(paired_df.alpha.unique())
    xs = np.arange(len(alphas))
    w = 0.35
    for i, ref in enumerate(("A", "D")):
        sub = paired_df[paired_df.comparison == f"C vs {ref}"].sort_values("alpha")
        color = COLORS[ref]
        ax.bar(xs + (i - 0.5) * w, sub["mean_pct"].values, w, yerr=sub["sem_val"].values,
               capsize=3, color=color, label=f"C vs {ref}", edgecolor="black", linewidth=0.4)
        for j, (_, row) in enumerate(sub.iterrows()):
            p = row["p"]
            stars = "****" if p < 1e-4 else "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "ns"
            ax.text(xs[j] + (i - 0.5) * w, row["mean_pct"] + row["sem_val"] + 0.3, stars,
                    ha="center", va="bottom", fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"α={a:.2f}" for a in alphas])
    ax.set_ylabel("Paired IS improvement (%, higher better for C)")
    ax.set_title("C vs {A, D} pooled paired improvement (n=50 per α)")
    ax.axhline(0, color="black", lw=0.6)
    ax.legend(loc="upper right", frameon=False)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "fig9_alpha_improvement.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_sscv_across_alpha(df):
    """SSCV per α per method (pooled boxplots)."""
    alphas = sorted(df.alpha.unique())
    fig, axes = plt.subplots(1, len(alphas), figsize=(12, 3.6), sharey=True)
    for ax, a in zip(axes, alphas):
        data = [df[(df.method == m) & (df.alpha == a)].sscv_pct.values for m in METHODS]
        bp = ax.boxplot(data, tick_labels=list(METHODS), widths=0.55,
                        patch_artist=True, showfliers=False)
        for patch, m in zip(bp["boxes"], METHODS):
            patch.set_facecolor(COLORS[m]); patch.set_alpha(0.7)
        ax.set_title(f"α = {a:.2f}")
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].set_ylabel("SSCV (%, lower is better)")
    fig.suptitle("Size-Stratified Coverage Violation across α (pooled n=50)", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "fig10_sscv_alpha.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_coverage_across_alpha(df):
    """Mean coverage per α per method, with target line."""
    alphas = sorted(df.alpha.unique())
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    xs = np.arange(len(alphas))
    w = 0.2
    for i, m in enumerate(METHODS):
        means = [df[(df.method == m) & (df.alpha == a)]["cov"].mean() for a in alphas]
        errs  = [df[(df.method == m) & (df.alpha == a)]["cov"].std(ddof=1) /
                  np.sqrt(len(df[(df.method == m) & (df.alpha == a)])) for a in alphas]
        ax.bar(xs + (i - 1.5) * w, means, w, yerr=errs, capsize=3,
               color=COLORS[m], label=f"({m})", edgecolor="black", linewidth=0.4)
    for j, a in enumerate(alphas):
        ax.axhline(1 - a, xmin=(j + 0.05) / len(alphas), xmax=(j + 0.95) / len(alphas),
                   color="red", lw=1.2, ls="--")
    ax.set_xticks(xs); ax.set_xticklabels([f"α={a:.2f}\n(target={1-a:.2f})" for a in alphas])
    ax.set_ylabel("Empirical coverage"); ax.set_ylim(0.80, 1.0)
    ax.set_title("Coverage vs α — all methods satisfy target (red line)")
    ax.legend(loc="lower right", ncol=4, frameon=False, fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "fig11_coverage_alpha.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    print("Loading 50 pkl from Drive-synced path...")
    df = load_all()
    print(f"Loaded {len(df)} (seed, α, method) rows "
          f"({df.dataset.nunique()} datasets, {df.seed.nunique()} seeds, {df.alpha.nunique()} α, {df.method.nunique()} methods).")

    # CSVs
    df.to_csv(OUT / "per_seed_alpha.csv", index=False)
    pooled = (df.groupby(["alpha", "method"])
                .agg(is_mean=("is", "mean"), is_std=("is", "std"),
                     cov_mean=("cov", "mean"), size_mean=("size", "mean"),
                     sscv_mean=("sscv_pct", "mean"), sscv_std=("sscv_pct", "std"),
                     n=("seed", "count")).reset_index())
    pooled.to_csv(OUT / "alpha_sweep_pooled.csv", index=False)

    paired = pooled_paired_tests(df)
    paired.to_csv(OUT / "alpha_sweep_paired_tests.csv", index=False)

    print("\n=== Pooled α-sweep (n=50 per α × method) ===")
    print(pooled.round(3).to_string(index=False))
    print("\n=== Paired tests (C vs A, C vs D) per α ===")
    print(paired.round(4).to_string(index=False))

    # Figures
    plot_alpha_sweep_is(df)
    plot_alpha_improvement(paired)
    plot_sscv_across_alpha(df)
    plot_coverage_across_alpha(df)
    print(f"\nSaved: {OUT_FIG}/fig8_alpha_sweep_is.png, fig9_alpha_improvement.png, "
          f"fig10_sscv_alpha.png, fig11_coverage_alpha.png")
    print(f"CSVs : {OUT}/per_seed_alpha.csv, alpha_sweep_pooled.csv, alpha_sweep_paired_tests.csv")


if __name__ == "__main__":
    main()
