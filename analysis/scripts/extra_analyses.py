"""
SSCV + Size-vs-Difficulty analyses for HSI experiment (50 pkl).

SSCV: Size-Stratified Coverage Violation (Angelopoulos et al.). For each method,
bucket test samples by predicted-set size, compute |coverage_bucket - (1-alpha)|,
take the max across buckets. Smaller = more fair conditional coverage.

Size-vs-Difficulty: bucket test samples by Standard-CP set size as a difficulty
proxy (true-label rank unavailable without softmax). For each method, report
mean set size per difficulty bucket. Analogous to Fig 4 in Liu 2024.
"""
import pickle, os, json
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
CKPT = ROOT / "hsi_experiment" / "results" / "checkpoints"
OUT_FIG = ROOT / "analysis" / "figures"
OUT_FIG.mkdir(exist_ok=True, parents=True)

METHODS = [("standard_cp", "D"), ("sacp_default", "A"),
           ("sacp_cv_r", "B"), ("sacp_geocp", "C")]
COLORS = {"D": "#9e9e9e", "A": "#1f77b4", "B": "#2ca02c", "C": "#d62728"}
DATASETS = ["ip", "pu", "sa", "ksc", "botswana"]
NICE = {"ip": "Indian Pines", "pu": "Pavia Univ.", "sa": "Salinas",
        "ksc": "KSC", "botswana": "Botswana"}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})


def load_all():
    rows = []
    for p in sorted(CKPT.glob("*.pkl")):
        with open(p, "rb") as f:
            d = pickle.load(f)
        rec = {"dataset": d["dataset"], "seed": d["seed"],
               "alpha": d["alpha"], "y_te": np.asarray(d["y_te"])}
        for m, _ in METHODS:
            rec[m + "_sets"] = [set(s) for s in d[m]["pred_sets"]]
        rows.append(rec)
    return rows


def sscv(sets, y, alpha, buckets=((1, 1), (2, 2), (3, 3), (4, 4), (5, 10**9))):
    """Max |cov_bucket - (1-alpha)| across size buckets, ignoring empty buckets."""
    sizes = np.array([len(s) for s in sets])
    covered = np.array([y[i] in sets[i] for i in range(len(y))])
    worst = 0.0
    per_bucket = []
    for lo, hi in buckets:
        mask = (sizes >= lo) & (sizes <= hi)
        n = mask.sum()
        if n == 0:
            per_bucket.append((lo, hi, n, np.nan))
            continue
        cov = covered[mask].mean()
        per_bucket.append((lo, hi, int(n), float(cov)))
        worst = max(worst, abs(cov - (1 - alpha)))
    return worst, per_bucket


def difficulty_bucket(size_D):
    """Bucket Standard-CP set size into {1,2,3,4,>=5}."""
    if size_D <= 1: return 1
    if size_D <= 4: return int(size_D)
    return 5


def analyze(rows):
    # === SSCV ===
    sscv_rows = []
    for rec in rows:
        for m, lab in METHODS:
            w, _ = sscv(rec[m + "_sets"], rec["y_te"], rec["alpha"])
            sscv_rows.append({"dataset": rec["dataset"], "seed": rec["seed"],
                              "method": lab, "sscv": w * 100})
    sscv_df = pd.DataFrame(sscv_rows)

    # === Size-vs-Difficulty ===
    dif_rows = []
    for rec in rows:
        sets_D = rec["standard_cp_sets"]
        diffs = np.array([difficulty_bucket(len(s)) for s in sets_D])
        y = rec["y_te"]
        for m, lab in METHODS:
            sets_m = rec[m + "_sets"]
            sizes_m = np.array([len(s) for s in sets_m])
            covered = np.array([y[i] in sets_m[i] for i in range(len(y))])
            for b in range(1, 6):
                mask = diffs == b
                if mask.sum() == 0:
                    continue
                dif_rows.append({"dataset": rec["dataset"], "seed": rec["seed"],
                                 "method": lab, "difficulty": b,
                                 "mean_size": float(sizes_m[mask].mean()),
                                 "coverage": float(covered[mask].mean()),
                                 "n": int(mask.sum())})
    dif_df = pd.DataFrame(dif_rows)
    return sscv_df, dif_df


def plot_sscv(sscv_df):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [3, 1]})

    ax = axes[0]
    xs = np.arange(len(DATASETS))
    w = 0.2
    for i, (_, lab) in enumerate(METHODS):
        means, errs = [], []
        for d in DATASETS:
            sub = sscv_df[(sscv_df.dataset == d) & (sscv_df.method == lab)]
            means.append(sub.sscv.mean())
            errs.append(sub.sscv.std() / np.sqrt(len(sub)))
        ax.bar(xs + (i - 1.5) * w, means, w, yerr=errs, capsize=2,
               color=COLORS[lab], label=f"({lab})", edgecolor="black", linewidth=0.4)
    ax.set_xticks(xs)
    ax.set_xticklabels([NICE[d] for d in DATASETS], rotation=15)
    ax.set_ylabel("SSCV (%, lower is better)")
    ax.set_title("Size-Stratified Coverage Violation — per dataset (mean ± s.e.m., n=10)")
    ax.legend(loc="upper right", ncol=4, frameon=False, fontsize=9)

    ax = axes[1]
    pooled = [sscv_df[sscv_df.method == lab].sscv.values for _, lab in METHODS]
    bp = ax.boxplot(pooled, labels=[lab for _, lab in METHODS], widths=0.55,
                    patch_artist=True, showfliers=False)
    for patch, (_, lab) in zip(bp["boxes"], METHODS):
        patch.set_facecolor(COLORS[lab]); patch.set_alpha(0.7)
    ax.set_ylabel("SSCV (%)")
    ax.set_title("Pooled (n=50)")

    fig.tight_layout()
    fig.savefig(OUT_FIG / "fig6_sscv.png", dpi=180)
    plt.close(fig)


def plot_difficulty(dif_df):
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.3), sharey=True)
    xs = np.arange(1, 6)
    xticklabels = ["1", "2", "3", "4", "≥5"]
    for ax, d in zip(axes, DATASETS):
        for _, lab in METHODS:
            sub = dif_df[(dif_df.dataset == d) & (dif_df.method == lab)]
            means = sub.groupby("difficulty").mean_size.mean().reindex(xs).values
            errs = sub.groupby("difficulty").mean_size.apply(
                lambda x: x.std() / np.sqrt(len(x))).reindex(xs).values
            ax.errorbar(xs, means, yerr=errs, marker="o", ms=5, lw=1.6, capsize=2,
                        color=COLORS[lab], label=f"({lab})")
        ax.set_xticks(xs); ax.set_xticklabels(xticklabels)
        ax.set_title(NICE[d])
        ax.set_xlabel("Difficulty (Standard-CP size bucket)")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Mean prediction-set size")
    axes[-1].legend(loc="upper left", frameon=False, fontsize=9)
    fig.suptitle("Adaptiveness: mean set size vs difficulty bucket "
                 "(difficulty = Standard-CP set size as proxy)", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "fig7_size_vs_difficulty.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def summaries(sscv_df, dif_df, out_dir):
    # SSCV pooled table
    rows = []
    for _, lab in METHODS:
        s = sscv_df[sscv_df.method == lab].sscv
        rows.append({"method": lab, "mean_sscv": s.mean(), "median_sscv": s.median(),
                     "std": s.std(), "n": len(s)})
    pooled = pd.DataFrame(rows)
    pooled.to_csv(out_dir / "sscv_pooled.csv", index=False)

    per_ds = sscv_df.groupby(["dataset", "method"]).sscv.agg(["mean", "std"]).reset_index()
    per_ds.to_csv(out_dir / "sscv_per_dataset.csv", index=False)

    # Size-vs-difficulty table
    dif_agg = (dif_df.groupby(["dataset", "difficulty", "method"])
                     .agg(mean_size=("mean_size", "mean"),
                          coverage=("coverage", "mean"),
                          n=("n", "sum")).reset_index())
    dif_agg.to_csv(out_dir / "size_vs_difficulty.csv", index=False)

    # Pooled Size-vs-Difficulty
    pooled_dif = (dif_df.groupby(["difficulty", "method"])
                         .mean_size.mean().unstack("method").reset_index())
    pooled_dif.to_csv(out_dir / "size_vs_difficulty_pooled.csv", index=False)

    return pooled, pooled_dif


def main():
    print("Loading 50 HSI pkl...")
    rows = load_all()
    print(f"Loaded {len(rows)} runs.")
    sscv_df, dif_df = analyze(rows)
    out_dir = ROOT / "analysis"
    pooled, pooled_dif = summaries(sscv_df, dif_df, out_dir)
    print("\n=== SSCV pooled (n=50) ===")
    print(pooled.to_string(index=False))
    print("\n=== Size-vs-Difficulty pooled (mean set size) ===")
    print(pooled_dif.to_string(index=False))
    plot_sscv(sscv_df)
    plot_difficulty(dif_df)
    print(f"\nFigures saved to: {OUT_FIG}/fig6_sscv.png, fig7_size_vs_difficulty.png")
    print(f"Tables saved to: {out_dir}/sscv_*.csv, size_vs_difficulty*.csv")


if __name__ == "__main__":
    main()
