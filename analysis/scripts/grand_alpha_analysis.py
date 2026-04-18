"""
Grand-pooled α-sweep (HSI 50 + S2 29 = 79) and HSI-vs-S2 α-sensitivity comparison.

Reads the per-config CSVs produced by alpha_sweep_analysis.py and
s2_alpha_sweep_analysis.py.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as sstats

ROOT = Path(__file__).resolve().parents[2]
ANL = ROOT / "analysis"
OUT_FIG = ANL / "figures"

COLORS = {"D": "#9e9e9e", "A": "#1f77b4", "B": "#2ca02c", "C": "#d62728"}
EXP_COLORS = {"HSI": "#7b3294", "S2": "#008837"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})


def load():
    hsi = pd.read_csv(ANL / "per_seed_alpha.csv")
    s2  = pd.read_csv(ANL / "s2_per_config_alpha.csv")
    hsi["experiment"] = "HSI"
    hsi["config_id"]  = hsi["dataset"] + "_seed" + hsi["seed"].astype(str)
    s2["experiment"]  = "S2"
    s2["config_id"]   = s2["tile"] + "_s" + s2["size_px"].astype(str)
    keep = ["experiment", "config_id", "alpha", "method", "is", "cov", "size", "sscv_pct"]
    return pd.concat([hsi[keep], s2[keep]], ignore_index=True)


def paired_impr(df, ref, target="C"):
    """Per (experiment, alpha) paired (ref - target)/ref % per config."""
    out = []
    for exp in ("HSI", "S2", "GRAND"):
        sub = df if exp == "GRAND" else df[df.experiment == exp]
        for a in sorted(sub.alpha.unique()):
            r = sub[(sub.alpha == a) & (sub.method == ref)].sort_values("config_id").reset_index(drop=True)
            c = sub[(sub.alpha == a) & (sub.method == target)].sort_values("config_id").reset_index(drop=True)
            assert (r.config_id == c.config_id).all(), f"misalign {exp} α={a}"
            impr = (r["is"].values - c["is"].values) / r["is"].values * 100
            t, p = sstats.ttest_1samp(impr, 0.0)
            out.append({"experiment": exp, "alpha": a, "comparison": f"{target} vs {ref}",
                        "mean_pct": impr.mean(),
                        "sem_val": impr.std(ddof=1) / np.sqrt(len(impr)),
                        "t": t, "p": p, "n": len(impr)})
    return pd.DataFrame(out)


def plot_alpha_sensitivity(paired_A, paired_D):
    """Side-by-side: HSI vs S2 α-sensitivity for both C-vs-A and C-vs-D."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=False)
    for ax, paired, ref in zip(axes, (paired_A, paired_D), ("A", "D")):
        for exp in ("HSI", "S2"):
            sub = paired[(paired.experiment == exp)].sort_values("alpha")
            ax.errorbar(sub.alpha, sub.mean_pct, yerr=sub.sem_val,
                        marker="o", ms=8, lw=2, capsize=4,
                        color=EXP_COLORS[exp], label=f"{exp} (n={int(sub.iloc[0].n)})")
            for _, row in sub.iterrows():
                p = row["p"]
                stars = "****" if p < 1e-4 else "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "ns"
                ax.text(row.alpha, row.mean_pct + row.sem_val + 0.3, stars,
                        ha="center", va="bottom", fontsize=8, color=EXP_COLORS[exp])
        ax.set_xlabel("Target miscoverage α")
        ax.set_ylabel(f"Paired IS improvement: C vs {ref} (%)")
        ax.set_title(f"C vs {ref}: α-sensitivity by experiment")
        ax.axhline(0, color="black", lw=0.6)
        ax.set_xticks([0.05, 0.10, 0.15])
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right" if ref == "A" else "best", frameon=False)
    fig.suptitle("HSI vs S2 — does GeoCP's local-threshold gain depend on α?", y=1.02)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "fig12_alpha_sensitivity_HSI_vs_S2.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_grand_pooled(paired_A, paired_D):
    """Bar chart: grand-pooled C-vs-A and C-vs-D per α (n=79)."""
    grand_A = paired_A[paired_A.experiment == "GRAND"].sort_values("alpha")
    grand_D = paired_D[paired_D.experiment == "GRAND"].sort_values("alpha")
    fig, ax = plt.subplots(figsize=(7, 3.8))
    alphas = grand_A.alpha.values
    xs = np.arange(len(alphas)); w = 0.35
    for i, (df_, ref) in enumerate([(grand_A, "A"), (grand_D, "D")]):
        ax.bar(xs + (i - 0.5) * w, df_.mean_pct.values, w, yerr=df_.sem_val.values,
               capsize=3, color=COLORS[ref], label=f"C vs {ref}",
               edgecolor="black", linewidth=0.4)
        for j, (_, row) in enumerate(df_.iterrows()):
            p = row["p"]
            stars = "****" if p < 1e-4 else "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "ns"
            ax.text(xs[j] + (i - 0.5) * w, row.mean_pct + row.sem_val + 0.3, stars,
                    ha="center", va="bottom", fontsize=9)
    ax.set_xticks(xs); ax.set_xticklabels([f"α={a:.2f}" for a in alphas])
    ax.set_ylabel("Paired IS improvement (%, mean ± s.e.m.)")
    ax.set_title("Grand-pooled (HSI 50 + S2 29 = 79) C vs {A, D}")
    ax.axhline(0, color="black", lw=0.6); ax.legend(loc="upper right", frameon=False)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_FIG / "fig13_grand_alpha_improvement.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    df = load()
    print(f"Loaded {len(df)} rows; experiments: {df.experiment.value_counts().to_dict()}")
    print(f"configs/exp: HSI={df[df.experiment=='HSI'].config_id.nunique()}, "
          f"S2={df[df.experiment=='S2'].config_id.nunique()}")

    paired_A = paired_impr(df, ref="A", target="C")
    paired_D = paired_impr(df, ref="D", target="C")
    paired_A.to_csv(ANL / "alpha_sensitivity_C_vs_A.csv", index=False)
    paired_D.to_csv(ANL / "alpha_sensitivity_C_vs_D.csv", index=False)

    print("\n=== C vs A — by experiment & α ===")
    print(paired_A.round(4).to_string(index=False))
    print("\n=== C vs D — by experiment & α ===")
    print(paired_D.round(4).to_string(index=False))

    plot_alpha_sensitivity(paired_A, paired_D)
    plot_grand_pooled(paired_A, paired_D)
    print(f"\nSaved: {OUT_FIG}/fig12_alpha_sensitivity_HSI_vs_S2.png, fig13_grand_alpha_improvement.png")
    print(f"CSVs : {ANL}/alpha_sensitivity_C_vs_A.csv, alpha_sensitivity_C_vs_D.csv")


if __name__ == "__main__":
    main()
