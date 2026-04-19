"""
Aggregate SSFormer (Transformer-based HSI) results and compare against 3D-CNN.

Builds:
  - per_seed_alpha_sstn.csv  (SSFormer pooled per (seed, α, method))
  - backbone_comparison.csv  (3D-CNN vs SSFormer per dataset)
  - accuracy_vs_improvement_10pts.csv  (10 points: 5 HSI × 2 backbones)

Prints:
  - Per-dataset C vs A and C vs D mean improvements under SSFormer
  - Pooled + cluster-aggregated significance tests
  - Pearson r with the 10-point dataset × backbone table
"""
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats as sstats

CKPT_SSTN = Path("/Users/peng/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/"
                 "My Drive/hsi_sstn_alpha_sweep/checkpoints")
ROOT = Path("/Users/peng/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/"
            "My Drive/Research/0_GeoCP_LISA/GeoCP_RS/analysis")

# --- Load SSFormer pkls ---
rows = []
for p in sorted(CKPT_SSTN.glob("*.pkl")):
    with open(p, "rb") as f:
        d = pickle.load(f)
    for a_str, block in d["per_alpha"].items():
        for m in ("D", "A", "B", "C"):
            b = block[m]
            rows.append({"backbone": "SSFormer", "dataset": d["dataset"], "seed": d["seed"],
                         "alpha": float(a_str), "method": m,
                         "is": b["is"], "cov": b["cov"], "size": b["size"], "sscv_pct": b["sscv_pct"],
                         "accuracy": d["accuracy"],
                         "r_C": d["best_r_geocp"], "bw_C": d["best_bw_geocp"]})
df_sstn = pd.DataFrame(rows)
df_sstn.to_csv(ROOT / "per_seed_alpha_sstn.csv", index=False)
print(f"Wrote per_seed_alpha_sstn.csv ({len(df_sstn)} rows, "
      f"{df_sstn.dataset.nunique()} datasets, {df_sstn.seed.nunique()} seeds)")

df_cnn = pd.read_csv(ROOT / "per_seed_alpha.csv")
df_cnn["backbone"] = "3D-CNN"

df_both = pd.concat([df_cnn, df_sstn], ignore_index=True)

# --- Per-dataset × backbone means at α=0.10 ---
print("\n=== Accuracy and C-vs-A improvement per (dataset, backbone) at α=0.10 ===")
out_rows = []
for (bb, ds), sub in df_both[df_both.alpha == 0.10].groupby(["backbone", "dataset"]):
    acc = sub["accuracy"].mean()
    A = sub[sub.method == "A"].sort_values("seed")["is"].values
    C = sub[sub.method == "C"].sort_values("seed")["is"].values
    D = sub[sub.method == "D"].sort_values("seed")["is"].values
    impr_CA = (A - C) / A * 100
    impr_CD = (D - C) / D * 100
    out_rows.append({"backbone": bb, "dataset": ds, "accuracy": acc,
                     "C_vs_A_pct": impr_CA.mean(), "C_vs_A_sem": impr_CA.std(ddof=1)/np.sqrt(len(impr_CA)),
                     "C_vs_D_pct": impr_CD.mean(), "C_vs_D_sem": impr_CD.std(ddof=1)/np.sqrt(len(impr_CD))})
comp_df = pd.DataFrame(out_rows)
comp_df.to_csv(ROOT / "backbone_comparison.csv", index=False)
print(comp_df.round(3).to_string(index=False))

# --- 10-point accuracy-vs-improvement Pearson ---
print("\n=== Accuracy-vs-improvement correlation (10 points: 5 datasets × 2 backbones) ===")
r_CA, p_CA = sstats.pearsonr(comp_df["accuracy"], comp_df["C_vs_A_pct"])
r_CD, p_CD = sstats.pearsonr(comp_df["accuracy"], comp_df["C_vs_D_pct"])
print(f"  C vs A: Pearson r = {r_CA:+.3f}, p = {p_CA:.4f}")
print(f"  C vs D: Pearson r = {r_CD:+.3f}, p = {p_CD:.4f}")
comp_df.to_csv(ROOT / "accuracy_vs_improvement_10pts.csv", index=False)

# --- Pooled paired + cluster paired tests for SSFormer ---
print("\n=== SSFormer paired tests (C vs A, C vs D) ===")
for a in sorted(df_sstn.alpha.unique()):
    for ref in ("A", "D"):
        sub_r = df_sstn[(df_sstn.alpha == a) & (df_sstn.method == ref)].sort_values(["dataset","seed"]).reset_index(drop=True)
        sub_c = df_sstn[(df_sstn.alpha == a) & (df_sstn.method == "C")].sort_values(["dataset","seed"]).reset_index(drop=True)
        impr = (sub_r["is"].values - sub_c["is"].values) / sub_r["is"].values * 100
        t_p, p_p = sstats.ttest_1samp(impr, 0.0)
        # Cluster: per-dataset mean
        tmp = pd.DataFrame({"dataset": sub_r["dataset"].values, "impr": impr})
        clu = tmp.groupby("dataset")["impr"].mean().values
        t_c, p_c = sstats.ttest_1samp(clu, 0.0) if len(clu) > 1 else (float("nan"), float("nan"))
        print(f"  α={a:.2f}  C vs {ref}:  pooled {impr.mean():+.2f}% ± {impr.std(ddof=1)/np.sqrt(len(impr)):.2f}  "
              f"(n={len(impr)}, p={p_p:.2e})  ||  cluster {clu.mean():+.2f}% (n={len(clu)}, p={p_c:.3e})")

# --- Radius selection under SSFormer ---
print("\n=== CV-selected r under SSFormer (method C) ===")
r_counts = df_sstn.drop_duplicates(["dataset","seed"])["r_C"].value_counts().sort_index()
print(r_counts.to_string())
print(f"r=1 chosen: {(df_sstn.drop_duplicates(['dataset','seed'])['r_C']==1).sum()} / 50")
