"""
Cluster-aggregated significance tests for reviewer #5.

Addresses: per-seed (HSI) and per-scale (S2) observations within the same
dataset / tile are not independent. Pooled t-tests on n=79 therefore over-
state significance. We report two parallel analyses:

  (I) Pooled (seed-level): the original n=79 test, keep as upper-bound.
 (II) Clustered: aggregate to per-dataset (HSI, n=5) and per-tile (S2, n=10)
      means, then run the paired t-test on 15 independent cluster means.

Reports mean ± s.e.m., t, p, and effective n for both.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats as sstats

ROOT = Path("/Users/peng/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/"
            "My Drive/Research/0_GeoCP_LISA/GeoCP_RS/analysis")

df_hsi = pd.read_csv(ROOT / "per_seed_alpha.csv")
df_s2  = pd.read_csv(ROOT / "s2_per_config_alpha.csv")

df_hsi["experiment"] = "HSI"; df_hsi["cluster"] = df_hsi["dataset"]
df_hsi["config_id"] = df_hsi["dataset"] + "_seed" + df_hsi["seed"].astype(str)

df_s2["experiment"]  = "S2"; df_s2["cluster"]  = df_s2["tile"]
df_s2["config_id"]   = df_s2["tile"] + "_s" + df_s2["size_px"].astype(str)

df_all = pd.concat([df_hsi, df_s2], ignore_index=True)


def paired_improvements(df, ref="A", target="C"):
    """Long-format (run, alpha) dataframe of percent improvement (ref - target)/ref."""
    rows = []
    for (exp, cid), g in df.groupby(["experiment", "config_id"]):
        clu = g["cluster"].iloc[0]
        for a in sorted(g.alpha.unique()):
            sub = g[g.alpha == a]
            r_val = sub.loc[sub.method == ref, "is"].values
            c_val = sub.loc[sub.method == target, "is"].values
            if len(r_val) == 0 or len(c_val) == 0:
                continue
            impr = (r_val[0] - c_val[0]) / r_val[0] * 100
            rows.append({"experiment": exp, "cluster": clu, "config_id": cid,
                         "alpha": a, "comparison": f"{target} vs {ref}",
                         "impr_pct": float(impr)})
    return pd.DataFrame(rows)


def tests(df_impr, exp=None):
    """Return pooled (seed-level) and clustered (per-dataset/tile-mean) t-tests."""
    if exp is not None:
        df_impr = df_impr[df_impr.experiment == exp]
    out = []
    for a in sorted(df_impr.alpha.unique()):
        sub = df_impr[df_impr.alpha == a]
        # (I) Pooled seed-level
        pooled = sub["impr_pct"].values
        t_p, p_p = sstats.ttest_1samp(pooled, 0.0)
        # (II) Clustered: per-cluster means
        clu_means = sub.groupby("cluster")["impr_pct"].mean().values
        if len(clu_means) > 1:
            t_c, p_c = sstats.ttest_1samp(clu_means, 0.0)
        else:
            t_c, p_c = float("nan"), float("nan")
        out.append({
            "alpha": a,
            "mean_pct": float(pooled.mean()),
            "pooled_n": int(len(pooled)),
            "pooled_sem": float(pooled.std(ddof=1)/np.sqrt(len(pooled))),
            "pooled_t": float(t_p), "pooled_p": float(p_p),
            "clust_n": int(len(clu_means)),
            "clust_mean_pct": float(clu_means.mean()),
            "clust_sem": float(clu_means.std(ddof=1)/np.sqrt(len(clu_means))) if len(clu_means) > 1 else float("nan"),
            "clust_t": float(t_c), "clust_p": float(p_c),
        })
    return pd.DataFrame(out)


for ref in ("A", "D"):
    df_impr = paired_improvements(df_all, ref=ref, target="C")
    for scope_name in ("HSI", "S2", "GRAND"):
        exp = None if scope_name == "GRAND" else scope_name
        res = tests(df_impr, exp=exp)
        res["experiment"] = scope_name
        res["comparison"] = f"C vs {ref}"
        # Save
        path = ROOT / f"clustered_tests_C_vs_{ref}_{scope_name}.csv"
        res.to_csv(path, index=False)

    # Combined printout for this ref
    all_scopes = []
    for scope_name in ("HSI", "S2", "GRAND"):
        exp = None if scope_name == "GRAND" else scope_name
        res = tests(df_impr, exp=exp).assign(experiment=scope_name, comparison=f"C vs {ref}")
        all_scopes.append(res)
    big = pd.concat(all_scopes, ignore_index=True)
    big.to_csv(ROOT / f"clustered_tests_C_vs_{ref}.csv", index=False)

    print(f"\n{'='*84}")
    print(f"C vs {ref}: pooled (seed-level) vs clustered (per-dataset/tile) paired t-tests")
    print(f"{'='*84}")
    print(f"{'scope':>7s}  {'alpha':>5s}  {'mean%':>7s}  "
          f"{'pooled-n':>8s} {'pooled-p':>12s}  "
          f"{'clust-n':>7s} {'clust-p':>12s}  {'ratio':>7s}")
    for _, r in big.iterrows():
        print(f"{r['experiment']:>7s}  {r['alpha']:>5.2f}  "
              f"{r['clust_mean_pct']:>+6.2f}%  "
              f"n={r['pooled_n']:>3d}  p={r['pooled_p']:>10.2e}  "
              f"n={r['clust_n']:>3d}  p={r['clust_p']:>10.2e}  "
              f"×{r['pooled_p']/r['clust_p'] if r['clust_p']>0 else float('inf'):>6.1e}")

print("\nAll clustered test CSVs written to", ROOT)
