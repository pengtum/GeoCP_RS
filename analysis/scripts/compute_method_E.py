"""
Post-hoc computation of Method (E) = GeoCP-only (no SACP smoothing).

E = raw APS scores + CV-selected-bw Gaussian-weighted quantile (per-pixel threshold).

Reads existing pkls (no model retraining):
    HSI: /Users/.../hsi_alpha_sweep/checkpoints/*.pkl    (50 runs)
    S2:  /Users/.../s2_alpha_sweep/checkpoints/*.pkl     (29 non-degenerate runs)

Writes:
    analysis/method_E_per_run.csv  -- per (run, alpha) IS / cov / size / sscv / best_bw
    analysis/method_E_paired_tests.csv -- E vs A, E vs C, and D, at each alpha
"""
import pickle, os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.model_selection import KFold
from scipy import stats as sstats

HSI_CKPT = Path("/Users/peng/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/"
                "My Drive/hsi_alpha_sweep/checkpoints")
S2_CKPT  = Path("/Users/peng/Library/CloudStorage/GoogleDrive-pluopku@gmail.com/"
                "My Drive/s2_alpha_sweep/checkpoints")
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis"

BW_GRID = [3, 5, 7, 10, 15, 20, 30, 50, 100]
ALPHA_GRID = (0.05, 0.10, 0.15)
ALPHA_CV = 0.10
CV_FOLDS = 5
MAX_CAL_S2 = 20000


def conformal_quantile(scores, alpha):
    n = len(scores)
    return float(np.quantile(scores, np.ceil((n + 1) * (1 - alpha)) / n))


def vwq(sorted_scores, d_mat, order, bw, alpha):
    log_w = -0.5 * (d_mat / bw) ** 2
    log_w -= log_w.max(axis=1, keepdims=True)
    w = np.exp(log_w)
    w_sorted = w[:, order]
    ws = w_sorted / w_sorted.sum(axis=1, keepdims=True)
    cum = np.cumsum(ws, axis=1)
    k_star = np.argmax(cum >= (1 - alpha), axis=1)
    return sorted_scores[k_star]


def eval_set(score_mat, q_vec, y_true, alpha):
    in_set = score_mat < q_vec[:, None]
    sizes = in_set.sum(axis=1)
    covered = in_set[np.arange(len(y_true)), y_true.astype(int)]
    IS = float((sizes + (2.0 / alpha) * (~covered).astype(np.float64)).mean())
    size_mean = float(sizes.mean())
    cov_mean = float(covered.mean())
    # SSCV
    buckets = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 10**9)]
    worst = 0.0
    for lo, hi in buckets:
        m = (sizes >= lo) & (sizes <= hi)
        if m.sum() == 0:
            continue
        c = float(covered[m].mean())
        worst = max(worst, abs(c - (1 - alpha)))
    return IS, cov_mean, size_mean, worst * 100


def stratified_sub(labels, n, seed):
    if n is None or n >= len(labels):
        return np.arange(len(labels))
    uniq, cnts = np.unique(labels, return_counts=True)
    per_cls_n = np.maximum(1, (n * cnts / cnts.sum()).astype(int))
    rng = np.random.RandomState(seed)
    out = []
    for c, nc in zip(uniq, per_cls_n):
        pool = np.where(labels == c)[0]
        out.append(pool if len(pool) <= nc else rng.choice(pool, size=nc, replace=False))
    return np.concatenate(out)


def run_method_E_one(d, experiment):
    """One pkl -> dict of per-alpha results for method E."""
    H = int(d.get("H", d.get("h")))
    W = int(d.get("W", d.get("w")))
    cal_true = np.asarray(d["cal_true_aps"])
    cal_all  = np.asarray(d["cal_all_aps"])
    test_all = np.asarray(d["test_all_aps"])
    cal_idx  = np.asarray(d.get("cal_flat_idx", d.get("ca_gi")))
    te_idx   = np.asarray(d.get("test_flat_idx", d.get("te_gi")))
    y_ca     = np.asarray(d["y_ca"])
    y_te     = np.asarray(d["y_te"])

    # For S2, CV uses a subsample to match main pipeline
    if experiment == "S2":
        seed = 0  # same as main pipeline
        sub = stratified_sub(y_ca, MAX_CAL_S2, seed * 100 + 43)
    else:
        sub = np.arange(len(cal_idx))

    cal_true_sub = cal_true[sub]
    cal_all_sub  = cal_all[sub]
    cal_idx_sub  = cal_idx[sub]
    y_ca_sub     = y_ca[sub]
    n_sub = len(sub)

    coords_sub = np.stack([cal_idx_sub // W, cal_idx_sub % W], 1).astype(float)
    coords_te  = np.stack([te_idx // W, te_idx % W], 1).astype(float)

    # 5-fold CV on cal_sub at alpha=0.10 to pick bw (proper set-based IS)
    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=0)
    cv_is = {bw: [] for bw in BW_GRID}
    for f_tr, f_val in kf.split(np.arange(n_sub)):
        tr_scores   = cal_true_sub[f_tr]
        val_scores  = cal_all_sub[f_val]       # (n_val, K) -- for proper set construction
        val_y       = y_ca_sub[f_val]
        order_tr = np.argsort(tr_scores)
        sorted_tr = tr_scores[order_tr]
        BATCH = 500
        for bw in BW_GRID:
            q_val = np.empty(len(f_val))
            for b0 in range(0, len(f_val), BATCH):
                b1 = min(b0 + BATCH, len(f_val))
                dcv = cdist(coords_sub[f_val[b0:b1]], coords_sub[f_tr])
                q_val[b0:b1] = vwq(sorted_tr, dcv, order_tr, bw, ALPHA_CV)
            # proper interval score on the val fold
            IS_val, _, _, _ = eval_set(val_scores, q_val, val_y, ALPHA_CV)
            cv_is[bw].append(IS_val)
    cv_mean = {bw: float(np.mean(v)) for bw, v in cv_is.items()}
    best_bw = int(min(cv_mean, key=cv_mean.get))

    # Final per-test-pixel weighted quantile, using all of cal_sub as calibration
    order = np.argsort(cal_true_sub)
    sorted_cal = cal_true_sub[order]
    n_te = len(te_idx)
    BATCH_TEST = 1000
    out_per_alpha = {}
    for a in ALPHA_GRID:
        q_E = np.empty(n_te)
        for b0 in range(0, n_te, BATCH_TEST):
            b1 = min(b0 + BATCH_TEST, n_te)
            dtest = cdist(coords_te[b0:b1], coords_sub)
            q_E[b0:b1] = vwq(sorted_cal, dtest, order, best_bw, a)
        # IS, cov, size, sscv with raw APS test scores
        IS, cov, sz, sscv = eval_set(test_all, q_E, y_te, a)
        out_per_alpha[a] = {"is": IS, "cov": cov, "size": sz, "sscv_pct": sscv}
    return best_bw, out_per_alpha


def main():
    rows = []
    print("HSI runs (50)...")
    for p in sorted(HSI_CKPT.glob("*.pkl")):
        with open(p, "rb") as f: d = pickle.load(f)
        best_bw, per_a = run_method_E_one(d, "HSI")
        for a, r in per_a.items():
            rows.append({"experiment": "HSI", "dataset": d["dataset"], "seed": d["seed"],
                         "config_id": f'{d["dataset"]}_seed{d["seed"]}',
                         "alpha": a, "method": "E", "best_bw": best_bw, **r})

    print("S2 runs (~29)...")
    for p in sorted(S2_CKPT.glob("*.pkl")):
        with open(p, "rb") as f: d = pickle.load(f)
        if d.get("degenerate"): continue
        best_bw, per_a = run_method_E_one(d, "S2")
        for a, r in per_a.items():
            rows.append({"experiment": "S2", "tile": d["tile"], "size_px": d["size_px"],
                         "config_id": f'{d["tile"]}_s{d["size_px"]}',
                         "alpha": a, "method": "E", "best_bw": best_bw, **r})

    df_E = pd.DataFrame(rows)
    df_E.to_csv(OUT / "method_E_per_run.csv", index=False)
    print(f"Wrote method_E_per_run.csv ({len(df_E)} rows)")

    # Pooled summary
    summary = (df_E.groupby(["experiment", "alpha"])
                    .agg(is_mean=("is", "mean"), cov_mean=("cov", "mean"),
                         size_mean=("size", "mean"), sscv_mean=("sscv_pct", "mean"),
                         n=("alpha", "count")).reset_index())
    print("\n=== Method E pooled ===")
    print(summary.round(3).to_string(index=False))

    # Paired tests E vs (A, B, C, D) at each alpha
    df_main_hsi = pd.read_csv(OUT / "per_seed_alpha.csv")
    df_main_s2  = pd.read_csv(OUT / "s2_per_config_alpha.csv")

    results = []

    # HSI: config_id = dataset + '_seed' + seed
    df_main_hsi["config_id"] = df_main_hsi["dataset"] + "_seed" + df_main_hsi["seed"].astype(str)
    df_main_s2["config_id"]  = df_main_s2["tile"] + "_s" + df_main_s2["size_px"].astype(str)

    for exp, df_main in [("HSI", df_main_hsi), ("S2", df_main_s2)]:
        df_E_exp = df_E[df_E.experiment == exp]
        for a in ALPHA_GRID:
            sub_E = df_E_exp[df_E_exp.alpha == a].sort_values("config_id").reset_index(drop=True)
            for ref in ("D", "A", "B", "C"):
                sub_R = df_main[(df_main.alpha == a) & (df_main.method == ref)]\
                            .sort_values("config_id").reset_index(drop=True)
                if len(sub_R) != len(sub_E):
                    print(f"[WARN] {exp} α={a} vs {ref}: len mismatch {len(sub_E)} vs {len(sub_R)}")
                    continue
                assert (sub_R.config_id.values == sub_E.config_id.values).all()
                impr = (sub_R["is"].values - sub_E["is"].values) / sub_R["is"].values * 100
                t, p = sstats.ttest_1samp(impr, 0.0)
                results.append({"experiment": exp, "alpha": a, "comparison": f"E vs {ref}",
                                "mean_pct": impr.mean(),
                                "sem": impr.std(ddof=1) / np.sqrt(len(impr)),
                                "t": t, "p": p, "n": len(impr)})
    # Grand pooled (for E vs A, E vs C primarily)
    df_main_all = pd.concat([df_main_hsi.assign(experiment="HSI"),
                              df_main_s2.assign(experiment="S2")], ignore_index=True)
    for a in ALPHA_GRID:
        sub_E = df_E[df_E.alpha == a].sort_values(["experiment", "config_id"]).reset_index(drop=True)
        for ref in ("D", "A", "B", "C"):
            sub_R = df_main_all[(df_main_all.alpha == a) & (df_main_all.method == ref)]\
                        .sort_values(["experiment", "config_id"]).reset_index(drop=True)
            if len(sub_R) != len(sub_E): continue
            assert (sub_R.config_id.values == sub_E.config_id.values).all()
            impr = (sub_R["is"].values - sub_E["is"].values) / sub_R["is"].values * 100
            t, p = sstats.ttest_1samp(impr, 0.0)
            results.append({"experiment": "GRAND", "alpha": a, "comparison": f"E vs {ref}",
                            "mean_pct": impr.mean(),
                            "sem": impr.std(ddof=1) / np.sqrt(len(impr)),
                            "t": t, "p": p, "n": len(impr)})

    df_tests = pd.DataFrame(results)
    df_tests.to_csv(OUT / "method_E_paired_tests.csv", index=False)
    print("\n=== Paired tests (E vs others) ===")
    print(df_tests.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
