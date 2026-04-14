"""Aggregate per-seed checkpoints into summary.json / per_seed.csv / stats.json
/ results_table.tex.

Usage:
    python -m scripts.aggregate_results --in ./results --out ./results
    geocp-rs-aggregate --in ./results --out ./results
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import pickle

import numpy as np
from scipy import stats as sstats


DATASETS = ["ip", "pu", "sa", "ksc", "botswana"]
NICE = {
    "ip": "Indian Pines",
    "pu": "Pavia University",
    "sa": "Salinas",
    "ksc": "KSC",
    "botswana": "Botswana",
}


def _ms(xs):
    arr = np.array(xs, dtype=float)
    return {"mean": float(arr.mean()), "std": float(arr.std()), "n": int(len(arr))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_dir", default="./results",
                    help="Directory containing checkpoints/{ds}_seed{n}.pkl")
    ap.add_argument("--out", default=None,
                    help="Output directory (default: same as --in)")
    ap.add_argument("--n-seeds", type=int, default=10)
    args = ap.parse_args()

    out_dir = args.out or args.in_dir
    ckpt_dir = os.path.join(args.in_dir, "checkpoints")
    os.makedirs(out_dir, exist_ok=True)

    # ---- Load all checkpoints ----
    all_results: dict[str, list] = {ds: [] for ds in DATASETS}
    for ds in DATASETS:
        for seed in range(args.n_seeds):
            p = os.path.join(ckpt_dir, f"{ds}_seed{seed}.pkl")
            if os.path.exists(p):
                with open(p, "rb") as f:
                    all_results[ds].append(pickle.load(f))

    # ---- summary.json ----
    summary = {}
    for ds in DATASETS:
        rs = all_results[ds]
        if not rs:
            continue
        summary[ds] = {
            "nice_name": NICE[ds],
            "n_seeds": len(rs),
            "accuracy": _ms([r["accuracy"] for r in rs]),
            "standard_cp": {k: _ms([r["standard_cp"][k] for r in rs])
                             for k in ("cov", "size", "is")},
            "sacp_0.3": {k: _ms([r["sacp"][0.3][k] for r in rs])
                          for k in ("cov", "size", "is")},
            "sacp_0.5": {k: _ms([r["sacp"][0.5][k] for r in rs])
                          for k in ("cov", "size", "is")},
            "sacp_0.7": {k: _ms([r["sacp"][0.7][k] for r in rs])
                          for k in ("cov", "size", "is")},
            "sacp_geocp": {k: _ms([r["sacp_geocp"][k] for r in rs])
                            for k in ("cov", "size", "is")},
            "bw_selected": [int(r["sacp_geocp"]["bw"]) for r in rs],
        }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[saved] {out_dir}/summary.json")

    # ---- per_seed.csv ----
    csv_path = os.path.join(out_dir, "per_seed.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "seed", "accuracy",
                    "std_cp_cov", "std_cp_size", "std_cp_is",
                    "sacp03_cov", "sacp03_size", "sacp03_is",
                    "sacp05_cov", "sacp05_size", "sacp05_is",
                    "sacp07_cov", "sacp07_size", "sacp07_is",
                    "geocp_cov", "geocp_size", "geocp_is", "geocp_bw"])
        for ds in DATASETS:
            for r in all_results[ds]:
                w.writerow([ds, r["seed"], r["accuracy"],
                            r["standard_cp"]["cov"], r["standard_cp"]["size"], r["standard_cp"]["is"],
                            r["sacp"][0.3]["cov"],   r["sacp"][0.3]["size"],   r["sacp"][0.3]["is"],
                            r["sacp"][0.5]["cov"],   r["sacp"][0.5]["size"],   r["sacp"][0.5]["is"],
                            r["sacp"][0.7]["cov"],   r["sacp"][0.7]["size"],   r["sacp"][0.7]["is"],
                            r["sacp_geocp"]["cov"],  r["sacp_geocp"]["size"],  r["sacp_geocp"]["is"],
                            r["sacp_geocp"]["bw"]])
    print(f"[saved] {csv_path}")

    # ---- stats.json ----
    stats_out = {}
    all_imps, pool_sacp, pool_gc = [], [], []
    for ds in DATASETS:
        rs = all_results[ds]
        if not rs:
            continue
        sacp_is = np.array([r["sacp"][0.5]["is"] for r in rs])
        gc_is = np.array([r["sacp_geocp"]["is"] for r in rs])
        imps = (sacp_is - gc_is) / sacp_is * 100
        t, p = (sstats.ttest_rel(sacp_is, gc_is) if len(rs) > 1 else (0, 1))
        stats_out[ds] = {
            "n_seeds": len(rs),
            "mean_imp_pct": float(imps.mean()),
            "t_stat": float(t),
            "p_value": float(p),
        }
        all_imps.extend(imps.tolist())
        pool_sacp.extend(sacp_is.tolist())
        pool_gc.extend(gc_is.tolist())

    pool_sacp = np.array(pool_sacp)
    pool_gc = np.array(pool_gc)
    if len(pool_sacp) > 1:
        t_all, p_all = sstats.ttest_rel(pool_sacp, pool_gc)
        try:
            w_stat, p_wil = sstats.wilcoxon(pool_sacp, pool_gc)
        except Exception:
            w_stat, p_wil = 0, 1
    else:
        t_all = p_all = w_stat = p_wil = float("nan")

    stats_out["pooled"] = {
        "n": int(len(pool_sacp)),
        "mean_imp_pct": float(np.mean(all_imps)) if all_imps else 0.0,
        "t_stat": float(t_all),
        "p_value_t": float(p_all),
        "wilcoxon_W": float(w_stat),
        "p_value_wilcoxon": float(p_wil),
        "n_positive": int(np.sum(np.array(all_imps) > 0)),
    }
    with open(os.path.join(out_dir, "stats.json"), "w") as f:
        json.dump(stats_out, f, indent=2)
    print(f"[saved] {out_dir}/stats.json")

    # ---- results_table.tex ----
    PM = r"$\pm$"
    def fmt(d, digits=3):
        return f"{d['mean']:.{digits}f}{PM}{d['std']:.{digits}f}"

    lines = [
        r"\begin{tabular}{l|c|ccc|ccc|ccc}",
        r"\toprule",
        r" & & \multicolumn{3}{c|}{Standard CP} & \multicolumn{3}{c|}{SACP ($\lambda$=0.5)} & \multicolumn{3}{c}{SACP+GeoCP} \\",
        r"Dataset & Acc & Cov & Size & IS & Cov & Size & IS & Cov & Size & IS \\",
        r"\midrule",
    ]
    for ds in DATASETS:
        if ds not in summary:
            continue
        S = summary[ds]
        row = (f"{S['nice_name']} & {S['accuracy']['mean']:.3f} & "
               f"{fmt(S['standard_cp']['cov'])} & {fmt(S['standard_cp']['size'], 2)} & {fmt(S['standard_cp']['is'], 2)} & "
               f"{fmt(S['sacp_0.5']['cov'])} & {fmt(S['sacp_0.5']['size'], 2)} & {fmt(S['sacp_0.5']['is'], 2)} & "
               f"{fmt(S['sacp_geocp']['cov'])} & {fmt(S['sacp_geocp']['size'], 2)} & {fmt(S['sacp_geocp']['is'], 2)} \\\\")
        lines.append(row)
    lines += [r"\bottomrule", r"\end{tabular}"]
    tex_path = os.path.join(out_dir, "results_table.tex")
    with open(tex_path, "w") as f:
        f.write("\n".join(lines))
    print(f"[saved] {tex_path}")


if __name__ == "__main__":
    main()
