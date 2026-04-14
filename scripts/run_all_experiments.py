"""End-to-end experiment driver: 5 datasets × 10 seeds, with checkpointing.

Usage (after installing the package):
    python -m scripts.run_all_experiments --data-dir ./data --out ./results
    # or through the console script
    geocp-rs-run-all --data-dir ./data --out ./results

Every (dataset, seed) combination is pickled to ``{out}/checkpoints/{ds}_seed{n}.pkl``
immediately on completion. Re-running skips any already-cached seeds, so you
can interrupt and resume safely.
"""
from __future__ import annotations
import argparse
import os
import pickle
import time

import numpy as np

from geocp_rs.datasets import DATASETS, download_dataset, load_dataset
from geocp_rs.train import train_3dcnn
from geocp_rs.evaluate import evaluate_all_cp_methods


def _atomic_pickle_dump(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser(description="Run 5 HSI datasets × 10 seeds.")
    ap.add_argument("--data-dir", default="./data",
                    help="Directory under which datasets/{key}/*.mat live.")
    ap.add_argument("--out", default="./results",
                    help="Output directory for summary.json / checkpoints.")
    ap.add_argument("--datasets", nargs="*",
                    default=["ip", "pu", "sa", "ksc", "botswana"])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--download", action="store_true",
                    help="Auto-download datasets from EHU mirror before running.")
    args = ap.parse_args()

    ckpt_dir = os.path.join(args.out, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    log_path = os.path.join(args.out, "run_log.txt")

    total = len(args.datasets) * args.seeds
    done = 0
    t_start = time.time()

    for ds in args.datasets:
        spec = DATASETS[ds]
        if args.download:
            download_dataset(ds, args.data_dir)
        hsi, gt, n_classes, n_bands = load_dataset(ds, args.data_dir)
        h, w, _ = hsi.shape

        print(f"\n{'='*70}\n{spec.nice_name} ({ds}) — {h}×{w}×{n_bands}, {n_classes} classes\n{'='*70}")

        for seed in range(args.seeds):
            ckpt_path = os.path.join(ckpt_dir, f"{ds}_seed{seed}.pkl")
            if os.path.exists(ckpt_path):
                with open(ckpt_path, "rb") as f:
                    r = pickle.load(f)
                done += 1
                print(f"  seed={seed} [cached]  acc={r['accuracy']:.3f}  "
                      f"SACP+GeoCP IS={r['sacp_geocp']['is']:.3f}  [{done}/{total}]")
                continue

            t0 = time.time()
            train_out = train_3dcnn(
                hsi, gt, n_classes, n_bands, seed=seed, epochs=args.epochs)

            cp_out = evaluate_all_cp_methods(
                probs_cal=train_out["probs_cal"],
                probs_test=train_out["probs_test"],
                y_cal=train_out["y_cal"],
                y_test=train_out["y_test"],
                cal_flat_idx=train_out["cal_flat_idx"],
                test_flat_idx=train_out["test_flat_idx"],
                h=train_out["h"], w=train_out["w"],
                alpha=args.alpha,
            )

            result = {
                "dataset": ds, "nice_name": spec.nice_name, "seed": int(seed),
                "h": train_out["h"], "w": train_out["w"],
                "n_classes": n_classes, "bands": n_bands, "alpha": args.alpha,
                "accuracy": train_out["accuracy"],
                "standard_cp": cp_out["standard_cp"],
                "sacp": {0.3: cp_out["sacp_0.3"],
                          0.5: cp_out["sacp_0.5"],
                          0.7: cp_out["sacp_0.7"]},
                "sacp_geocp": cp_out["sacp_geocp"],
                "ca_gi": train_out["cal_flat_idx"].tolist(),
                "te_gi": train_out["test_flat_idx"].tolist(),
                "y_ca": train_out["y_cal"].tolist(),
                "y_te": train_out["y_test"].tolist(),
            }
            _atomic_pickle_dump(ckpt_path, result)

            done += 1
            imp = ((cp_out["sacp_0.5"]["is"] - cp_out["sacp_geocp"]["is"])
                   / cp_out["sacp_0.5"]["is"] * 100)
            msg = (f"  seed={seed}  acc={train_out['accuracy']:.3f}  "
                   f"StdCP={cp_out['standard_cp']['is']:.3f}  "
                   f"SACP(0.5)={cp_out['sacp_0.5']['is']:.3f}  "
                   f"SACP+GeoCP={cp_out['sacp_geocp']['is']:.3f} ({imp:+.1f}%)  "
                   f"bw={cp_out['sacp_geocp']['bw']}  "
                   f"[{time.time()-t0:.0f}s]  [{done}/{total}]")
            print(msg)
            with open(log_path, "a") as f:
                f.write(f"{ds} {msg}\n")

    print(f"\nDONE: {done}/{total} runs in {(time.time()-t_start)/60:.1f} minutes")


if __name__ == "__main__":
    main()
