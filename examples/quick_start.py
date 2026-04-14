"""Minimal smoke test / quick-start example.

Runs the full SACP+GeoCP pipeline on a synthetic 50×50 image and prints
the resulting coverage, mean set size, and interval score. No dataset
download, no GPU, no PyTorch required.

    python examples/quick_start.py
"""
import numpy as np

from geocp_rs import (
    aps_scores, sacp_smooth, geocp_local_threshold,
    coverage_and_size, interval_score, run_sacp_geocp,
)


def main():
    rng = np.random.RandomState(0)
    H, W, K = 50, 50, 4
    N = H * W

    # Synthetic softmax with mild spatial structure
    probs_all = rng.dirichlet(np.ones(K), size=N)
    y_all = np.argmax(probs_all + 0.1 * rng.randn(N, K), axis=1)

    # 50/50 calibration / test split
    idx = rng.permutation(N)
    cal_idx, test_idx = idx[:N // 2], idx[N // 2:]

    coords_cal = np.stack([cal_idx // W, cal_idx % W], axis=1).astype(float)
    coords_test = np.stack([test_idx // W, test_idx % W], axis=1).astype(float)

    result = run_sacp_geocp(
        probs_cal=probs_all[cal_idx],
        probs_test=probs_all[test_idx],
        y_cal=y_all[cal_idx],
        coords_cal_rowcol=coords_cal,
        coords_test_rowcol=coords_test,
        grid_shape=(H, W),
        cal_flat_idx=cal_idx,
        test_flat_idx=test_idx,
        alpha=0.1, lmd=0.5, bandwidth=10.0, rng=rng,
    )

    cov, sz = coverage_and_size(result["pred_sets"], y_all[test_idx])
    is_val = interval_score(result["pred_sets"], y_all[test_idx], alpha=0.1)
    print(f"Quick start OK")
    print(f"  Coverage : {cov:.3f}  (target 0.9)")
    print(f"  Mean size: {sz:.2f}")
    print(f"  IS       : {is_val:.3f}")
    print(f"  Bandwidth: {result['bandwidth']:.1f}")


if __name__ == "__main__":
    main()
