# Usage Guide

How to call `geocp_rs` on your own data.

## Minimal pattern

SACP+GeoCP is **post-hoc**: you train any classifier, then wrap it.

```python
import numpy as np
from geocp_rs import run_sacp_geocp

# 1. Train (or load) any classifier that outputs per-class softmax.
#    Here the classifier is external — not part of geocp_rs.
probs_cal  = get_softmax(model, X_cal)    # (n_cal, K)
probs_test = get_softmax(model, X_test)   # (n_test, K)
y_cal = cal_labels                         # (n_cal,)

# 2. Tell SACP+GeoCP where each pixel lives on the image grid.
H, W = 145, 145
cal_flat_idx  = ...   # (n_cal,)  int, row*W + col
test_flat_idx = ...   # (n_test,) int
coords_cal  = np.stack([cal_flat_idx  // W, cal_flat_idx  % W], axis=1).astype(float)
coords_test = np.stack([test_flat_idx // W, test_flat_idx % W], axis=1).astype(float)

# 3. Run the procedure.
result = run_sacp_geocp(
    probs_cal=probs_cal, probs_test=probs_test, y_cal=y_cal,
    coords_cal_rowcol=coords_cal, coords_test_rowcol=coords_test,
    grid_shape=(H, W),
    cal_flat_idx=cal_flat_idx, test_flat_idx=test_flat_idx,
    alpha=0.1,          # target miscoverage
    lmd=0.5,            # SACP smoothing strength (fixed in paper)
    bandwidth=None,     # None → median cal-cal distance; or pick via CV
)

pred_sets = result["pred_sets"]   # list[list[int]] per test pixel
local_q   = result["local_q"]     # per-pixel GeoCP thresholds
```

## Returning the full set of CP variants

For research / ablations, use `evaluate_all_cp_methods` instead. It runs Standard CP, SACP at three λ values, and SACP+GeoCP with CV bandwidth selection in one call:

```python
from geocp_rs.evaluate import evaluate_all_cp_methods

results = evaluate_all_cp_methods(
    probs_cal=probs_cal, probs_test=probs_test,
    y_cal=y_cal, y_test=y_test,
    cal_flat_idx=cal_flat_idx, test_flat_idx=test_flat_idx,
    h=H, w=W,
    alpha=0.1,
)

# results contains keys:
#   'standard_cp', 'sacp_0.3', 'sacp_0.5', 'sacp_0.7', 'sacp_geocp',
#   'bandwidth_selected'
# each method dict has: cov, size, is, pred_sets (and extras)
```

## Using geocp_rs without PyTorch

The entire `geocp_rs.pipeline` module depends only on numpy + scipy + scikit-learn. You can run SACP+GeoCP on softmax outputs produced by:

- TensorFlow/JAX/PyTorch models
- scikit-learn classifiers (`predict_proba`)
- Bayesian ensembles (average posterior softmax)
- SVMs with Platt scaling
- Any external source that writes probabilities to disk

The only things that need PyTorch are:

- `geocp_rs.models.CNN3D` (the 3D-CNN baseline)
- `geocp_rs.train.train_3dcnn` (the 3D-CNN training loop)
- `scripts/run_all_experiments.py` (because it trains the 3D-CNN)

## Wrapping a 3D-CNN using the built-in training helper

```python
from geocp_rs.datasets import load_dataset, download_dataset
from geocp_rs.train import train_3dcnn
from geocp_rs.evaluate import evaluate_all_cp_methods

download_dataset("ip", "./data")
hsi, gt, n_classes, n_bands = load_dataset("ip", "./data")

train_out = train_3dcnn(hsi, gt, n_classes, n_bands, seed=0, epochs=100)
cp_out = evaluate_all_cp_methods(
    probs_cal=train_out["probs_cal"],
    probs_test=train_out["probs_test"],
    y_cal=train_out["y_cal"],
    y_test=train_out["y_test"],
    cal_flat_idx=train_out["cal_flat_idx"],
    test_flat_idx=train_out["test_flat_idx"],
    h=train_out["h"], w=train_out["w"],
    alpha=0.1,
)

print(f"IS  StdCP    : {cp_out['standard_cp']['is']:.3f}")
print(f"IS  SACP 0.5 : {cp_out['sacp_0.5']['is']:.3f}")
print(f"IS  SACP+GeoCP: {cp_out['sacp_geocp']['is']:.3f}")
```

## Which classifiers does SACP+GeoCP support?

**Any** model with a per-pixel softmax output. This includes:

| Category | Example architectures | Compatible? |
|---|---|---|
| Pixel-wise MLP/CNN on HSI patches | 2D-CNN, 3D-CNN (ours), HybridNet, SSTN | ✓ |
| Semantic segmentation on regular rasters | U-Net, DeepLab, SegFormer, Mask2Former | ✓ (single-scene) |
| Bayesian CNN / MC-Dropout | Haut et al. 2018, Blundell et al. 2015 | ✓ |
| SVM with Platt / multi-class | `sklearn.svm.SVC(probability=True)` | ✓ |
| Instance / panoptic segmentation | Mask R-CNN, Mask2Former (panoptic mode) | ✗ (instance IDs, not class probs) |
| Regression-only segmentation (distance maps) | — | ✗ (no softmax) |

## Important: exchangeability assumption

SACP+GeoCP's coverage guarantee holds when **calibration and test pixels are exchangeable**. In our HSI protocol they are: labeled pixels within a single scene are stratified-split into train/calib/test, and the conformal procedure is transductive over that scene.

**It does not directly apply to cross-image semantic segmentation** (e.g., Cityscapes train/val/test are different cities). There the geographic distance between a train pixel and a test pixel is undefined. Adapting SACP+GeoCP to that setting requires replacing the geographic kernel with an image-level or feature-space similarity and re-deriving the weighted-CP guarantee. See `paper/main.tex` §4 Applicability.

## Bandwidth selection

The only free hyperparameter beyond SACP's λ is the GeoCP Gaussian bandwidth $h$ (in pixels). We recommend one of:

1. **Pass `bandwidth=None`** to `run_sacp_geocp`. This defaults to the median pairwise cal-cal distance — a reasonable, data-driven starting point.
2. **Pass an explicit value** (e.g., `bandwidth=10.0`) if you know the characteristic spatial scale of your scene.
3. **Use `evaluate_all_cp_methods`**, which runs a 5-fold CV on the calibration set over the grid `{3, 5, 7, 10, 15, 20, 30, 50, 100}` and automatically picks the bandwidth minimizing the mean inner-validation IS. This is what the paper does.

Empirically, the CV picks:

| Dataset | Median bandwidth | Reason |
|---|---|---|
| Indian Pines | 10 | Small image, fine class patches |
| Salinas | 5 | Thin parallel crop strips |
| Pavia University | 100 | Large urban scene |
| KSC | 50 | Sparse wetlands — needs large bw to find neighbors |
| Botswana | 100 | Very long, sparse strip |

## Interpreting `local_q`

The per-pixel threshold $\hat q_j$ is returned for every test pixel. Visualizing it as an image shows *where* the conformal procedure becomes permissive (high $\hat q$ → wider sets) versus strict (low $\hat q$ → narrower sets). Characteristically:

- **Class interiors** → small $\hat q$, tight sets (the model is confident).
- **Class boundaries** → larger $\hat q$, wider sets (the model is uncertain, SACP+GeoCP compensates).

Figure 4 in the paper (`figures/fig_qualitative.png`) shows this behavior on all 5 datasets.

## Advanced: caching for multi-model comparison

If you want to compare two or more base classifiers, compute their softmax outputs once and cache them. Pass each cached softmax into `run_sacp_geocp` separately — nothing in `geocp_rs` depends on which model produced the probabilities.
