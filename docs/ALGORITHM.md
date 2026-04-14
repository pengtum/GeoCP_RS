# The SACP+GeoCP Algorithm

This document walks through the method step by step, explains where each piece of the coverage guarantee comes from, and points at the corresponding code in `code/sacp_geocp_core.py`.

## Inputs and notation

- **Classifier**: a trained black-box function $f : \mathcal{X} \to \Delta^{K-1}$ that outputs softmax probabilities over $K$ classes.
- **Calibration set**: $(X_i, Y_i, \mathbf{c}_i)$ for $i \in \mathcal{I}_{\text{cal}}$, where $\mathbf{c}_i \in \mathbb{Z}^2$ is the integer pixel coordinate (row, col) of sample $i$.
- **Test set**: $(X_j, \mathbf{c}_j)$ for $j \in \mathcal{I}_{\text{test}}$, where we do *not* observe $Y_j$.
- **Exchangeability**: calibration and test pixels are drawn from the same single hyperspectral scene, so they are jointly exchangeable.
- **Target miscoverage**: $\alpha \in (0, 1)$; we use $\alpha = 0.1$ throughout.

## Step 1 — Softmax probabilities

Run the classifier on both calibration and test pixels to obtain $\pi_i, \pi_j \in \Delta^{K-1}$. No retraining, no probe of internal layers, no gradient access required.

**Code**: this happens outside `sacp_geocp_core.py`, in whatever training driver you use (see `code/train_and_eval.py` or cell 10 of the Colab notebook).

## Step 2 — APS non-conformity scores

For each sample $i$ and each candidate class $y$, compute the Adaptive Prediction Set (APS) score of Romano, Sesia, Candès (2020):

$$
s_{\text{APS}}(x_i, y) = \sum_{k\,:\,\pi_{i,k} > \pi_{i,y}} \pi_{i,k} \;+\; U \cdot \pi_{i,y}, \qquad U \sim \text{Unif}(0,1).
$$

Smaller scores indicate the true class is confidently ranked near the top. The randomized term $U$ guarantees the score is continuous, which ensures the split-CP guarantee is *exact* rather than merely $\ge 1-\alpha$.

**Code**: `aps_scores()` in `sacp_geocp_core.py`, lines 30–65.

## Step 3 — SACP spatial smoothing (Liu et al., 2024)

Place all APS scores on the $h \times w$ image grid and smooth each pixel's $K$-dimensional score vector with an 8-neighbor local average:

$$
\tilde{s}_i(y) = (1 - \lambda)\, s_i(y) + \lambda \cdot \frac{1}{|N(i)|} \sum_{j \in N(i)} s_j(y),
$$

where $N(i)$ is the set of up-to-8 valid neighbors (pixels that belong to either the calibration or test set). We use $\lambda = 0.5$ throughout; the paper ablates $\lambda \in \{0.3, 0.5, 0.7\}$.

**Why coverage survives.** The smoothing operator is a deterministic linear function applied identically to calibration and test scores (same kernel, same neighbors). Formally, the tuple of smoothed scores is a pushforward of the tuple of raw scores under a symmetric operator, so exchangeability of the smoothed scores is preserved (Liu et al., 2024, Proposition 2).

**Code**: `sacp_smooth()` in `sacp_geocp_core.py`, lines 90–135. Implementation is fully vectorized with NumPy shifts and masks, O(hwK) time.

## Step 4 — GeoCP local threshold (Lou et al., 2024)

Instead of computing one global quantile $\hat q$ over all calibration scores, compute a **per-test-pixel** quantile by geographically weighting the calibration set:

$$
\hat q_j = \text{WeightedQuantile}\!\left(\{\tilde s_i(Y_i)\}_{i \in \mathcal{I}_{\text{cal}}},\ \{w_{ji}\}_{i \in \mathcal{I}_{\text{cal}}};\ 1 - \alpha\right),
$$

$$
w_{ji} = \exp\!\left(-\tfrac{1}{2}\,\|\mathbf{c}_j - \mathbf{c}_i\|^2 / h^2\right).
$$

The bandwidth $h$ is measured in pixels. We select it per dataset via 5-fold CV on the calibration set (see EXPERIMENT_PROTOCOL.md) from the grid $\{3, 5, 7, 10, 15, 20, 30, 50, 100\}$.

**Why coverage survives.** Because $w_{ji}$ depends only on pixel coordinates, not on labels, the weighted split CP framework of Tibshirani, Barber, Candès, Ramdas (2019) applies directly:

$$
\Pr\bigl[Y_j \in \mathcal{C}(X_j)\bigr] \;\ge\; 1 - \alpha, \quad j \in \mathcal{I}_{\text{test}}.
$$

The guarantee is marginal; per-pixel conditional coverage is not guaranteed (and is not claimed).

**Numerical stability note.** For a far-outside-range test pixel, $\exp(-d^2 / h^2)$ can underflow for every calibration point simultaneously, yielding all-zero weights. `geocp_local_threshold()` uses the log-sum-exp trick (subtracting the max log-weight before exponentiating) to prevent this.

**Code**: `geocp_local_threshold()` in `sacp_geocp_core.py`, lines 151–171.

## Step 5 — Prediction sets

For each test pixel $j$, emit

$$
\mathcal{C}(X_j) = \{y \in \{1, \dots, K\} : \tilde s_j(y) < \hat q_j\}.
$$

**Code**: tail of `run_sacp_geocp()` in `sacp_geocp_core.py`, lines 252–266.

---

## End-to-end pseudocode

```
INPUT: probs_cal, probs_test, y_cal,
       coords_cal, coords_test, grid_shape (h, w),
       cal_flat_idx, test_flat_idx, alpha, lmd, bandwidth

1. cal_all  = APS(probs_cal)                              # (n_cal, K)
2. test_all = APS(probs_test)                             # (n_test, K)
3. Place cal_all and test_all onto the h*w grid, leaving
   zeros at unlabeled positions; call this S.
4. S_tilde = SACP_smooth(S, h, w, valid_idx = cal∪test, lmd)
5. cal_scores_true = [S_tilde[cal_flat_idx[i], y_cal[i]] for i]
6. test_scores_all = [S_tilde[test_flat_idx[j], :]        for j]
7. if bandwidth is None:
       bandwidth = median pairwise cal-cal Euclidean distance
8. for j in test set:
       q_j = WeightedQuantile(cal_scores_true,
                              gauss_kernel(coords_test[j], coords_cal, h),
                              1 - alpha)
       C_j = {y : test_scores_all[j, y] < q_j}
9. OUTPUT: {C_j}_j in test set
```

This is Algorithm 1 of the paper, and it is implemented verbatim in `run_sacp_geocp()`.

## Complexity

Let $h \times w = N$ pixels, $n_{\text{cal}}$ calibration pixels, $n_{\text{test}}$ test pixels, $K$ classes.

- APS: $O((n_{\text{cal}} + n_{\text{test}}) K \log K)$
- SACP smoothing: $O(N K)$ (convolution with a $3 \times 3$ kernel)
- Pairwise distance matrix (coords_test $\times$ coords_cal): $O(n_{\text{cal}} n_{\text{test}})$
- Weighted quantile per test pixel: $O(n_{\text{cal}} \log n_{\text{cal}})$
- Total: **$O(N K + n_{\text{cal}} n_{\text{test}} + n_{\text{test}} n_{\text{cal}} \log n_{\text{cal}})$**

On the largest scene (Pavia University, $610 \times 340$ = 207k pixels, $n_{\text{test}} \approx 20{,}000$, $K = 9$): end-to-end CP evaluation takes ~5 seconds on a modern laptop on top of the 3D-CNN training. Memory footprint is dominated by the $(n_{\text{test}}, n_{\text{cal}})$ distance matrix (~4 GB in float64, ~2 GB in float32). For larger scenes, one can use blocked computation over test pixels.

## Relation to the individual components

| Method | What gets smoothed? | What's the threshold? | Guarantee |
|---|---|---|---|
| Standard CP | nothing | single global $\hat q$ | marginal $1 - \alpha$ |
| SACP | APS scores, 8-neighbor avg | single global $\hat q$ | marginal $1 - \alpha$ |
| GeoCP | nothing | per-pixel $\hat q_j$ via geographic kernel | marginal $1 - \alpha$ (weighted CP) |
| **SACP+GeoCP** | APS scores, 8-neighbor avg | per-pixel $\hat q_j$ via geographic kernel | marginal $1 - \alpha$ (weighted CP, on smoothed scores) |

The two extensions are compositional and orthogonal.

## Why the composition helps empirically

SACP reduces the *variance* of calibration-to-test score transfer by averaging nearby pixels. GeoCP reduces the *bias* from a global quantile in spatially heterogeneous scenes by allowing the threshold to move with spatial structure. Removing either leaves one of these error sources uncorrected; doing both together is what produces the statistically significant improvement documented in RESULTS_ANALYSIS.md.
