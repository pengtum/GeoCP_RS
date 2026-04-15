# The SACP+GeoCP Algorithm

This document is the long-form companion to Section 4 of `paper/main.tex`. It walks through every step of the algorithm in detail, points at the corresponding code in `geocp_rs/`, and states the formal coverage guarantee with proof.

## Notation

| Symbol | Meaning |
|---|---|
| $\mathcal{X}$ | Pixel feature space (a 5×5 spatial patch of all bands, in our 3D-CNN setting) |
| $\mathcal{Y} = \{1, \dots, K\}$ | Set of class labels |
| $\mathcal{I}_{\text{train}}, \mathcal{I}_{\text{cal}}, \mathcal{I}_{\text{test}}$ | Disjoint subsets of labeled pixel indices |
| $X_i \in \mathcal{X}$ | Feature patch for pixel $i$ |
| $Y_i \in \mathcal{Y}$ | True label of pixel $i$ (observed for cal, hidden for test) |
| $\mathbf{c}_i \in \mathbb{Z}^2$ | Row/col coordinates of pixel $i$ on the image grid |
| $f : \mathcal{X} \to \Delta^{K-1}$ | Trained classifier producing per-class softmax probabilities |
| $\pi_i = f(X_i) \in \Delta^{K-1}$ | Softmax output at pixel $i$ |
| $\alpha \in (0, 1)$ | Target miscoverage (we use $\alpha = 0.1$) |
| $\lambda \in [0, 1]$ | SACP smoothing weight (we fix $\lambda = 0.5$) |
| $h > 0$ | GeoCP Gaussian kernel bandwidth in pixel units (CV-selected per dataset) |
| $N(i)$ | Set of valid 8-neighbors of pixel $i$ on the image grid |

## The motivation: a corn–wheat boundary

A 3D-CNN trained on a Midwest farmland HSI cube is highly confident deep inside corn or wheat patches but uncertain along the thin band where the two meet. From an uncertainty quantification standpoint, the two regimes need different prediction sets:

- **Inside a patch**: the classifier is essentially right; the user wants `{corn}` or `{wheat}` (size 1) and a wider set is wasted information.
- **At the boundary**: the spectral signature is mixed and the top-1 might be wrong; the user wants `{corn, wheat}` (size 2) so that the truth is genuinely covered.

A reliable spatial CP method must do **both**: tighten the prediction set in interiors and widen it at boundaries, while still hitting the user's nominal coverage budget. SACP+GeoCP's whole reason for existing is to satisfy this constraint with a finite-sample guarantee.

Standard split CP cannot. It uses a single global threshold $\hat q$ over all calibration scores and applies it identically to every test pixel. This threshold is, by construction, neither tight enough for the interior nor wide enough for the boundary. The two recent extensions in the literature each fix one side of the problem:

| Method | Fixes | Leaves untouched |
|---|---|---|
| **SACP** ([Liu et al., 2024](https://arxiv.org/abs/2409.01236)) | Score-side noise via 8-neighbor smoothing | Still uses one global threshold |
| **GeoCP** ([Lou et al., 2024](https://arxiv.org/abs/2412.08661)) | Threshold-side adaptivity via geographic kernel | Operates on raw, noisy scores |
| **SACP+GeoCP** (this work) | Both | — |

## Algorithm

```
INPUT:  trained classifier f, calibration set with labels and coordinates,
        test set with coordinates, image (H, W),
        miscoverage alpha, smoothing weight lambda, bandwidth h

Step 1 -- APS scores (per-pixel, per-class):
  for i in calibration ∪ test, for y in 1..K:
      s_i(y) = sum_{k : pi_{i,k} > pi_{i,y}} pi_{i,k}  +  U_{i,y} * pi_{i,y}
  // U_{i,y} ~ Uniform(0,1) for randomized tie-breaking

Step 2 -- SACP score smoothing:
  for i in calibration ∪ test, for y in 1..K:
      tilde_s_i(y) = (1 - lambda) * s_i(y)
                   + lambda * mean_{j in N(i)} s_j(y)
  // 8-neighbor mean over valid pixels only

Step 3 -- GeoCP local quantile:
  for j in test:
      w_{ji} = exp( -||c_j - c_i||^2 / (2 * h^2) )       for i in calibration
      q_j    = WeightedQuantile( {tilde_s_i(Y_i)},
                                  {w_{ji}}, 1 - alpha )

Step 4 -- Form prediction set:
  for j in test:
      C(X_j) = { y in 1..K : tilde_s_j(y) < q_j }

OUTPUT: { C(X_j) : j in test }
```

## Step 1 — APS nonconformity scores

For every (pixel, class) pair we compute the Adaptive Prediction Set score of [Romano, Sesia, Candès (2020)](https://arxiv.org/abs/2006.02544):

$$
s_i(y) = \sum_{k\,:\,\pi_{i,k} > \pi_{i,y}} \pi_{i,k} \;+\; U_{i,y} \cdot \pi_{i,y}, \qquad U_{i,y} \sim \mathrm{Unif}(0,1).
$$

Two facts that matter for the rest of the pipeline:

1. **Direction of the score.** Small $s_i(y)$ means "class $y$ is confidently top-ranked at pixel $i$"; large $s_i(y)$ means "class $y$ is buried under many more confident classes." This is the opposite direction of softmax probability, but it is the right direction for conformal procedures that admit classes whose score is *below* a threshold.

2. **Continuity via randomization.** Without the $U$ term, $s_i(\cdot)$ is a sum of probabilities and could collide on ties; with the $U$ term, the empirical distribution of $\{s_i(Y_i)\}$ is continuous, and the standard split-CP coverage bound becomes *exact*: $1 - \alpha \le \Pr[Y \in C(X)] \le 1 - \alpha + 1/(n+1)$ rather than just $\ge 1 - \alpha$.

**Code:** `geocp_rs.core.aps_scores`.

## Step 2 — SACP score smoothing

The classifier sees each pixel's $5\times5$ patch but evaluates one pixel at a time; consecutive pixels are not jointly regularized at inference. The raw score field $\{s_i(y)\}$ therefore carries pixel-level noise that is uncorrelated with the underlying class structure. A single noisy outlier in calibration could pull the standard $\hat q$ in the wrong direction even though the surrounding 8 pixels disagree with it.

SACP corrects this with the simplest spatial regularizer: a per-class 8-neighbor mean.

$$
\tilde s_i(y) = (1 - \lambda)\, s_i(y) \;+\; \lambda \cdot \frac{1}{|N(i)|}\sum_{j \in N(i)} s_j(y).
$$

Three properties to internalize:

- **Per class.** The score for "is corn at pixel $i$" is averaged with the neighbors' "is corn" scores, not their "is wheat" scores. So a corn pixel in the middle of a corn patch does not get its score corrupted by the neighbors' (large) wheat scores.
- **Symmetric across calibration and test.** The smoothing operator never asks "is this pixel calibration or test?" It only consults the spatial neighborhood graph $N(\cdot)$. This is what makes the marginal CP coverage guarantee survive the operation (Lemma 1 below).
- **Bidirectional uncertainty signal.** If a single pixel is noisy in the interior of a confident class patch, smoothing pulls its score toward the surrounding (small) average → noise removed. If a whole patch of pixels is uncertain (the corn–wheat boundary), smoothing averages them with each other and the elevated scores stay elevated → uncertainty preserved.

**Code:** `geocp_rs.sacp.sacp_smooth` (vectorized NumPy, $O(HWK)$).

## Step 3 — GeoCP local quantile

After SACP smoothing, the cleanest scores live in the calibration set, evaluated at the *true* labels: $\tilde s_i(Y_i)$ for $i \in \mathcal{I}_{\text{cal}}$. Standard split CP would compute a single $(1 - \alpha)$ quantile over these scalars and use it as the threshold for every test pixel. GeoCP refuses. Instead, for each test pixel $j$ it computes a **local** quantile by reweighting the calibration scores with a Gaussian kernel on geographic distance:

$$
w_{ji} = \exp\!\left(-\frac{1}{2}\frac{\|\mathbf{c}_j - \mathbf{c}_i\|^2}{h^2}\right), \qquad
\hat q_j = \mathrm{WeightedQuantile}\bigl(\{\tilde s_i(Y_i)\}_{i \in \mathcal{I}_{\text{cal}}},\ \{w_{ji}\},\ 1 - \alpha\bigr).
$$

The bandwidth $h$ controls the spatial scale at which calibration pixels "vote" on the test pixel's threshold. Small $h$ makes $\hat q_j$ depend almost only on the immediate geographic neighbors (high variance, very local); large $h$ recovers the global quantile in the limit $h \to \infty$ (low variance, no spatial adaptation).

### Why the local threshold automatically tracks class boundaries

The crucial fact is that **the calibration scores at the true labels naturally encode model uncertainty along the class boundary**. The GeoCP step never has to be told where the boundary is — the spatial distribution of nonconformity reveals it.

To see this concretely, imagine three test pixels in our corn / wheat motivating example:

**Test pixel deep in the corn patch.** Its geographic neighborhood is full of calibration pixels whose true label is also corn. On these calibration pixels, the classifier is confident (model says "corn 0.95"), so $\tilde s_i(Y_i) \approx 0$. The weighted quantile $\hat q_j$ inherits these small numbers and is itself small. The prediction set $\{y : \tilde s_j(y) < \text{small}\}$ contains only corn → **size 1**.

**Test pixel deep in the wheat patch.** Symmetric story: nearby calibration pixels are wheat, $\tilde s_i(Y_i) \approx 0$ on them, $\hat q_j$ is small, the set is `{wheat}` → **size 1**.

**Test pixel right at the corn–wheat boundary.** Its neighborhood mixes corn and wheat calibration pixels, and the classifier is uncertain on *both* — corn calibration pixels get true-label scores around 0.4 (model's top class wasn't always corn even when it was), and wheat calibration pixels get true-label scores around 0.4 too. The weighted quantile $\hat q_j$ reads this elevated distribution and produces a larger threshold. The prediction set widens to `{corn, wheat}` → **size 2**, exactly enough to cover the truth.

Notice three things:

- GeoCP **never inspects the test pixel's predicted class**. It does not partition the calibration set by class. The class-conditional structure is implicit in the spatial distribution of nonconformity, and the geographic kernel reads it out for free.
- The argument relies on the spatial autocorrelation that is intrinsic to HSI: nearby pixels almost always share class identity. Without this, GeoCP would not be a meaningful operation.
- The CV-selected bandwidth $h$ corresponds to the characteristic class-patch size of each dataset (Indian Pines: 10 pixels, Salinas: 5, Pavia U.: 100). The bandwidth selection itself learns the dataset's spatial scale.

### Numerical stability

For a test pixel very far from any calibration pixel (block-OOD), $\exp(-d^2 / (2h^2))$ underflows to zero on every term simultaneously, and the weighted quantile becomes ill-defined. We use the standard log-sum-exp trick: subtract the maximum log-weight before exponentiating, so that at least one weight is exactly 1.0 and the rest are in $[0, 1]$. This is implemented inside `geocp_rs.geocp.geocp_local_threshold`.

**Code:** `geocp_rs.geocp.geocp_local_threshold`.

## Step 4 — Form the prediction set

Given $\hat q_j$ for each test pixel:

$$
\mathcal{C}(X_j) = \{y \in \mathcal{Y}\;:\; \tilde s_j(y) < \hat q_j\}.
$$

This is the standard conformal sublevel-set rule applied to the *smoothed* scores at the *local* threshold. Smaller $\hat q_j$ admits fewer classes (interior pixels), larger $\hat q_j$ admits more (boundary pixels). The empirical coverage and per-pixel size visualizations in `figures/fig_qualitative.png` are exactly this rule applied at every test pixel.

**Code:** the loop body of `geocp_rs.pipeline.run_sacp_geocp`.

## Coverage guarantee

We now state the formal guarantee. Both pieces (the lemma and the proposition) appear verbatim as Lemma 1 and Proposition 1 in the paper.

### Lemma 1 (SACP smoothing preserves exchangeability)

Let $T_\lambda$ denote the SACP smoothing operator that maps a score sequence $(s_1, \ldots, s_n, s_{n+1})$ on the image grid to its smoothed version $(\tilde s_1, \ldots, \tilde s_n, \tilde s_{n+1})$. The operator $T_\lambda$ is a deterministic function that does not consult the calibration / test split — it only reads the spatial neighborhood graph $N(\cdot)$. Therefore, if $(s_1, \ldots, s_n, s_{n+1})$ is an exchangeable sequence of scalars, then $T_\lambda(s_1, \ldots, s_{n+1})$ is also exchangeable.

**Proof sketch.** The smoothing rule for pixel $i$ is

$$
\tilde s_i(y) = (1 - \lambda)\, s_i(y) + \lambda \cdot |N(i)|^{-1} \sum_{j \in N(i)} s_j(y).
$$

The set $N(i)$ depends only on $i$'s position on the image grid, not on its calibration / test status. So $T_\lambda$ is a fixed function $\mathbb{R}^{n+1} \to \mathbb{R}^{n+1}$ that does not single out any particular index. By the standard fact that a deterministic symmetric function of an exchangeable sequence stays exchangeable, the conclusion follows. $\square$

### Proposition 1 (Marginal coverage of SACP+GeoCP)

Let $\{(X_i, Y_i, \mathbf{c}_i)\}_{i=1}^n$ be a calibration sample and $(X_{n+1}, Y_{n+1}, \mathbf{c}_{n+1})$ a test point such that the joint sequence of $(n+1)$ tuples is exchangeable. Let $\mathcal{C}(X_{n+1})$ be the prediction set produced by SACP+GeoCP with parameters $\alpha, \lambda, h$. Then

$$
\Pr[Y_{n+1} \in \mathcal{C}(X_{n+1})] \;\ge\; 1 - \alpha.
$$

**Proof.**

1. **SACP smoothed scores remain exchangeable.** Apply Lemma 1 to the sequence $(s_1(Y_1), \ldots, s_{n+1}(Y_{n+1}))$ obtained by evaluating the per-pixel APS scores at the true labels. Exchangeability of the original tuples plus the symmetry of $T_\lambda$ gives that $(\tilde s_1(Y_1), \ldots, \tilde s_{n+1}(Y_{n+1}))$ is also exchangeable.

2. **Apply weighted CP on the smoothed sequence.** The GeoCP weights $w_{(n+1),i} = \exp(-\|\mathbf{c}_{n+1} - \mathbf{c}_i\|^2 / (2h^2))$ depend only on the coordinates $(\mathbf{c}_1, \ldots, \mathbf{c}_{n+1})$. They do not depend on $Y_{n+1}$ or on the smoothed scores themselves. By Theorem 1 of [Tibshirani, Barber, Candès, Ramdas (2019)](https://arxiv.org/abs/1904.06019), applying the weighted quantile rule to an exchangeable sequence of nonconformity scores with covariate-only weights yields the marginal coverage bound

$$
\Pr[\tilde s_{n+1}(Y_{n+1}) \le \hat q_{n+1}] \;\ge\; 1 - \alpha.
$$

3. **Sublevel set is the prediction set.** By construction $\mathcal{C}(X_{n+1}) = \{y : \tilde s_{n+1}(y) < \hat q_{n+1}\}$, so $Y_{n+1} \in \mathcal{C}(X_{n+1})$ iff $\tilde s_{n+1}(Y_{n+1}) < \hat q_{n+1}$. Combining with step 2 yields the bound. $\square$

### Scope of the guarantee

What Proposition 1 says — and what it does **not** say — matters for honest deployment:

- ✅ **Marginal coverage**: the bound holds in expectation over a fresh draw from the joint calibration/test distribution. Empirically (see `RESULTS_ANALYSIS.md`), the realized coverage on each dataset is within $\pm 0.01$ of the target $0.9$.
- ❌ **Class-conditional coverage**: not guaranteed. Some classes may under-cover while others over-cover; the marginal average still hits $1 - \alpha$. Class-conditional guarantees would require **Mondrian CP** stratified by class, which is orthogonal to what we do here.
- ❌ **Per-pixel conditional coverage**: $\Pr[Y_j \in C(X_j) \mid X_j = x_j]$ is **provably impossible** to guarantee in a distribution-free setting except for trivial $C(X) = \mathcal{Y}$ (Vovk 2012; Lei & Wasserman 2014). No method achieves this; SACP+GeoCP does not claim to.
- ❌ **Per-region coverage**: $\Pr[Y \in C(X) \mid X \in R]$ for a spatial sub-region $R$ is not directly guaranteed. Empirically, GeoCP's local thresholding does much better than a single global quantile here, but the proof above only delivers the marginal bound.
- ❌ **Cross-scene transfer**: if calibration is from Indian Pines but test is from Pavia University, the exchangeability assumption fails and Proposition 1 does not apply. We restrict to the standard transductive single-scene HSI evaluation protocol.

The relationship to the literature is summarized in this table:

| Method | Marginal cov. | Class-conditional | Spatial adaptation | Composability |
|---|---|---|---|---|
| Standard split CP | ✓ | ✗ | ✗ | base |
| SACP | ✓ | ✗ | scores | composable on score side |
| GeoCP | ✓ | ✗ | thresholds | composable on threshold side |
| **SACP+GeoCP** | **✓** | ✗ | **both** | **(this work)** |
| Mondrian CP (per class) | strict per-class | ✓ | ✗ | future work |

The empirical bottom line, documented in `RESULTS_ANALYSIS.md`: SACP+GeoCP attains a strictly lower mean interval score than every other row on all 5 HSI benchmarks, while paying nothing in the theoretical guarantee column.

## Bandwidth selection by 5-fold CV (no test-set leakage)

The bandwidth $h$ is the only hyperparameter introduced beyond standard SACP. Setting it correctly matters: too small and $\hat q_j$ relies on a handful of nearest calibration pixels (high variance); too large and it converges to the global quantile (no spatial adaptation). We select $h$ per dataset by **5-fold cross-validation on the calibration set only** — no test labels are touched at any point.

The procedure, implemented in `geocp_rs.evaluate.evaluate_all_cp_methods`:

1. Fix a candidate grid $h \in \{3, 5, 7, 10, 15, 20, 30, 50, 100\}$ pixels.
2. Partition calibration indices $\mathcal{I}_{\text{cal}}$ into 5 folds via `sklearn.KFold(n_splits=5, shuffle=True, random_state=42)`.
3. For each fold $k$ and each $h$:
   - Treat folds $\ne k$ as inner-calibration, fold $k$ as inner-validation.
   - Run the GeoCP step using the inner-cal scores and the inner-val coordinates, computing prediction sets on the inner-val pixels.
   - Evaluate the interval score on the inner-val labels.
4. Average IS across the 5 folds for each $h$ in the grid.
5. Pick $\hat h = \arg\min_h \text{mean IS}(h)$.
6. At test time, run GeoCP once with $h = \hat h$ on the full calibration set and report results on the test set.

The resulting median bandwidths recover each dataset's characteristic spatial scale automatically:

| Dataset | Median CV bandwidth (pixels) | Interpretation |
|---|---|---|
| Indian Pines | 10 | Small image, fine class patches |
| Salinas | 5 | Thin parallel crop strips |
| Pavia University | 100 | Large urban scene with big homogeneous blocks |
| KSC | 50 | Sparse coastal wetlands |
| Botswana | 100 | Very long narrow floodplain |

## Complexity

For an image of $H \times W = N$ pixels, $n_{\text{cal}}$ calibration pixels, $n_{\text{test}}$ test pixels, and $K$ classes:

- **APS scoring**: $O((n_{\text{cal}} + n_{\text{test}}) K \log K)$ — sort each row of softmax probabilities.
- **SACP smoothing**: $O(N K)$ — vectorized 8-shift convolution over the image.
- **Distance matrix** (test vs. cal): $O(n_{\text{cal}} n_{\text{test}})$.
- **Per-pixel weighted quantile**: $O(n_{\text{cal}} \log n_{\text{cal}})$ × $n_{\text{test}}$ test pixels.

On Pavia University (the largest scene, $610 \times 340 = 207{,}400$ pixels, $n_{\text{test}} \approx 21{,}000$, $K = 9$), the entire SACP+GeoCP step takes about 5 seconds on a modern laptop, on top of a few minutes of 3D-CNN training. Memory peaks at a few hundred MB for the distance matrix.

## File ↔ algorithm step mapping

| Algorithm step | Implementation |
|---|---|
| Step 1: APS scoring | `geocp_rs/core.py::aps_scores` |
| Step 2: SACP smoothing | `geocp_rs/sacp.py::sacp_smooth` (vectorized) |
| Step 3: GeoCP local threshold | `geocp_rs/geocp.py::geocp_local_threshold` (log-stable) |
| Step 4: Form prediction sets | tail of `geocp_rs/pipeline.py::run_sacp_geocp` |
| 5-fold CV bandwidth selection | `geocp_rs/evaluate.py::evaluate_all_cp_methods` |
| Coverage / size / IS metrics | `geocp_rs/metrics.py` |
| End-to-end driver (5 datasets × 10 seeds) | `scripts/run_all_experiments.py` |
| Aggregation to JSON / CSV / LaTeX | `scripts/aggregate_results.py` |
| Figure regeneration | `scripts/make_figures.py` |
| Qualitative 5×4 grid | `geocp_rs/viz.py::plot_qualitative_grid` |

Run `python examples/quick_start.py` for a 30-second smoke test on synthetic data.
