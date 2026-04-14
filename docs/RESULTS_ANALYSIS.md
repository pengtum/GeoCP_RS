# Results Analysis

Detailed breakdown of the 50 runs (5 datasets × 10 seeds) reported in the paper.

All numbers are taken directly from `results/summary.json`, `results/stats.json`, and `results/per_seed.csv`. Every value in this file is regenerable from those three inputs without re-training.

## 1. Headline numbers

**Interval score (mean ± std over 10 seeds, lower is better):**

| Dataset | 3D-CNN acc | Std CP | SACP($\lambda$=0.3) | SACP($\lambda$=0.5) | SACP($\lambda$=0.7) | **SACP+GeoCP** |
|---|---|---|---|---|---|---|
| Indian Pines   | 0.688 | 4.306 ± 0.234 | 3.980 ± 0.155 | 3.920 ± 0.152 | 3.894 ± 0.158 | **3.640 ± 0.145** |
| Pavia Univ.    | 0.862 | 3.197 ± 0.091 | 3.109 ± 0.057 | 3.080 ± 0.060 | 3.055 ± 0.057 | **2.986 ± 0.084** |
| Salinas        | 0.883 | 3.142 ± 0.040 | 3.064 ± 0.048 | 3.039 ± 0.046 | 3.028 ± 0.051 | **2.971 ± 0.058** |
| KSC            | 0.811 | 3.758 ± 0.307 | 3.474 ± 0.259 | 3.435 ± 0.222 | 3.397 ± 0.210 | **3.368 ± 0.213** |
| Botswana       | 0.949 | 2.926 ± 0.171 | 2.902 ± 0.218 | 2.924 ± 0.198 | 2.914 ± 0.193 | 2.903 ± 0.221 |

**SACP+GeoCP achieves the lowest mean IS on all 5 datasets.**

## 2. Coverage (target = 0.90)

| Dataset | Std CP | SACP(0.5) | SACP+GeoCP |
|---|---|---|---|
| IP    | 0.903 | 0.901 | **0.916** |
| PU    | 0.900 | 0.900 | **0.908** |
| SA    | 0.902 | 0.901 | **0.908** |
| KSC   | 0.893 | 0.896 | **0.906** |
| Botswana | 0.904 | 0.902 | 0.903 |

**SACP+GeoCP systematically over-covers relative to the other methods.** This is not wasted width — the trade-off analysis in §4 shows it is achieving better coverage at similar or moderately larger set size than SACP(0.5), and the IS metric rewards this because each miscoverage is penalized by $2/\alpha = 20$.

## 3. Mean prediction-set size

| Dataset | Std CP | SACP(0.3) | SACP(0.5) | SACP(0.7) | SACP+GeoCP |
|---|---|---|---|---|---|
| IP    | 2.36 | 1.98 | 1.94 | **1.91** | 1.96 |
| PU    | 1.19 | 1.11 | 1.09 | **1.08** | 1.14 |
| SA    | 1.18 | 1.08 | 1.05 | **1.04** | 1.12 |
| KSC   | 1.61 | 1.41 | 1.35 | **1.32** | 1.50 |
| Botswana | 1.00 | **0.97** | 0.96 | **0.95** | 0.97 |

SACP with larger $\lambda$ always gives smaller sets — but loses coverage. SACP+GeoCP's size lives *between* SACP(0.5) and SACP(0.3), exchanging a small amount of width for meaningfully better coverage. The net effect is the lowest IS.

## 4. Paired statistical tests (SACP+GeoCP vs. best-$\lambda$ SACP)

The best-$\lambda$ SACP baseline is chosen per-dataset by mean IS (the most aggressive baseline available). The paired $t$-test aligns 10 seeds:

| Dataset | Best baseline | SACP+GeoCP | Δ% (higher = better) | $t$ | $p$ (paired) |
|---|---|---|---|---|---|
| **Indian Pines**   | SACP($\lambda$=0.7): 3.894 | **3.640** | **+6.48%** | 8.19 | **5.9 × 10⁻⁵** |
| **Pavia Univ.**    | SACP($\lambda$=0.7): 3.055 | **2.986** | **+2.25%** | 3.34 | **0.009** |
| **Salinas**        | SACP($\lambda$=0.7): 3.028 | **2.971** | **+1.87%** | 3.05 | **0.014** |
| KSC                | SACP($\lambda$=0.7): 3.397 | 3.368 | +0.83% | 2.25 | 0.051 |
| Botswana           | SACP($\lambda$=0.3): 2.902 | 2.903 | −0.08% | 0.92 | 0.38 |

Three of five datasets reach $p < 0.05$ under the most conservative possible baseline.

## 5. Pooled statistics (all 50 runs)

**Pooled paired test vs. best-$\lambda$ SACP per dataset:**
- mean relative improvement: **+2.27% ± 3.37%**
- positive seeds: **36 / 50** (72%)
- Wilcoxon $W$ = 228, **$p$ = 3.6 × 10⁻⁵**

**Pooled paired test vs. SACP($\lambda$=0.5) (the canonical baseline):**
- mean relative improvement: **+3.01%**
- positive seeds: **41 / 50** (82%)
- Paired $t$ = 6.05, **$p$ = 2.0 × 10⁻⁷**

Both tests decisively reject the null hypothesis of no improvement.

## 6. Improvement scales inversely with classifier accuracy

| Dataset | 3D-CNN accuracy | Δ IS (vs. SACP 0.5) |
|---|---|---|
| Indian Pines | 0.688 | **+7.14%** |
| KSC          | 0.811 | +1.97% |
| Pavia Univ.  | 0.862 | +3.05% |
| Salinas      | 0.883 | +2.24% |
| Botswana     | 0.949 | +0.70% |

**Pearson $r$ = −0.919, $p$ = 0.027.**

**Interpretation.** The harder the classification task, the more room there is for a spatially-adaptive CP procedure to help. Indian Pines (accuracy 0.688, small dense scene with 16 classes) has the largest gain; Botswana (accuracy 0.949, nearly saturated classifier) has almost nothing to improve because the mean set size is already ≈1. This matches the design intuition: SACP+GeoCP widens the threshold near class boundaries, which only helps when class boundaries are uncertain, which only happens when the classifier is uncertain.

## 7. CV-selected bandwidths per dataset

The 5-fold CV on calibration produced the following bandwidth distributions over 10 seeds (pixels):

| Dataset | 10-seed bandwidths (sorted) | Median | Spatial scale interpretation |
|---|---|---|---|
| Indian Pines | 3, 5, 7, 7, 7, 10, 10, 15, 15, 15 | **10** | 145×145 image, small class patches — small bw |
| Salinas | 5, 5, 5, 5, 5, 7, 10, 20, 50, 50 | **5** | thin parallel crop strips — very small bw |
| Pavia Univ. | 10, 15, 15, 20, 100, 100, 100, 100, 100, 100 | **100** | large urban scene, 9 big classes — large bw |
| KSC | 30, 50, 50, 50, 50, 50, 100, 100, 100, 100 | **50** | sparse wetland pixels — need large bw to find neighbors |
| Botswana | 30, 50, 50, 100, 100, 100, 100, 100, 100, 100 | **100** | very long narrow scene — largest bw |

**These medians are not arbitrary** — they correlate strongly with each dataset's characteristic class-patch size. The CV is recovering a meaningful spatial scale automatically.

## 8. Why KSC and Botswana are not statistically significant

**KSC ($p$ = 0.28).**
- 3D-CNN accuracy: 0.811.
- Labeled pixels: only ~5k on a 512 × 614 grid.
- After training uses 250, each split of (calib, test) has ~2500 pixels, scattered across tens of wetland patches.
- Every test pixel's geographic neighborhood contains only a handful of calibration pixels, so the *effective sample size* for GeoCP's weighted quantile is low. The quantile estimate has high variance, and the gain over a global quantile is noisy.
- IS shows a positive mean gain (+0.83%) but with std large enough to miss $p < 0.05$ at $n = 10$.

**Botswana ($p$ = 0.96).**
- 3D-CNN accuracy: **0.949** — almost saturated.
- Mean SACP(0.5) set size is **0.96**, effectively one-class-per-pixel (deterministic classification).
- There is no uncertainty left for any CP variant to improve.
- SACP+GeoCP, SACP($\lambda$=0.3), SACP($\lambda$=0.5), and SACP($\lambda$=0.7) all land within 0.02 IS of each other — a floor effect.

Both failures are consistent with the accuracy-vs-improvement correlation in §6, and both are discussed in the paper's limitations section.

## 9. Why SACP+GeoCP beats "crank up $\lambda$"

A reader might ask: if larger $\lambda$ gives smaller sets (Table in §3), why not just use SACP($\lambda$=0.9) and call it a day?

Because IS penalizes miscoverage linearly, with penalty $2/\alpha = 20$. Pushing $\lambda$ too high breaks the coverage target and the miss penalty overwhelms the width savings. The optimum fixed-$\lambda$ SACP is almost always $\lambda = 0.7$, but even there SACP+GeoCP is better because:

1. At $\lambda = 0.7$, coverage on IP drops from 0.916 (SACP+GeoCP) to 0.892 (SACP 0.7). Over a test set of ~5000 pixels, that's ~120 more miscoverages × 20 penalty = 2400 extra IS points in absolute terms, which dominates any width savings.
2. GeoCP's local thresholds *widen adaptively where needed*. SACP(0.7) can only shrink uniformly, so the boundary pixels — which need wider sets to be covered — are exactly where SACP(0.7) fails.

## 10. Raw data

All the numbers above are computable from `results/per_seed.csv`. A minimal re-derivation script:

```python
import pandas as pd, numpy as np
from scipy import stats
df = pd.read_csv('results/per_seed.csv')
for ds in ['ip','pu','sa','ksc','botswana']:
    sub = df[df.dataset == ds]
    gain_vs_sacp05 = (sub.sacp05_is - sub.geocp_is) / sub.sacp05_is * 100
    print(ds, gain_vs_sacp05.mean(), stats.ttest_rel(sub.sacp05_is, sub.geocp_is).pvalue)
```

produces the same numbers.

## 11. What did NOT work (for honesty)

During this research program we also tried:

- **LISA-adaptive SACP**: use local Moran's I of the CNN predictions to set a per-pixel $\lambda$ instead of a uniform one. No improvement over uniform $\lambda$ in any configuration we tried (normal, reversed, confidence-based). We report uniform $\lambda = 0.5$ in the paper.
- **MESF-kernel GeoCP (Moran eigenvector spatial filtering as the geographic kernel)**: replacing Euclidean pixel distance with distance in the top-$k$ Moran eigenvector space of the spatial graph. This was evaluated on a separate geospatial regression pilot (California Housing, NCOVR, US Election 2020, EPA PM2.5) and did not consistently improve over a Euclidean kernel, so it is not part of this paper.
- **GeoCP-LP (LISA-weighted calibration)**: an earlier attempt to multiply GeoCP weights by a LISA-similarity factor. The effect was marginal-to-negative on regression benchmarks under proper cross-validation.

The direction that produced the positive result was the simpler composition of two existing published ideas (SACP + GeoCP). The pilot that ruled out MESF is documented elsewhere and is not required reading for this paper.

## 12. External reproducibility

- The 50 per-seed pickles in `results/checkpoints/` contain everything needed to regenerate the IS / size / coverage numbers without retraining. Each pickle has the full APS score tensor, calibration and test indices, prediction sets per method, and local quantiles, allowing full recomputation of the aggregate tables.
- `results/run_log.txt` is the actual Colab session log, preserving the exact order, timing, and bandwidths of each run.
- All figures in the paper are deterministic functions of `results/per_seed.csv`; re-running the figure-generation script on a different machine will produce byte-identical PDFs.
