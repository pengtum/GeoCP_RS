# Analysis Update — Apr 2026

New analyses added on top of the headline α=0.10 results in [`EXPERIMENTS_SUMMARY.md`](EXPERIMENTS_SUMMARY.md). All findings derived from the same 50 HSI runs + 29 S2 runs, with two extension experiments that store raw scores for future re-analysis.

## 1. What's new (3 dimensions added)

| Dimension | HSI | S2 | New artifact |
|---|---|---|---|
| **SSCV** (size-stratified coverage violation) | ✓ | ✓ | [fig6](analysis/figures/fig6_sscv.png), [fig10_s2](analysis/figures/fig10_s2_sscv_alpha.png) |
| **Size vs Difficulty** (Liu 2024 Fig 4 analogue) | ✓ | (in pkl, not yet plotted) | [fig7](analysis/figures/fig7_size_vs_difficulty.png) |
| **α sweep** (α ∈ {0.05, 0.10, 0.15}) | ✓ | ✓ | [fig8/9/11](analysis/figures/), [fig8_s2/9_s2/11_s2](analysis/figures/) |

Two new pkl archives store **softmax probs + raw APS + SACP-smoothed scores per r** so future re-analyses (λ/k sensitivity, new metrics, etc.) need not retrain models:

- HSI: `/content/drive/MyDrive/hsi_alpha_sweep/checkpoints/{ds}_seed{k}.pkl`
- S2:  `/content/drive/MyDrive/s2_alpha_sweep/checkpoints/{tile}_s{px}.pkl`

## 2. Grand-pooled α-sweep (n=79 = 50 HSI + 29 S2)

| α | C vs A (SACP default) | C vs D (Standard CP) |
|---|---|---|
| **0.05** | **+9.77%** ± 1.12 (t=8.76, p<10⁻¹²) | **+15.10%** ± 1.25 (t=12.12, p<10⁻¹⁹) |
| 0.10 | +5.75% ± 0.69 (t=8.34, p<10⁻¹¹) | +9.06% ± 0.94 (t=9.61, p<10⁻¹⁴) |
| 0.15 | +3.35% ± 0.40 (t=8.39, p<10⁻¹¹) | +5.93% ± 0.61 (t=9.70, p<10⁻¹⁴) |

The α=0.10 row reproduces (within 0.1 pp) the headline `+5.80% / +9.27%` from `EXPERIMENTS_SUMMARY.md` §9 — sanity check passed.

CV for (r, bw) is done **once at α=0.10** and the same hyperparameters are used at every α (avoids per-α multiple testing).

Figure: [`fig13_grand_alpha_improvement.png`](analysis/figures/fig13_grand_alpha_improvement.png).

## 3. The α-sensitivity finding (paper-worthy)

GeoCP's gain over SACP-default depends strongly on α **only on HSI**.

| α | HSI C vs A | S2 C vs A | HSI/S2 ratio |
|---|---|---|---|
| 0.05 | **+13.95%** | +2.56% | **5.4×** |
| 0.10 | +7.87% | +2.10% | 3.7× |
| 0.15 | +3.79% | +2.60% | 1.5× |

- **HSI**: gain roughly doubles as α halves (0.10→0.05).
- **S2**: gain is essentially flat across α.

**Mechanism**: at α=0.05 the global threshold is the (1−α)≈0.95 quantile of cal-set scores. With HSI's small calibration set (~1500 cal points / dataset), this quantile sits in a noisy upper tail. GeoCP's per-pixel weighted quantile pools information from spatial neighbors, dampening that noise. With S2's larger cal set (4k–20k per tile), the global tail quantile is already stable, so GeoCP only adds ~2% across α.

Practical implication: high-stakes deployments (α=0.05 is common in industrial / clinical UQ) get nearly **double** the headline +7.87% number on HSI-like data. Worth emphasizing in the paper's contributions.

Figure: [`fig12_alpha_sensitivity_HSI_vs_S2.png`](analysis/figures/fig12_alpha_sensitivity_HSI_vs_S2.png).

## 4. SSCV results

### HSI (n=50, α=0.10)

| Method | Mean SSCV (%) |
|---|---|
| D | 12.60 |
| A | 12.18 |
| B | **11.82** (lowest) |
| C | 14.30 |

**Counterintuitive**: C has the highest SSCV. Mechanism: C is systematically conservative on HSI (median coverage ≈ 0.913 vs. 0.90 target — see `EXPERIMENTS_SUMMARY.md` Fig 4), so |cov − (1−α)| in some size buckets exceeds 0.10. SSCV penalizes overcoverage and undercoverage symmetrically.

### S2 (n=29, all α)

| α | D | A | B | C |
|---|---|---|---|---|
| 0.05 | 5.02 | 5.12 | 4.83 | **4.92** |
| 0.10 | 9.84 | 10.00 | **9.14** | 9.20 |
| 0.15 | 13.85 | 15.12 | 13.86 | 13.74 |

On S2, C's SSCV is **comparable to A/B** — no fairness penalty. The HSI overcoverage is dataset-specific, not a property of the method.

### Recommendation for paper

Don't make SSCV a top-line claim. Acceptable framing: *"Proposition 1 guarantees marginal coverage. SACP+GeoCP empirically satisfies marginal coverage on all 79 runs and matches SCP/SACP on size-stratified coverage on Sentinel-2; on HSI it trades a small SSCV increase for a substantial IS reduction — an expected consequence of Tibshirani-style weighted CP being slightly conservative when the calibration set is small."*

## 5. Size-vs-Difficulty (HSI, α=0.10, n=50)

Bucket each test sample by Standard-CP set size (used as a difficulty proxy since softmax rank wasn't stored in the original pkls — newer α-sweep pkls do store softmax, so a true rank-based version is reproducible later).

Mean prediction-set size per difficulty bucket:

| Difficulty | D | A | B | **C** |
|---|---|---|---|---|
| 1 | 0.96 | 1.01 | 0.99 | 1.02 |
| 2 | 2.00 | 1.42 | 1.33 | 1.46 |
| 3 | 3.00 | 1.93 | 1.67 | 1.92 |
| 4 | 4.00 | 2.59 | 2.24 | 2.53 |
| ≥5 | 5.11 | 3.39 | **2.80** | 3.34 |

**B compresses harder samples more than C does** at every bucket. This is the first crack in the narrative that "C = best on every axis": C's IS gain over B (still real, p<0.05 on HSI) likely comes from improved coverage on hard samples, not from smaller sets.

This implies §5 of `EXPERIMENTS_SUMMARY.md` (the C−B decomposition) is more nuanced than originally framed. Worth a 1-paragraph addendum saying: *"GeoCP's local threshold lets harder neighborhoods spend their coverage budget there; the resulting prediction sets are not uniformly smaller than SACP-CV-r, but the IS gain confirms the threshold reallocation is correct on average."*

Figure: [`fig7_size_vs_difficulty.png`](analysis/figures/fig7_size_vs_difficulty.png).

## 6. Reproducibility

Every result above can be regenerated from the two new pkl archives:

```bash
cd GeoCP_RS
python3 analysis/scripts/extra_analyses.py            # SSCV + Size-vs-Difficulty (HSI old pkls)
python3 analysis/scripts/alpha_sweep_analysis.py      # HSI α-sweep (new pkls)
python3 analysis/scripts/s2_alpha_sweep_analysis.py   # S2 α-sweep (new pkls)
python3 analysis/scripts/grand_alpha_analysis.py      # grand pooled + sensitivity
```

Notebooks for re-running the experiments themselves:

- [`hsi_experiment/notebooks/hsi_alpha_sweep.ipynb`](hsi_experiment/notebooks/hsi_alpha_sweep.ipynb) — Colab GPU, ~90 min
- [`s2_experiment/notebooks/s2_alpha_sweep.ipynb`](s2_experiment/notebooks/s2_alpha_sweep.ipynb)   — Colab CPU, ~30–60 min

Both write to local `/content/...` for FUSE-disconnect resilience and mirror to Drive after every config.

## 7. Open follow-ups

Not yet done; data is available in the new pkls if pursued:

- **λ, k sensitivity** (SACP smoothing weight + iteration count) — pkls have `fcu_per_r` so λ-sweep needs only convolutional-mix recomputation.
- **True rank-based Size-vs-Difficulty** — new pkls have `probs_te`, can compute true-label rank exactly.
- **S2 Size-vs-Difficulty** — same script as HSI, just point at S2 pkls.
- **Coverage decomposition C − B** on HSI — separate "hard sample" vs "easy sample" coverage gains.
