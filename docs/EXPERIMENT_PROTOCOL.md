# Experiment Protocol

Exact specification for reproducing every number in the paper.

## 1. Datasets

Five standard hyperspectral image (HSI) benchmarks, all publicly available from the University of the Basque Country (EHU) Computational Intelligence Group. The Colab notebook (`code/sacp_geocp_colab.ipynb`) downloads them automatically with `wget`.

| Short name | Full name | Shape (H × W × B) | Classes | Total labeled pixels | Source URL |
|---|---|---|---|---|---|
| `ip` | Indian Pines | 145 × 145 × 200 | 16 | 10,249 | `https://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat` + `/c/c4/Indian_pines_gt.mat` |
| `pu` | Pavia University | 610 × 340 × 103 | 9 | 42,776 | `/e/ee/PaviaU.mat` + `/5/50/PaviaU_gt.mat` |
| `sa` | Salinas | 512 × 217 × 204 | 16 | 54,129 | `/a/a3/Salinas_corrected.mat` + `/f/fa/Salinas_gt.mat` |
| `ksc` | Kennedy Space Center | 512 × 614 × 176 | 13 | 5,211 | `/2/26/KSC.mat` + `/a/a6/KSC_gt.mat` |
| `botswana` | Botswana | 1476 × 256 × 145 | 14 | 3,248 | `/7/72/Botswana.mat` + `/5/58/Botswana_gt.mat` |

All URL prefixes are `https://www.ehu.eus/ccwintco/uploads/`.

## 2. Pre-processing

For each dataset, identically:

1. Cast HSI cube to `float32`.
2. Mean-center per band, divide by the global max: `(hsi - hsi.mean(axis=(0,1))) / hsi.max()`.
3. Transpose to channels-first `(bands, H, W)`.

Label ID 0 indicates unlabeled background and is excluded from train/calibration/test selection.

## 3. Train / calibration / test split

For each `(dataset, seed)` combination:

1. Collect the flat indices of all labeled pixels.
2. Stratified-sample **250 training pixels** (`sklearn.model_selection.train_test_split(train_size=250, stratify=y)`).
3. Split the remaining labeled pixels **50/50** into calibration and test, stratified by class.
4. Random state = `seed * 100 + 42`.

This yields approximately `{5000, 21000, 27000, 2500, 1500}` calibration pixels (and the same for test) on `{ip, pu, sa, ksc, botswana}` respectively.

## 4. Input patches

Each pixel is represented by a 5 × 5 spatial patch of the full spectral cube, extracted with reflect padding at the image borders. Patch tensor shape: `(N, 1, bands, 5, 5)`.

## 5. 3D-CNN architecture

Implementation follows Hamida et al. (2018), as used in the original SACP paper. PyTorch definition is in `code/zoo.py`.

Layer sequence:

```
Conv3d(1  → 20, kernel=(3,3,3), stride=1, padding=0)
Conv3d(20 → 20, kernel=(3,1,1), stride=(2,1,1), padding=(1,0,0))   # pool spectral
Conv3d(20 → 35, kernel=(3,3,3), stride=1, padding=(1,0,0))
Conv3d(35 → 35, kernel=(3,1,1), stride=(2,1,1), padding=(1,0,0))   # pool spectral
Conv3d(35 → 35, kernel=(3,1,1), stride=1, padding=(1,0,0))
Conv3d(35 → 35, kernel=(2,1,1), stride=(2,1,1), padding=(1,0,0))   # final pool
Linear(flattened → n_classes)
```

Nonlinearity is ReLU after each conv. The feature-map size before the linear layer is computed once at model construction from a zero forward pass.

## 6. Training

- **Optimizer**: Adam, lr = 1e-3, default momentum/weight-decay.
- **Loss**: CrossEntropyLoss.
- **Batch size**: 64.
- **Epochs**: 100.
- **Seed**: `torch.manual_seed(seed*100+42)` and `np.random.seed(seed*100+42)`.

Training completes in ~1–3 minutes per run on a T4 GPU and ~2–4 minutes on an Apple M1/M2 MPS backend (larger scenes are slower).

## 7. Conformal prediction variants

After training, we evaluate five CP methods on the same (softmax_cal, softmax_test) outputs:

| Method | SACP λ | Threshold |
|---|---|---|
| Standard CP | 0 (no smoothing) | global `conformal_quantile` |
| SACP($\lambda=0.3$) | 0.3 | global `conformal_quantile` |
| SACP($\lambda=0.5$) | 0.5 | global `conformal_quantile` |
| SACP($\lambda=0.7$) | 0.7 | global `conformal_quantile` |
| **SACP+GeoCP** | 0.5 | per-pixel `geocp_local_threshold` with CV bw |

Target miscoverage $\alpha = 0.1$ throughout.

## 8. Bandwidth selection (CV, no test-set leakage)

For SACP+GeoCP, the only free hyperparameter is the Gaussian kernel bandwidth $h$. We select it by **5-fold cross-validation on the calibration set**:

1. Partition `cal_indices` into 5 folds via `sklearn.KFold(n_splits=5, shuffle=True, random_state=42)`.
2. For each fold $k$:
   - Treat the other 4 folds as inner-calibration and fold $k$ as inner-validation.
   - For each bandwidth $h \in \{3, 5, 7, 10, 15, 20, 30, 50, 100\}$, compute SACP+GeoCP prediction sets on the inner-validation fold using the inner-calibration smoothed scores.
   - Compute the **interval score** on the inner-validation fold.
3. Average the 5 inner-validation interval scores per $h$.
4. Choose the $h$ with the lowest mean inner-validation IS.

**No test label is accessed during selection.** The selected bandwidth is then used once to produce the reported test-set numbers.

The chosen bandwidths per dataset (median over 10 seeds) are:

| Dataset | Median CV bandwidth (pixels) | Interpretation |
|---|---|---|
| Indian Pines | 10 | dense, small image, fine-grained class patches |
| Salinas | 5 | thin parallel crop strips |
| Pavia University | 100 | large sparse urban scene |
| KSC | 50 | sparse coastal wetland |
| Botswana | 100 | very long narrow floodplain |

## 9. Metrics

All metrics are reported as mean ± std over 10 seeds per dataset.

- **Coverage**: fraction of test pixels whose true label falls inside the returned prediction set. Target: 0.9 (since $\alpha = 0.1$).
- **Mean prediction-set size**: average number of classes inside the set. Lower is better *given* valid coverage.
- **Interval Score** (Gneiting & Raftery, 2007, adapted to sets):
  $$\mathrm{IS}(\mathcal{C}, Y; \alpha) = |\mathcal{C}| + \frac{2}{\alpha}\,\mathbf{1}\{Y \notin \mathcal{C}\}.$$
  Lower is better. Combines size and miscoverage into a single proper scoring rule, so it is unambiguous even when coverage and size point in opposite directions.

## 10. Statistical tests

- Per-dataset: **paired $t$-test** on the 10-seed-aligned IS of SACP+GeoCP vs.\ the best-$\lambda$ SACP baseline for that dataset.
- Pooled (all 50 runs): **paired $t$-test** and **Wilcoxon signed-rank** test, same alignment.

All reported $p$-values come from `scipy.stats.ttest_rel` and `scipy.stats.wilcoxon`.

## 11. Random seeds

Seeds ∈ `{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}`. For each seed, we set:

```python
torch.manual_seed(seed * 100 + 42)
np.random.seed(seed * 100 + 42)
```

APS randomization inside the conformal procedure uses a separate `np.random.RandomState(seed*100+42)` so that the APS tie-break is reproducible but decoupled from training randomness.

## 12. Hardware used to produce the reported results

- **Colab runtime**: T4 GPU (NVIDIA Tesla T4, 16 GB VRAM).
- **Total wall time** for 50 runs × 5 CP variants: ~1 hour 20 minutes end-to-end (including dataset downloads).
- Per-seed breakdown (on T4): IP ~20s, PU ~70s, SA ~50s, KSC ~15s, Botswana ~8s. Running for 10 seeds: ~45 s/seed averaged.

## 13. Reproducing locally without Colab

`code/train_and_eval.py` is an equivalent Python driver for MPS (Apple Silicon) or CUDA. It supports:

```bash
python3 train_and_eval.py ip pu sa
```

and will write checkpoints next to itself. Note: KSC and Botswana may hit memory limits on MPS due to unified memory pressure; for those two datasets we recommend using Colab instead.

## 14. Invariants we rely on

- **Stratified sampling ensures all classes appear in train/cal/test.** Without this, some rare classes in IP/KSC wouldn't have enough calibration examples for a meaningful quantile.
- **5-fold CV for bandwidth selection must use a fixed RandomState** (we use `random_state=42`), so that different seeds produce the same folding given their `cal_indices`.
- **The SACP smoothing is applied uniformly to calibration and test score entries**; this is what preserves marginal coverage.
- **The GeoCP weights are computed from pixel coordinates only**, never from labels. Otherwise the weighted-CP guarantee breaks.

## 15. File-to-section mapping

| Paper Section | File(s) providing it |
|---|---|
| §3 Background (formulas) | `code/sacp_geocp_core.py` |
| §4 Algorithm 1 | `code/sacp_geocp_core.py::run_sacp_geocp` |
| §5 Experimental setup | this document |
| Table 2 | `results/results_table.tex` |
| Figure 1 (IS bars) | `figures/fig_is_bars.pdf` |
| Figure 2 (acc vs imp) | `figures/fig_acc_vs_improvement.pdf` |
| Figure 3 (cov vs size) | `figures/fig_coverage_vs_size.pdf` |
| Figure 4 (qualitative) | `figures/fig_qualitative.png` |
| Figure 5 (bandwidth) | `figures/fig_bandwidth_selection.pdf` |
| Pooled statistics | `results/stats.json` |
| Per-seed raw numbers | `results/per_seed.csv` |
