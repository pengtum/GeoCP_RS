# Unified Analysis — Figures and Tables for HSI + S2 Experiments

All paper-ready figures for both experiments, generated in a single strictly-unified style.

## Quick index

| File                                  | Content                                                                     |
| ------------------------------------- | --------------------------------------------------------------------------- |
| `fig1_headline_bars.png`              | Mean IS of (D, A, B, C) × (5 HSI datasets, 3 S2 patch sizes), with ±std     |
| `fig2_paired_improvements.png`        | Paired % improvement (C vs A) and (C vs D) per dataset/size + pooled, ± s.e.m., significance stars |
| `fig3_radius_distribution.png`        | Histogram of CV-selected Moore radius r for methods B and C on each experiment |
| `fig4_coverage_check.png`             | Box-and-strip plot of per-run coverage for each method (target α = 0.10 reference line) |
| `fig5_per_run_scatter.png`            | Per-run (C vs A) scatter — every dot below the diagonal is a paired win for C |
| `spatial_hsi_{ip,pu,sa,ksc,botswana}.png` | 5-panel spatial map per HSI dataset (representative seed)                |
| `spatial_s2_{polk_iowa,sacramento_ca,lubbock_tx}_s{100,200,500}.png` | 5-panel spatial map per S2 (tile, size)     |

Unified style across all 19 figures:

- **Method colors** (fixed):
  - **(D) Standard CP** → gray `#9e9e9e`
  - **(A) SACP default** (r=1) → blue `#1f77b4`
  - **(B) SACP-CV-r** → green `#2ca02c`
  - **(C) SACP+GeoCP** → red `#d62728`
- **Set-size colormap**: `viridis`, shared vmin/vmax between A and C panels in each spatial figure
- **q_j colormap**: `magma`, per-figure 2–98% percentile clip
- **Typography**: DejaVu Sans, 10 pt body / 11 pt title / 8.5 pt tick labels
- **DPI**: 180. Tight layout. No top/right spines.

## Summary figures (paper Section 4)

### `fig1_headline_bars.png`

Mean Interval Score of the four methods on (left) five HSI datasets and (right) three S2 patch sizes.

- HSI: C (red) is visibly lower than A (blue) on **every** dataset. The gap is largest on Indian Pines (lowest classifier accuracy).
- S2: the four bars are closer together (smaller effect size than HSI), but C remains below D/A/B at all three scales.

### `fig2_paired_improvements.png`

Paired improvements C-vs-A (blue) and C-vs-D (gray) with ± s.e.m. and significance stars (`****` p<10⁻⁴, `***` p<10⁻³, `**` p<10⁻², `*` p<0.05, `ns` not significant).

Key reading:

- HSI: **C vs A stars are all `****` except Botswana (ns)**. Pooled n=50 gives C vs A p < 10⁻⁴.
- S2: C vs A is **ns at 1 km** (underpowered at n=9), significant at 2 km and 5 km, highly significant pooled (n=29).

### `fig3_radius_distribution.png`

Counts of CV-selected radius r across runs. **r=1 (SACP default) vertical line is annotated**. 

- HSI (n=50): **C never selects r=1** (0/50); prefers r=5 (20×) and r=10 (27×). B selects r=1 once (Botswana seed 8).
- S2 (n=29): same pattern at smaller scales — C prefers r=2 (16×) with some r=5/10; neither B nor C ever picks r=1.

Headline: **0 out of 79 joint-CV runs select SACP's default 3×3 Moore neighborhood**.

### `fig4_coverage_check.png`

Per-run coverage distribution. Dashed line = α=0.10 target (0.90 coverage).

- All methods' medians sit exactly at 0.90 on both experiments.
- Method C on HSI is **slightly conservative** (median ≈ 0.913) — GeoCP's per-pixel q_j pads uncertain regions. This is expected from Proposition 1 (Tibshirani-style weighted CP tends toward conservative coverage).
- No method under-covers.

### `fig5_per_run_scatter.png`

Each dot is one run. X-axis = (A) SACP default IS, Y-axis = (C) SACP+GeoCP IS. Below the diagonal = C better.

- HSI: **46/50 runs below diagonal**, colored by dataset. Indian Pines points are furthest below (biggest relative gains).
- S2: **23/29 runs below diagonal**, colored by patch size (viridis). 5 km points cluster closest together (lowest variance).

## Spatial figures (paper Section 4.4)

Each spatial figure has the same 5-panel layout:

```
[ GT ]  [ Prediction / Test pixels ]  [ (A) SACP size ]  [ (C) SACP+GeoCP size ]  [ q_j ]
```

Set-size panels 2 & 3 share the same viridis colormap range (computed per figure from max of the two), so differences between (A) and (C) are directly visible as brightness shifts. q_j panel uses magma with 2–98% percentile clip.

### HSI (5 figures, one per dataset)

Representative seed chosen as the one whose C-vs-A improvement is closest to the dataset mean (a typical run). Panel 1 is the categorical ESA-style GT; panel 2 is the 3D-CNN's predicted class on the test pixels (recolored with the same categorical cmap).

- `spatial_hsi_ip.png` — Indian Pines. Large q_j spatial variation (sd=0.109); visibly different set-size patterns between (A) and (C).
- `spatial_hsi_pu.png` — Pavia University. Urban scene; bw=10 selected; clear per-region q_j variation.
- `spatial_hsi_sa.png` — Salinas. Crop stripes; thin CV-selected bw exploits local structure.
- `spatial_hsi_ksc.png` — KSC. Sparse wetlands; bw selected at 50; q_j tracks local class-imbalance.
- `spatial_hsi_botswana.png` — Botswana. Long strip scene; near-flat q_j (sd=0.010); SACP+GeoCP reduces to approximately global threshold.

### S2 (9 figures, 3 tiles × 3 patch sizes)

- `polk_iowa` (row-crop)
- `sacramento_ca` (delta, the tile with the largest per-tile C-vs-A improvement at 1 km: +5.45%)
- `lubbock_tx` (irrigated dryland)

For each: s100 (1 km), s200 (2 km), s500 (5 km). Panel 2 shows which pixels are in the test split (colored by ESA GT, remaining pixels gray-masked). Panels 3 and 4 show (A) vs (C) set size on the dense test grid.

## Data sources (read by `scripts/generate_analysis.py`)

- `hsi_experiment/results/per_seed.csv` — 50 rows
- `hsi_experiment/results/checkpoints/*.pkl` — 50 pickles (for spatial)
- `s2_experiment/results/csv/main_comparison_multisize.csv` — 29 rows
- `s2_experiment/results/spatial_maps/*.npz` — 29 dense npz (for spatial)
- `sacp_geocp/datasets/*/` (external to repo, Drive only) — HSI `.mat` files for GT raster

## Regenerating

```bash
cd GeoCP_RS
python3 analysis/scripts/generate_analysis.py
```

~30 seconds total. Overwrites everything in `analysis/figures/`.

## Paper figure mapping

| Paper section    | Figure                                          | Claim                                                                                               |
| ---------------- | ----------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| 4.1 HSI          | `fig1` left panel, `fig2` left panel            | C beats A by +7.94% pooled (p < 10⁻¹¹) across 5 benchmarks × 10 seeds.                              |
| 4.2 S2           | `fig1` right panel, `fig2` right panel          | C beats A by +2.10% pooled (p < 10⁻⁴) across 10 tiles × 3 sizes.                                    |
| 4.3 Joint radius | `fig3`                                          | First systematic ablation of SACP's neighborhood radius; r=1 is never CV-selected by C (0/79 runs). |
| 4.4 Coverage     | `fig4`                                          | Marginal coverage empirically verified on 79 × 4 = 316 method × run evaluations.                    |
| 4.5 Per-run wins | `fig5`                                          | Paired comparison: C wins 46/50 on HSI and 23/29 on S2.                                             |
| 4.6 Spatial $\hat q_j$ | `spatial_hsi_*.png`, `spatial_s2_*.png`  | Per-pixel threshold field as a new output type (second-order uncertainty).                          |
