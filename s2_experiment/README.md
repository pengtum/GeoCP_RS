# S2 Experiment — SACP+GeoCP with joint CV on (r, bw)

Self-contained folder for the Sentinel-2 land-cover experiment that produced the paper's headline comparison.

## Folder layout

```
s2_experiment/
├── README.md                                      (this file)
├── notebooks/
│   ├── sentinel2_landcover_pilot.ipynb           (GEE pull + initial pipeline)
│   └── multiscale_ablation.ipynb                 (final main comparison)
├── data/
│   └── tiles/                                    (10 × .npz: 13-band S2 + ESA label)
│       ├── polk_iowa.npz       ... lancaster_pa.npz
│       ├── hartford_ct.npz     ... everglades_fl.npz
│       ├── lubbock_tx.npz      ... sacramento_ca.npz
│       ├── phoenix_az.npz      ... yellowstone_wy.npz
│       └── seattle_wa.npz      ... mississippi_la.npz
└── results/
    ├── csv/                                      (7 CSVs, numeric outputs)
    ├── figures/                                  (9 PNGs)
    ├── checkpoints/                              (29 JSONs, per (tile, size) intermediate)
    └── spatial_maps/                             (29 NPZs, per-pixel set sizes + q_j)
```

## Headline results (from `results/csv/main_comparison_multisize.csv`)

Paired comparison over n = 29 (tile, size) configurations from 10 Sentinel-2 tiles × 3 sizes (1, 2, 5 km):

| Comparison                          | Mean       | Wins   | t     | p      |
| ----------------------------------- | ---------- | ------ | ----- | ------ |
| SACP default (A) vs Std CP (D)      | +1.88%     | 19/29  | +2.76 | 0.010  |
| SACP-CV-r (B) vs SACP default (A)   | +1.79%     | 20/29  | +3.68 | 0.001  |
| **SACP+GeoCP (C) vs SACP default (A)** | **+2.09%** | **23/29** | **+4.82** | **< 0.0001** |
| SACP+GeoCP (C) vs SACP-CV-r (B)     | +0.27%     | 15/29  | +0.66 | 0.515  |
| SACP+GeoCP (C) vs Std CP (D)        | +3.98%     | 25/29  | +6.45 | < 0.0001 |

**Per-size pattern (effect strengthens with size)**:

| Size   | n  | C vs A (%) | p         | wins  |
| ------ | -- | ---------- | --------- | ----- |
| 1 km   | 9  | +1.31      | 0.186     | 6/9   |
| 2 km   | 10 | +2.15      | 0.043     | 7/10  |
| 5 km   | 10 | **+2.75**  | **< 0.0001** | **10/10** |

**CV-selected radius distribution (29 configurations)**:
- SACP-CV-r: r=2 (11), r=3 (12), r=5 (2), r=10 (4) — **never r=1**
- SACP+GeoCP: r=2 (16), r=3 (3), r=5 (6), r=10 (4) — **never r=1**
- SACP+GeoCP (bw): bw=10 (6), bw=20 (12), bw=50 (11)

## Method definitions

Four methods compared on each (tile, size):
- **(D) Standard CP**: raw APS scores + global threshold (no spatial processing)
- **(A) SACP default**: r=1 (3×3 Moore) + global threshold — Liu et al. (2024) default
- **(B) SACP-CV-r**: CV-selected radius + global threshold
- **(C) SACP+GeoCP**: CV-selected (r, bw) + per-pixel threshold — **our method**

Generalized SACP score aggregation with Moore radius r:
```
V_k(B_i, y) = (1 − λ) V_{k−1}(B_i, y) + (λ / |N_i|) Σ_{B_j ∈ N_i} V_{k−1}(B_j, y)

N_i = ((2r+1) × (2r+1) Moore neighborhood) ∩ (cal ∪ test)   # excludes train
```
Hyperparameters selected jointly on calibration subsample via 5-fold CV. Test set used once per (tile, size, method).

## Data (per-tile .npz schema)

Each `data/tiles/{tile_key}.npz` contains:
- `emb`: `(H, W, 13)` float — Sentinel-2 L2A Harmonized median composite of 10 bands (B2–B12) + NDVI + NDWI + NBR, May–Sept 2020, cloud-masked
- `label`: `(H, W)` uint8 — ESA WorldCover v100 (2020), 10 m native
- H ≈ W ≈ 1300 px (≈ 13 km tile)

Geographic bboxes (0.12° × 0.12°):

| Tile            | Center (lat, lon) | Biome                    |
| --------------- | ----------------- | ------------------------ |
| polk_iowa       | (41.62, −93.71)   | row-crop agriculture     |
| lancaster_pa    | (40.08, −76.32)   | mixed agriculture        |
| hartford_ct     | (41.78, −72.69)   | urban + forest           |
| everglades_fl   | (25.82, −80.90)   | wetland                  |
| lubbock_tx      | (33.58, −101.86)  | irrigated dryland        |
| sacramento_ca   | (38.16, −121.52)  | river delta              |
| phoenix_az      | (33.52, −112.08)  | desert–urban             |
| yellowstone_wy  | (44.58, −110.54)  | forest                   |
| seattle_wa      | (47.64, −122.32)  | urban + water            |
| mississippi_la  | (29.78, −91.20)   | river delta (water-dom.) |

## Results files

### `results/csv/`

| File                          | Content                                                   |
| ----------------------------- | --------------------------------------------------------- |
| `main_comparison_multisize.csv` | **29 rows**: headline Standard CP / SACP / SACP-CV-r / SACP+GeoCP comparison |
| `radius_sweep_10tiles.csv`    | SACP radius diagnostic sweep (r ∈ {0,1,2,3,5,10}) on 9 tiles at 1 km |
| `prelim_radius_sweep.csv`     | Earlier 2-tile radius sweep (dev only)                    |
| `per_tile_s2.csv`             | Initial 10 km × 10 km pilot on S2                         |
| `per_tile_ae.csv`             | Same pilot on AlphaEarth (for contrast)                   |
| `comparison_ae_vs_s2.csv`     | Side-by-side AE vs S2 at 10 km scale                      |
| `sd_q_per_tile.csv`           | sd(q_j) diagnostic per (tile, feature_set)                |

### `results/figures/`

| File                                                  | Content                             |
| ----------------------------------------------------- | ----------------------------------- |
| `main_comparison_multisize.png`                       | **Headline**: 3-panel bar chart for paper Section 4 |
| `radius_sweep_10tiles.png`                            | SACP IS vs r on 9 tiles             |
| `spatial_sacramento_ca_s{100,200,500}.png`            | Per-tile 5-panel spatial map (GT / sizes / q_j) at 3 sizes |
| `spatial_polk_iowa_s{100,200,500}.png`                | Same for polk_iowa                  |
| `prelim_radius_sweep.png`                             | Earlier 2-tile preliminary (dev only) |

### `results/checkpoints/`

29 JSON checkpoint files, one per (tile, size_px), with raw metric values used for the comparison. Produced by `multiscale_ablation.ipynb` cell 18. Can be used to resume an interrupted run.

### `results/spatial_maps/`

29 `.npz` files, one per (tile, size_px). Each contains:
- `H`, `W`, `label` — patch shape and ESA ground truth
- `test_flat_idx`, `y_test` — test-pixel positions and labels
- `size_D`, `size_A`, `size_B`, `size_C` — per-pixel prediction set sizes for the 4 methods
- `q_C` — per-pixel GeoCP threshold ĥq_j
- `best_r_sacp`, `best_r_geocp`, `best_bw_geocp` — CV-selected hyperparameters

These are sufficient to regenerate any spatial visualization without re-running the experiment.

## Reproducing the experiment

1. Upload `notebooks/multiscale_ablation.ipynb` to Colab
2. Ensure `data/tiles/*.npz` is available at `/content/drive/MyDrive/sentinel2_landcover_pilot_10m/tiles/` (the notebook expects this path; adjust `TILE_DIR` in cell 4 if your Drive path differs)
3. Run cells 2 → 4 → 6 → 8 → 18. Total runtime ~35 min on Colab CPU.
   - Cell 16 (optional): pre-seed checkpoints from a previous run to skip already-completed configurations.

If checkpoints from `results/checkpoints/` are placed in `{WORK_DIR}/multiscale/checkpoints/` before running cell 18, the notebook will skip those (tile, size) pairs and only re-compute missing ones.

## Key paper claims (from this experiment)

1. **SACP+GeoCP beats SACP default by +2.09%** in Interval Score, paired t(28) = 4.82, p < 0.0001, 23/29 wins. This is the paper's main quantitative result.

2. **SACP's default radius (r=1) is suboptimal**: CV selects r ≥ 2 on 29/29 configurations. This is the first systematic ablation of SACP's neighborhood structure.

3. **Effect size grows with scale**: at 5 km, SACP+GeoCP wins **10/10 tiles** with p < 0.0001.

4. **GeoCP's marginal contribution on IS is small** once SACP radius is CV-tuned (C vs B: +0.27%, p = 0.51). The primary value of the GeoCP layer is the per-pixel threshold ĥq_j it produces, a spatially-resolved second-order uncertainty output that SACP cannot emit (see `results/spatial_maps/`).

5. **Coverage guarantee verified**: all 29 configurations × all 4 methods yield marginal coverage ∈ [0.88, 0.92] against the target 0.90.
