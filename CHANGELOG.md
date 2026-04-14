# Changelog

All notable changes to `geocp_rs` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2025-04-14

### Added

- First public release of `geocp_rs`.
- Core conformal-prediction primitives (`aps_scores`, `conformal_quantile`,
  `weighted_quantile`) implemented in pure NumPy.
- `sacp_smooth`: vectorized 8-neighbor spatial smoothing of APS scores
  (Liu et al., 2024).
- `geocp_local_threshold`: per-pixel geographic-kernel quantile with
  log-stable Gaussian weighting (Lou, Luo, Meng, 2024).
- `run_sacp_geocp`: end-to-end Algorithm 1 that composes SACP + GeoCP.
- `interval_score`, `coverage_and_size`: Gneiting–Raftery set interval
  score and coverage/size metrics.
- `geocp_rs.datasets`: loaders + auto-download for 5 standard HSI
  benchmarks (Indian Pines, Pavia University, Salinas, KSC, Botswana).
- `geocp_rs.models.CNN3D`: reference 3D-CNN backbone (Hamida et al., 2018);
  optional PyTorch dependency.
- `geocp_rs.train.train_3dcnn`: single-seed training helper that returns
  CP-ready softmax probabilities, labels, and flat pixel indices.
- `geocp_rs.evaluate.evaluate_all_cp_methods`: runs Standard CP, SACP at
  three λ values, and SACP+GeoCP with 5-fold CV bandwidth selection.
- `geocp_rs.viz.plot_qualitative_grid`: 5×4 classification-result grid
  (GT / coverage / size / local q).
- `scripts/run_all_experiments.py`, `scripts/aggregate_results.py`,
  `scripts/make_figures.py` with corresponding `geocp-rs-*` console entry
  points.
- `examples/quick_start.py`: synthetic-data smoke test, no dataset needed.
- `tests/test_core.py`: 7 unit tests for the CP primitives.
- `docs/INSTALLATION.md`, `USAGE.md`, `ALGORITHM.md`,
  `EXPERIMENT_PROTOCOL.md`, `RESULTS_ANALYSIS.md`.

### Experiments

- Per-seed checkpoints for 50 runs (5 datasets × 10 seeds) committed under
  `results/checkpoints/` (not part of the PyPI distribution, GitHub only).
- `results/summary.json`, `per_seed.csv`, `stats.json`, and
  `results_table.tex` regenerable via `geocp-rs-aggregate`.
- 4 paper figures under `figures/` regenerable via `geocp-rs-figures`.

### Notes

- On the 5 HSI datasets × 10 seeds = 50 runs, SACP+GeoCP achieves the
  lowest mean interval score on all 5 datasets; pooled paired t-test
  against SACP(λ=0.5) gives p = 2.0 × 10⁻⁷, Wilcoxon p = 3.6 × 10⁻⁵.
