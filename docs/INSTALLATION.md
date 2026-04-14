# Installation

## TL;DR

```bash
git clone https://github.com/pengtum/GeoCP_RS.git
cd GeoCP_RS
pip install -e '.[all]'
```

The `[all]` extras pulls in PyTorch (for the 3D-CNN training), matplotlib / pandas (for figure generation), and tqdm. After installation, run `python examples/quick_start.py` to verify — you should see a 4-line report with coverage ≈ 0.9.

## Install variants

`geocp_rs` is designed so that the core CP algorithm (`aps_scores`, `sacp_smooth`, `geocp_local_threshold`, `run_sacp_geocp`, `interval_score`, `coverage_and_size`) runs on **numpy + scipy + scikit-learn only** — no torch, no matplotlib. Everything else is optional.

| Variant | Install command | Gets you |
|---|---|---|
| **Core** | `pip install -e .` | CP primitives and the end-to-end `run_sacp_geocp` pipeline. Enough to wrap any external classifier. |
| **Core + Torch** | `pip install -e '.[torch]'` | Adds `geocp_rs.train` (3D-CNN training loop) and `geocp_rs.models.CNN3D`. |
| **Core + viz** | `pip install -e '.[viz]'` | Adds matplotlib + pandas so that `scripts/make_figures.py` and `geocp_rs.viz` work. |
| **All** *(recommended)* | `pip install -e '.[all]'` | Everything above plus `tqdm`. Enough to reproduce every paper number from scratch. |
| **Dev** | `pip install -e '.[dev]'` | Adds `pytest` and `ruff`. |

## Python version

`geocp_rs` targets Python ≥ 3.9 and is CI-tested on 3.9, 3.10, 3.11. 3.12 works. We use modern type hints (`X | None`, `tuple[int, int]`) so older versions will not import.

## Platform notes

### Linux / macOS (Intel)

The `[all]` extras installs PyTorch from PyPI. CPU inference is fine; for GPU training you want a CUDA-enabled PyTorch build. See [pytorch.org](https://pytorch.org/get-started/locally/) for the exact command for your CUDA version.

### macOS (Apple Silicon, M1/M2/M3)

PyTorch MPS is supported out of the box (`torch>=2.0`). However, training KSC or Botswana from scratch on MPS can hit unified-memory limits on 16 GB machines. In that case, either:

- Reduce `n_train_labeled` (default 250) in `scripts/run_all_experiments.py`, or
- Run only IP / PU / SA locally and use Colab for KSC / Botswana.

The precomputed checkpoints under `results/checkpoints/` already cover all 5 datasets × 10 seeds, so **you can skip the local training entirely and reproduce every paper number directly from the CSV / JSON files**.

### Google Colab

The notebook at `notebooks/sacp_geocp_colab.ipynb` is fully self-contained: it mounts Drive, installs any missing deps with `pip`, downloads the datasets, trains 50 models (≈ 1.5 hours on T4), and writes everything back to Drive with atomic per-seed checkpointing. Safe to interrupt and resume.

## Verifying your install

```bash
# 1. Unit tests
pytest tests/ -v
# Expected: 7 passed

# 2. Synthetic quick-start
python examples/quick_start.py
# Expected:
# Quick start OK
#   Coverage : ~0.90
#   Mean size: ~1.5
#   IS       : ~3.5
#   Bandwidth: 10.0

# 3. Import check
python -c "import geocp_rs; print(geocp_rs.__version__)"
# Expected: 0.1.0
```

## Common issues

**`ModuleNotFoundError: No module named 'geocp_rs'`**
You forgot to run `pip install -e .` after cloning. If you're on a shared Python install without `-e` rights, use `pip install --user -e .` or a fresh virtual environment.

**`ImportError: PyTorch is required for training`**
The core algorithm works without PyTorch, but `geocp_rs.train.train_3dcnn` and `geocp_rs.models.CNN3D` need it. Install with `pip install -e '.[torch]'`.

**`FileNotFoundError: .../data/ip/Indian_pines_corrected.mat`**
Pass `--download` to `geocp-rs-run-all` on the first invocation, or pre-download the `.mat` files into `./data/{key}/` yourself (URLs in `geocp_rs/datasets.py::DATASETS`).

**Figures look slightly different from the paper PDFs.**
Matplotlib version drift changes font rendering slightly. The underlying *numbers* are deterministic given `results/per_seed.csv`; only the rasterization can differ.
