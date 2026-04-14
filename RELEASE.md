# PyPI Release Checklist for `geocp-rs`

This document is the step-by-step procedure for cutting a new release of
`geocp-rs` to PyPI. It replaces one-off scripts with a reproducible recipe.

## Pre-release checklist

Before cutting a release, tick every box:

- [ ] `pyproject.toml` has the intended `version` (bump it **before** building).
- [ ] `CHANGELOG.md` has an entry for the new version with a dated header.
- [ ] `README.md` still reflects the current API surface (update code snippets
      if anything changed in `geocp_rs/__init__.py`).
- [ ] All unit tests pass: `pytest tests/ -v`.
- [ ] The synthetic smoke test passes: `python examples/quick_start.py`.
- [ ] The 4 paper figures re-render without error:
      `geocp-rs-figures --in results --out figures`.
- [ ] `pyproject.toml` author/email fields are set to the real author (not the
      placeholder `your-email@example.com`).
- [ ] You are on the `main` branch and the working tree is clean.

## Step 1 — Clean and build

```bash
cd GeoCP_RS
rm -rf dist/ build/ *.egg-info geocp_rs.egg-info
python -m build
```

You should see two artifacts in `dist/`:

```
dist/
├── geocp_rs-X.Y.Z-py3-none-any.whl    # ~30 KB (no data)
└── geocp_rs-X.Y.Z.tar.gz              # ~30 KB (sdist, MANIFEST.in controlled)
```

**Both should be well under 100 KB.** If the wheel is >1 MB, something from
`results/checkpoints/` or `paper/` leaked in — check `MANIFEST.in`.

## Step 2 — Validate with twine

```bash
python -m twine check dist/*
```

Expected output: two `PASSED` lines. If you see
`InvalidDistribution: Invalid distribution metadata: unrecognized or
malformed field 'license-file'`, your local `setuptools` is ≥ 77 and emits
PEP 639 metadata that older `twine` doesn't accept. The `pyproject.toml` in
this repo pins `setuptools<77` in `[build-system]`, but if you've overridden
that, either **upgrade twine** (`pip install -U twine`) or **reinstall the
pinned setuptools**:

```bash
pip install "setuptools>=61.0,<77"
```

## Step 3 — Smoke-test the built wheel

Always test the wheel in a **fresh, isolated virtual environment**. This
catches metadata bugs, missing files, and accidental dependencies on the
source checkout. Run everything from `/tmp` to avoid the working directory
leaking into `sys.path`:

```bash
# Fresh venv
python3 -m venv /tmp/geocp_rs_release_test
source /tmp/geocp_rs_release_test/bin/activate

# Install the wheel from dist/
pip install --upgrade pip
pip install dist/geocp_rs-X.Y.Z-py3-none-any.whl pytest

# Run tests from outside the repo (avoid cwd pollution)
cd /tmp
python -c "import geocp_rs; print(geocp_rs.__version__, geocp_rs.__file__)"
# Expected: version matches, path is inside /tmp/geocp_rs_release_test/lib/.../site-packages

python -m pytest /path/to/GeoCP_RS/tests/test_core.py -v
# Expected: 7 passed

python /path/to/GeoCP_RS/examples/quick_start.py
# Expected: Coverage ~0.90, IS ~3.5

which geocp-rs-run-all geocp-rs-aggregate geocp-rs-figures
# Expected: all three resolve inside the venv

deactivate
rm -rf /tmp/geocp_rs_release_test
```

**Do not proceed to upload until all four checks pass on the fresh wheel.**

## Step 4 — Upload to TestPyPI (dry run)

Always upload to [TestPyPI](https://test.pypi.org/) first. A broken release
on TestPyPI is recoverable; a broken release on real PyPI is a stain.

```bash
python -m twine upload --repository testpypi dist/*
```

Twine will ask for a username (use `__token__`) and an API token that you
generate at https://test.pypi.org/manage/account/token/.

Once uploaded, test install from TestPyPI in another fresh venv:

```bash
python3 -m venv /tmp/geocp_rs_from_testpypi
/tmp/geocp_rs_from_testpypi/bin/pip install --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    geocp-rs

/tmp/geocp_rs_from_testpypi/bin/python -c "import geocp_rs; print(geocp_rs.__version__)"
```

The `--extra-index-url` is important because TestPyPI does not mirror the
normal dependency graph — numpy/scipy/sklearn still need to come from real
PyPI.

## Step 5 — Upload to real PyPI

**Only after Step 4 succeeds**:

```bash
python -m twine upload dist/*
```

Enter username `__token__` and an API token from
https://pypi.org/manage/account/token/ (scope the token to the `geocp-rs`
project after the first successful upload).

You should see two upload progress bars followed by:

```
View at: https://pypi.org/project/geocp-rs/X.Y.Z/
```

Verify by going to that URL. The page should show README rendered as
markdown, the right version, all three console scripts listed, and an
"Install" button with `pip install geocp-rs`.

## Step 6 — Post-release

- [ ] Tag the release in git: `git tag -a vX.Y.Z -m "geocp-rs X.Y.Z"` and
      `git push origin vX.Y.Z`.
- [ ] Create a GitHub Release referencing the tag, pasting the relevant
      section of `CHANGELOG.md` as the release notes.
- [ ] Bump `pyproject.toml` to the next dev version (e.g. `X.Y.(Z+1).dev0`)
      and add a placeholder section at the top of `CHANGELOG.md`.
- [ ] Test the real install: `pip install geocp-rs` in a brand-new venv,
      run `python examples/quick_start.py` copied from the repo.

## Authentication setup (first time only)

`twine` looks for credentials in the following order:

1. `TWINE_USERNAME` and `TWINE_PASSWORD` environment variables.
2. `~/.pypirc` file.
3. `keyring` backend (macOS Keychain, Windows Credential Locker, libsecret).
4. Interactive prompt.

The recommended setup for a laptop is `~/.pypirc`:

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-AgEIcHlwaS5vcmcCJDxxxxxxxxxxxxxxxxxxxx...   # your scoped token

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-AgENdGVzdC5weXBpLm9yZwIkYxxxxxxxxxxxxxxxxxx...
```

`chmod 600 ~/.pypirc` so the tokens aren't world-readable.

Alternatively, use environment variables in a shell script you don't commit:

```bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-AgEI...
python -m twine upload dist/*
```

## Troubleshooting

**"File already exists"** — PyPI disallows overwriting an uploaded version.
If you need to fix a release, bump the patch version (e.g., `0.1.0 → 0.1.1`)
and upload a new one. The old version can be yanked but not replaced.

**"403 Invalid or non-existent authentication"** — Your API token has
expired or its scope doesn't include this project. Generate a new token
scoped to `geocp-rs` at https://pypi.org/manage/account/token/.

**README renders as plain text on PyPI** — Check that
`readme = {file = "README.md", content-type = "text/markdown"}` is set in
`pyproject.toml`, and that `METADATA` contains
`Description-Content-Type: text/markdown`.

**Wheel is huge (several MB)** — Something from `results/checkpoints/`,
`figures/`, `paper/`, or `notebooks/` is being picked up. Inspect with
`tar tzf dist/geocp_rs-X.Y.Z.tar.gz | sort` and `unzip -l dist/*.whl`.
Fix with stricter `prune` directives in `MANIFEST.in`.

## Current release state

- **Version**: `0.1.0` (from `pyproject.toml`)
- **Name on PyPI**: `geocp-rs` (canonical, hyphenated)
- **Import name**: `geocp_rs` (underscore, per PEP 8)
- **Wheel size**: ~30 KB
- **Sdist size**: ~30 KB
- **Console scripts**: `geocp-rs-run-all`, `geocp-rs-aggregate`, `geocp-rs-figures`
- **Artifacts**: `dist/geocp_rs-0.1.0-py3-none-any.whl`, `dist/geocp_rs-0.1.0.tar.gz`
- **twine check**: PASSED on both.
- **Name availability**: `https://pypi.org/pypi/geocp-rs/json` → 404 (free to claim).
