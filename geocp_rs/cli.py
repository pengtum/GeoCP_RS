"""Command-line entry points exposed by ``pyproject.toml``.

After ``pip install .``, the user can invoke:

    geocp-rs-run-all       -> scripts.run_all_experiments.main()
    geocp-rs-aggregate     -> scripts.aggregate_results.main()
    geocp-rs-figures       -> scripts.make_figures.main()

This thin wrapper defers to the standalone scripts in the top-level
``scripts/`` directory. They are intentionally not part of the installed
package (the CLI lives here only so that ``pyproject.toml`` can register
them as console scripts).
"""
from __future__ import annotations
import importlib.util
import os
import sys


def _load_script(name: str):
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    path = os.path.join(root, "scripts", f"{name}.py")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cannot find script: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def run_all_main():
    return _load_script("run_all_experiments").main()


def aggregate_main():
    return _load_script("aggregate_results").main()


def figures_main():
    return _load_script("make_figures").main()
