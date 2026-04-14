"""HSI dataset loaders and standardized preprocessing.

Supports the 5 public HSI benchmarks used in the paper: Indian Pines,
Pavia University, Salinas, KSC, Botswana. Each loader returns a tuple

    (hsi_cube, ground_truth_map, n_classes, n_bands)

where ``hsi_cube`` has shape (H, W, B) as float32, and ``ground_truth_map``
is a (H, W) int array with label 0 reserved for unlabeled pixels.

The URLs hit the EHU Computational Intelligence Group mirror; see
``docs/EXPERIMENT_PROTOCOL.md`` for details.
"""
from __future__ import annotations
import os
from dataclasses import dataclass

import numpy as np
import scipy.io as sio


@dataclass
class DatasetSpec:
    key: str
    nice_name: str
    hsi_file: str
    hsi_mat_key: str
    gt_file: str
    gt_mat_key: str
    n_classes: int
    n_bands: int
    url_hsi: str
    url_gt: str


DATASETS: dict[str, DatasetSpec] = {
    "ip": DatasetSpec(
        key="ip", nice_name="Indian Pines",
        hsi_file="Indian_pines_corrected.mat", hsi_mat_key="indian_pines_corrected",
        gt_file="Indian_pines_gt.mat", gt_mat_key="indian_pines_gt",
        n_classes=16, n_bands=200,
        url_hsi="https://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat",
        url_gt="https://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat",
    ),
    "pu": DatasetSpec(
        key="pu", nice_name="Pavia University",
        hsi_file="PaviaU.mat", hsi_mat_key="paviaU",
        gt_file="PaviaU_gt.mat", gt_mat_key="paviaU_gt",
        n_classes=9, n_bands=103,
        url_hsi="https://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat",
        url_gt="https://www.ehu.eus/ccwintco/uploads/5/50/PaviaU_gt.mat",
    ),
    "sa": DatasetSpec(
        key="sa", nice_name="Salinas",
        hsi_file="Salinas_corrected.mat", hsi_mat_key="salinas_corrected",
        gt_file="Salinas_gt.mat", gt_mat_key="salinas_gt",
        n_classes=16, n_bands=204,
        url_hsi="https://www.ehu.eus/ccwintco/uploads/a/a3/Salinas_corrected.mat",
        url_gt="https://www.ehu.eus/ccwintco/uploads/f/fa/Salinas_gt.mat",
    ),
    "ksc": DatasetSpec(
        key="ksc", nice_name="KSC",
        hsi_file="KSC.mat", hsi_mat_key="KSC",
        gt_file="KSC_gt.mat", gt_mat_key="KSC_gt",
        n_classes=13, n_bands=176,
        url_hsi="https://www.ehu.eus/ccwintco/uploads/2/26/KSC.mat",
        url_gt="https://www.ehu.eus/ccwintco/uploads/a/a6/KSC_gt.mat",
    ),
    "botswana": DatasetSpec(
        key="botswana", nice_name="Botswana",
        hsi_file="Botswana.mat", hsi_mat_key="Botswana",
        gt_file="Botswana_gt.mat", gt_mat_key="Botswana_gt",
        n_classes=14, n_bands=145,
        url_hsi="https://www.ehu.eus/ccwintco/uploads/7/72/Botswana.mat",
        url_gt="https://www.ehu.eus/ccwintco/uploads/5/58/Botswana_gt.mat",
    ),
}


def download_dataset(key: str, data_dir: str) -> None:
    """Download the two .mat files for a dataset if they are not already present."""
    spec = DATASETS[key]
    folder = os.path.join(data_dir, key)
    os.makedirs(folder, exist_ok=True)
    for fname, url in [(spec.hsi_file, spec.url_hsi),
                       (spec.gt_file, spec.url_gt)]:
        path = os.path.join(folder, fname)
        if os.path.exists(path) and os.path.getsize(path) > 10000:
            continue
        os.system(f'wget -q -O "{path}" "{url}"')


def load_dataset(key: str,
                 data_dir: str,
                 normalize: bool = True
                 ) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Load a dataset from its .mat files.

    Parameters
    ----------
    key : one of ``geocp_rs.datasets.DATASETS``.
    data_dir : path under which the dataset subfolders live.
    normalize : if True (default), subtract per-band mean and divide by
        global max, matching the protocol used in the paper.

    Returns
    -------
    hsi : (H, W, B) float32
    gt  : (H, W) int
    n_classes : int
    n_bands : int
    """
    spec = DATASETS[key]
    folder = os.path.join(data_dir, key)
    hsi = sio.loadmat(os.path.join(folder, spec.hsi_file))[spec.hsi_mat_key]
    gt = sio.loadmat(os.path.join(folder, spec.gt_file))[spec.gt_mat_key]

    if normalize:
        hsi = hsi.astype(np.float32)
        hsi = (hsi - hsi.mean(axis=(0, 1))) / (hsi.max() + 1e-8)

    return hsi, gt, spec.n_classes, spec.n_bands
