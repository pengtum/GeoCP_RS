"""3D-CNN training utility for HSI pixel classification.

Thin wrapper over :class:`geocp_rs.models.CNN3D` that runs a full training
loop and returns softmax probabilities on calibration and test patches.
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .models import CNN3D, extract_patches


def get_device() -> "torch.device":
    """Return CUDA if available, else MPS (Apple Silicon), else CPU."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for training.")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def stratified_split(n_labeled: int,
                      y_labeled: np.ndarray,
                      n_train: int = 250,
                      random_state: int = 42
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split labeled indices into train / calib / test stratified by label.

    - ``n_train`` labeled pixels go to the training set (default 250).
    - The remainder is split 50/50 between calibration and test, also
      stratified by the (zero-indexed) label.

    Returns
    -------
    (train_idx, calib_idx, test_idx) : arrays of integer indices into
    the range [0, n_labeled).
    """
    tr, tmp = train_test_split(
        np.arange(n_labeled), train_size=n_train,
        stratify=y_labeled, random_state=random_state)
    ca, te = train_test_split(
        tmp, test_size=0.5,
        stratify=y_labeled[tmp], random_state=random_state)
    return tr, ca, te


def train_3dcnn(hsi: np.ndarray,
                gt: np.ndarray,
                n_classes: int,
                n_bands: int,
                seed: int = 0,
                n_train_labeled: int = 250,
                patch_radius: int = 2,
                epochs: int = 100,
                lr: float = 1e-3,
                batch_size: int = 64,
                device: "torch.device | None" = None) -> dict:
    """Train a 3D-CNN on a hyperspectral cube and return CP-ready outputs.

    Parameters
    ----------
    hsi : (H, W, B) float32 array (already normalized).
    gt : (H, W) int ground-truth map (label 0 = unlabeled).
    n_classes, n_bands : dataset metadata.
    seed : random seed controlling train/calib/test split, weight init, and
        mini-batch shuffling. ``seed * 100 + 42`` is used as the base state.
    n_train_labeled : number of labeled pixels to use for training.
    patch_radius : spatial patch radius; the model sees (2*r+1) x (2*r+1) tiles.
    epochs, lr, batch_size : standard SGD hyperparameters.
    device : torch device. If None, picks via :func:`get_device`.

    Returns
    -------
    dict with keys
        probs_cal, probs_test : (n, K) numpy softmax outputs.
        y_cal, y_test : (n,) int labels.
        cal_flat_idx, test_flat_idx : (n,) flat pixel indices into h*w.
        accuracy : float test-set accuracy.
        h, w : image dimensions.
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for training.")

    device = device or get_device()
    torch.manual_seed(seed * 100 + 42)
    np.random.seed(seed * 100 + 42)

    h, w, _ = hsi.shape
    N = h * w
    hsi_chw = hsi.transpose(2, 0, 1)

    y_all = gt.reshape(N)
    labeled_idx = np.where(y_all > 0)[0]
    y_labeled = y_all[labeled_idx] - 1  # zero-indexed

    idx_tr, idx_ca, idx_te = stratified_split(
        len(labeled_idx), y_labeled, n_train=n_train_labeled,
        random_state=seed * 100 + 42)

    tr_gi = labeled_idx[idx_tr]
    ca_gi = labeled_idx[idx_ca]
    te_gi = labeled_idx[idx_te]
    y_tr = y_labeled[idx_tr]
    y_ca = y_labeled[idx_ca]
    y_te = y_labeled[idx_te]

    X_tr = extract_patches(hsi_chw, tr_gi, patch_size=patch_radius)
    X_ca = extract_patches(hsi_chw, ca_gi, patch_size=patch_radius)
    X_te = extract_patches(hsi_chw, te_gi, patch_size=patch_radius)

    model = CNN3D(n_bands, n_classes, patch_size=2 * patch_radius + 1).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_tr), torch.LongTensor(y_tr)),
        batch_size=batch_size, shuffle=True)

    for _ in range(epochs):
        model.train()
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            opt.zero_grad()
            crit(model(Xb), yb).backward()
            opt.step()

    model.eval()
    def get_probs(X):
        batch = DataLoader(TensorDataset(torch.FloatTensor(X)),
                           batch_size=256, shuffle=False)
        out = []
        with torch.no_grad():
            for (b,) in batch:
                out.append(torch.softmax(model(b.to(device)), dim=1).cpu().numpy())
        return np.concatenate(out)

    probs_ca = get_probs(X_ca)
    probs_te = get_probs(X_te)
    pred_te = np.argmax(probs_te, axis=1)
    acc = float(np.mean(pred_te == y_te))

    return {
        "probs_cal": probs_ca,
        "probs_test": probs_te,
        "y_cal": y_ca,
        "y_test": y_te,
        "cal_flat_idx": ca_gi,
        "test_flat_idx": te_gi,
        "accuracy": acc,
        "h": int(h),
        "w": int(w),
    }
