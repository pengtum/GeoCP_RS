"""3D-CNN backbone for HSI pixel classification.

Implements the architecture of Hamida et al. (2018), as used in the original
SACP paper. This is our reference base classifier but *any* PyTorch / NumPy
model that produces a per-class softmax can be plugged into
``geocp_rs.pipeline.run_sacp_geocp`` — SACP+GeoCP is classifier-agnostic.

Reference
---------
Hamida, A.B., Benoit, A., Lambert, P., Ben Amar, C. (2018).
    3-D Deep Learning Approach for Remote Sensing Image Classification.
    IEEE Trans. Geosci. Remote Sens., 56(8), 4420--4434.
"""
from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:

    class CNN3D(nn.Module):
        """3D-CNN for hyperspectral pixel classification.

        Parameters
        ----------
        input_channels : int
            Number of spectral bands.
        n_classes : int
            Number of target classes.
        patch_size : int
            Spatial patch edge length (must be odd). Default 5.
        """

        def __init__(self, input_channels: int, n_classes: int,
                     patch_size: int = 5):
            super().__init__()
            self.conv1 = nn.Conv3d(1, 20, (3, 3, 3), padding=0)
            self.pool1 = nn.Conv3d(20, 20, (3, 1, 1), stride=(2, 1, 1),
                                    padding=(1, 0, 0))
            self.conv2 = nn.Conv3d(20, 35, (3, 3, 3), padding=(1, 0, 0))
            self.pool2 = nn.Conv3d(35, 35, (3, 1, 1), stride=(2, 1, 1),
                                    padding=(1, 0, 0))
            self.conv3 = nn.Conv3d(35, 35, (3, 1, 1), padding=(1, 0, 0))
            self.conv4 = nn.Conv3d(35, 35, (2, 1, 1), stride=(2, 1, 1),
                                    padding=(1, 0, 0))
            self.patch_size = patch_size
            self.input_channels = input_channels
            self.features_size = self._compute_features_size()
            self.fc = nn.Linear(self.features_size, n_classes)

        def _compute_features_size(self) -> int:
            with torch.no_grad():
                x = torch.zeros(
                    (1, 1, self.input_channels, self.patch_size, self.patch_size)
                )
                x = self.pool1(self.conv1(x))
                x = self.pool2(self.conv2(x))
                x = self.conv3(x)
                x = self.conv4(x)
                return int(np.prod(x.size()[1:]))

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            x = F.relu(self.conv1(x)); x = self.pool1(x)
            x = F.relu(self.conv2(x)); x = self.pool2(x)
            x = F.relu(self.conv3(x)); x = F.relu(self.conv4(x))
            x = x.view(-1, self.features_size)
            return self.fc(x)

    def extract_patches(hsi_chw: np.ndarray,
                        indices: np.ndarray,
                        patch_size: int = 2) -> np.ndarray:
        """Extract (2*patch_size+1) spatial patches around the given flat indices.

        Reflection-padded at image borders. Output shape:
            (n, 1, bands, 2*patch_size+1, 2*patch_size+1)
        """
        d, h, w = hsi_chw.shape
        padded = np.pad(
            hsi_chw,
            ((0, 0), (patch_size, patch_size), (patch_size, patch_size)),
            mode="reflect",
        )
        edge = 2 * patch_size + 1
        patches = np.zeros((len(indices), 1, d, edge, edge), dtype=np.float32)
        for e, idx in enumerate(indices):
            r, c = int(idx // w), int(idx % w)
            patches[e, 0] = padded[:, r:r + edge, c:c + edge]
        return patches

else:  # torch not installed

    CNN3D = None  # type: ignore

    def extract_patches(*args, **kwargs):  # type: ignore
        raise ImportError(
            "PyTorch is required for geocp_rs.models.extract_patches. "
            "Install it via `pip install torch`.")
