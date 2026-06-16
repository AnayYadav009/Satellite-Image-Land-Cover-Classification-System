"""
Fusion dataset combining Sentinel-2 optical bands with synthetic Sentinel-1 SAR channels.

Extends the LandCoverDataset interface by concatenating VV, VH, and VV/VH ratio
channels to the optical bands, producing (C+3, H, W) tensors for SAR+optical
data fusion experiments.
"""

from pathlib import Path
import numpy as np
import torch
from .dataset import LandCoverDataset
from .download_sar import generate_sar_for_patch


class FusionDataset(LandCoverDataset):
    """PyTorch Dataset that fuses optical and SAR bands for each patch.

    Inherits data loading, resizing, and normalization logic from LandCoverDataset
    to eliminate code redundancy.
    """

    def __getitem__(self, idx):
        """Load an optical image, generate SAR channels, fuse, and return.

        Args:
            idx: Index of the patch to load.

        Returns:
            Tuple of (fused_tensor, label_tensor):
                fused_tensor: (C+3, H, W) float32 tensor with optical + SAR bands.
                label_tensor: (H, W) int64 tensor with class labels.
        """
        # Call the parent class's __getitem__ to load, resize, transform, and normalize the optical bands
        img_tensor, label_tensor = super().__getitem__(idx)
        label_np = label_tensor.numpy()

        try:
            sar = generate_sar_for_patch(label_np, seed=idx)
        except Exception as e:
            print(f"  Warning: SAR generation failed for patch {idx}: {e}. Using zeros.")
            sar = np.zeros((3, label_np.shape[0], label_np.shape[1]), dtype=np.float32)

        sar_tensor = torch.from_numpy(sar).float()
        fused = torch.cat([img_tensor, sar_tensor], dim=0)

        return fused, label_tensor
