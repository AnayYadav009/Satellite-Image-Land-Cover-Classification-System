"""
PyTorch Dataset for land-cover segmentation.

Loads multispectral .npy patches and corresponding label maps, applies
optional augmentations and per-band normalization, and returns tensors
ready for model training.
"""

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class LandCoverDataset(Dataset):
    """Dataset for loading multispectral image patches and label maps from .npy files.

    Handles automatic resizing to 256×256, optional albumentations-based
    augmentations, and per-band z-score normalization using precomputed
    statistics.
    """

    def __init__(self, image_dir, label_dir, transform=None, stats_path=None):
        """Initialize LandCoverDataset.

        Args:
            image_dir: Path to directory containing .npy image files of shape (C, H, W).
            label_dir: Path to directory containing .npy label files of shape (H, W).
            transform: Optional albumentations transform pipeline.
            stats_path: Path to band_stats.npy containing per-band [means, stds].
        """
        self.image_paths = sorted(Path(image_dir).glob("*.npy"))
        self.label_paths = sorted(Path(label_dir).glob("*.npy"))
        self.transform = transform

        self.means = None
        self.stds = None
        if stats_path and Path(stats_path).exists():
            stats = np.load(stats_path)
            self.means = stats[0].astype(np.float32).reshape(-1, 1, 1)
            self.stds = stats[1].astype(np.float32).reshape(-1, 1, 1)

        assert len(self.image_paths) == len(
            self.label_paths
        ), f"Mismatch: {len(self.image_paths)} images vs {len(self.label_paths)} labels"

    def __len__(self):
        """Return the number of patches in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """Load a patch and its label, apply transforms and normalization.

        Args:
            idx: Index of the patch to load.

        Returns:
            Tuple of (image_tensor, label_tensor):
                image_tensor: (C, H, W) float32 tensor.
                label_tensor: (H, W) int64 tensor.
        """
        img = np.load(self.image_paths[idx]).astype(np.float32)
        label = np.load(self.label_paths[idx]).astype(np.int64)

        if img.shape[1:] != (256, 256):
            img_hwc = img.transpose(1, 2, 0)
            img_hwc = cv2.resize(img_hwc, (256, 256), interpolation=cv2.INTER_LINEAR)
            img = img_hwc.transpose(2, 0, 1)
        if label.shape != (256, 256):
            label = cv2.resize(
                label.astype(np.uint8), (256, 256), interpolation=cv2.INTER_NEAREST
            ).astype(np.int64)

        if self.transform:
            img_hwc = img.transpose(1, 2, 0)
            augmented = self.transform(image=img_hwc, mask=label)
            img = augmented["image"].transpose(2, 0, 1)
            label = augmented["mask"]

        if self.means is not None:
            img = (img - self.means) / (self.stds + 1e-8)

        label = label.astype(np.int64)
        return torch.from_numpy(img.copy()), torch.from_numpy(label.copy())
