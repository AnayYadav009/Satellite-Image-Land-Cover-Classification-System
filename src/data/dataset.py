"""
PyTorch Dataset for land-cover segmentation.
Loads .npy patches (no rasterio dependency needed).
"""

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class LandCoverDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None, stats_path=None):
        self.image_paths = sorted(Path(image_dir).glob("*.npy"))
        self.label_paths = sorted(Path(label_dir).glob("*.npy"))
        self.transform = transform

        # Load band statistics for normalisation
        self.means = None
        self.stds = None
        if stats_path and Path(stats_path).exists():
            stats = np.load(stats_path)  # (2, C) — [means, stds]
            self.means = stats[0].astype(np.float32).reshape(-1, 1, 1)
            self.stds = stats[1].astype(np.float32).reshape(-1, 1, 1)

        assert len(self.image_paths) == len(self.label_paths), (
            f"Mismatch: {len(self.image_paths)} images vs {len(self.label_paths)} labels"
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = np.load(self.image_paths[idx]).astype(np.float32)  # (C, H, W)
        label = np.load(self.label_paths[idx]).astype(np.int64)  # (H, W)

        # Handle size mismatches (common with GEE exports)
        # resize expects (H, W, C) or (H, W)
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

        # Normalise
        if self.means is not None:
            img = (img - self.means) / (self.stds + 1e-8)

        label = label.astype(np.int64)  # ensure Long for CrossEntropyLoss
        return torch.from_numpy(img.copy()), torch.from_numpy(label.copy())
