"""
Fusion dataset combining Sentinel-2 optical bands with synthetic Sentinel-1 SAR channels.

Extends the LandCoverDataset interface by concatenating VV, VH, and VV/VH ratio
channels to the optical bands, producing (C+3, H, W) tensors for SAR+optical
data fusion experiments.
"""

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .download_sar import generate_sar_for_patch


class FusionDataset(Dataset):
    """PyTorch Dataset that fuses optical and SAR bands for each patch.

    For each patch:
        1. Loads optical image (C, H, W) and label (H, W) from .npy files.
        2. Generates synthetic SAR array (3, H, W) from the label map.
        3. Concatenates to produce fused output of shape (C+3, H, W).

    The model's ``in_channels`` must be set to ``NUM_BANDS + 3`` when using
    this dataset. Constructor signature matches LandCoverDataset for
    drop-in compatibility.
    """

    def __init__(self, image_dir, label_dir, transform=None, stats_path=None):
        """Initialize FusionDataset.

        Args:
            image_dir: Path to directory containing .npy image files.
            label_dir: Path to directory containing .npy label files.
            transform: Optional albumentations transform pipeline.
            stats_path: Path to band_stats.npy for optical band normalization.
        """
        self.image_paths = sorted(Path(image_dir).glob("*.npy"))
        self.label_paths = sorted(Path(label_dir).glob("*.npy"))
        self.transform = transform

        self.means = None
        self.stds = None
        if stats_path and Path(stats_path).exists():
            try:
                stats = np.load(stats_path)
                self.means = stats[0].astype(np.float32).reshape(-1, 1, 1)
                self.stds = stats[1].astype(np.float32).reshape(-1, 1, 1)
            except Exception as e:
                print(f"  Warning: Failed to load band stats: {e}. Skipping normalization.")

        assert len(self.image_paths) == len(
            self.label_paths
        ), f"Mismatch: {len(self.image_paths)} images vs {len(self.label_paths)} labels"

    def __len__(self):
        """Return the number of patches in the dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """Load an optical image, generate SAR channels, fuse, and return.

        Args:
            idx: Index of the patch to load.

        Returns:
            Tuple of (fused_tensor, label_tensor):
                fused_tensor: (C+3, H, W) float32 tensor with optical + SAR bands.
                label_tensor: (H, W) int64 tensor with class labels.
        """
        try:
            img = np.load(self.image_paths[idx]).astype(np.float32)
            label = np.load(self.label_paths[idx]).astype(np.int64)
        except IndexError:
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to load patch {idx}: {e}") from e

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

        try:
            sar = generate_sar_for_patch(label, seed=idx)
        except Exception as e:
            print(f"  Warning: SAR generation failed for patch {idx}: {e}. Using zeros.")
            sar = np.zeros((3, label.shape[0], label.shape[1]), dtype=np.float32)

        fused = np.concatenate([img, sar], axis=0)

        label = label.astype(np.int64)
        return torch.from_numpy(fused.copy()), torch.from_numpy(label.copy())
