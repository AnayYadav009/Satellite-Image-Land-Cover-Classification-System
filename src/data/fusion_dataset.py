"""
✅ FEATURE 3 - SAR: Fusion dataset that concatenates SAR channels to optical bands.
Extends LandCoverDataset to add synthetic Sentinel-1 (VV, VH, VV/VH ratio)
as extra input channels for SAR + optical data fusion.
"""

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .download_sar import generate_sar_for_patch  # ✅ FIXED: Relative import for robustness


class FusionDataset(Dataset):
    """
    Extends LandCoverDataset to concatenate SAR channels to optical bands.

    For each patch:
        1. Load optical image (C, H, W) and label (H, W) exactly as
           LandCoverDataset does — same normalization, same augmentations
        2. Generate SAR array (3, H, W) using generate_sar_for_patch(label_map)
           with seed=idx for reproducibility
        3. Concatenate: fused = np.concatenate([optical, sar], axis=0)
           Result shape: (C+3, H, W)

    The fused tensor is returned as the image. The model's in_channels must
    be set to NUM_BANDS + 3 when using this dataset.

    Constructor signature matches LandCoverDataset exactly so it can be
    used as a drop-in replacement.
    """

    def __init__(self, image_dir, label_dir, transform=None, stats_path=None):
        """
        Initialize FusionDataset.

        Args:
            image_dir: Path to directory containing .npy image files.
            label_dir: Path to directory containing .npy label files.
            transform: Optional albumentations transform pipeline.
            stats_path: Path to band_stats.npy for optical band normalization.
        """
        self.image_paths = sorted(Path(image_dir).glob("*.npy"))
        self.label_paths = sorted(Path(label_dir).glob("*.npy"))
        self.transform = transform

        # ✅ FEATURE 3 - SAR: Load band statistics for optical-only normalization
        self.means = None
        self.stds = None
        if stats_path and Path(stats_path).exists():
            try:
                stats = np.load(stats_path)  # (2, C) — [means, stds]
                self.means = stats[0].astype(np.float32).reshape(-1, 1, 1)
                self.stds = stats[1].astype(np.float32).reshape(-1, 1, 1)
            except Exception as e:
                print(f"  Warning: Failed to load band stats: {e}. Skipping normalization.")

        assert len(self.image_paths) == len(self.label_paths), (
            f"Mismatch: {len(self.image_paths)} images vs {len(self.label_paths)} labels"
        )

    def __len__(self):
        """Return number of patches in dataset."""
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Load optical image, generate SAR channels, concatenate, and return.

        Args:
            idx: Index of the patch to load.

        Returns:
            Tuple of (fused_tensor, label_tensor):
                fused_tensor: (C+3, H, W) float32 tensor with optical + SAR bands
                label_tensor: (H, W) int64 tensor with class labels
        """
        try:
            img = np.load(self.image_paths[idx]).astype(np.float32)  # (C, H, W)
            label = np.load(self.label_paths[idx]).astype(np.int64)  # (H, W)
        except IndexError:
            raise  # Let IndexError propagate for Python's iteration protocol
        except Exception as e:
            raise RuntimeError(f"Failed to load patch {idx}: {e}") from e

        # ✅ FEATURE 3 - SAR: Handle size mismatches (same as LandCoverDataset)
        if img.shape[1:] != (256, 256):
            img_hwc = img.transpose(1, 2, 0)
            img_hwc = cv2.resize(img_hwc, (256, 256), interpolation=cv2.INTER_LINEAR)
            img = img_hwc.transpose(2, 0, 1)
        if label.shape != (256, 256):
            label = cv2.resize(
                label.astype(np.uint8), (256, 256), interpolation=cv2.INTER_NEAREST
            ).astype(np.int64)

        # ✅ FEATURE 3 - SAR: Apply augmentations to optical + label BEFORE fusion
        if self.transform:
            img_hwc = img.transpose(1, 2, 0)
            augmented = self.transform(image=img_hwc, mask=label)
            img = augmented["image"].transpose(2, 0, 1)
            label = augmented["mask"]

        # ✅ FEATURE 3 - SAR: Normalize optical bands only
        if self.means is not None:
            img = (img - self.means) / (self.stds + 1e-8)

        # ✅ FEATURE 3 - SAR: Generate synthetic SAR from the ORIGINAL label map
        # Use idx as seed for reproducibility across epochs
        try:
            sar = generate_sar_for_patch(label, seed=idx)  # (3, H, W)
        except Exception as e:
            print(f"  Warning: SAR generation failed for patch {idx}: {e}. Using zeros.")
            sar = np.zeros((3, label.shape[0], label.shape[1]), dtype=np.float32)

        # ✅ FEATURE 3 - SAR: Concatenate optical + SAR → (C+3, H, W)
        fused = np.concatenate([img, sar], axis=0)

        label = label.astype(np.int64)
        return torch.from_numpy(fused.copy()), torch.from_numpy(label.copy())
