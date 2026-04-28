"""
Quick-start synthetic dataset generator.
Creates numpy .npy files (no rasterio/GDAL dependency needed).
Generates spatially-correlated multispectral patches with realistic
land-cover patterns for pipeline validation.
"""

from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

# ── Class schema ──────────────────────────────────────────────────────
CLASS_NAMES = [
    "Urban",
    "Forest",
    "Cropland",
    "Grassland",
    "Bare Soil",
    "Wetlands",
    "Water",
    "Snow",
    "Shrubland",
    "Clouds",
]
NUM_CLASSES = len(CLASS_NAMES)

# Approximate spectral signatures (13 S2 bands, normalised 0-1)
# Rows: classes, Cols: bands B1–B12 (simplified)
SPECTRAL_PROFILES = np.array(
    [
        [0.10, 0.10, 0.12, 0.15, 0.18, 0.20, 0.22, 0.22, 0.20, 0.05, 0.25, 0.23],  # Urban
        [0.03, 0.04, 0.04, 0.03, 0.08, 0.35, 0.40, 0.42, 0.38, 0.02, 0.10, 0.06],  # Forest
        [0.04, 0.05, 0.06, 0.06, 0.12, 0.30, 0.32, 0.30, 0.28, 0.02, 0.15, 0.10],  # Cropland
        [0.04, 0.05, 0.06, 0.05, 0.10, 0.25, 0.28, 0.26, 0.24, 0.02, 0.12, 0.08],  # Grassland
        [0.12, 0.14, 0.18, 0.22, 0.24, 0.22, 0.20, 0.20, 0.18, 0.05, 0.28, 0.30],  # Bare Soil
        [0.04, 0.04, 0.05, 0.04, 0.06, 0.18, 0.20, 0.22, 0.20, 0.03, 0.12, 0.08],  # Wetlands
        [0.06, 0.05, 0.04, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],  # Water
        [0.55, 0.60, 0.65, 0.70, 0.72, 0.70, 0.68, 0.65, 0.60, 0.20, 0.30, 0.25],  # Snow
        [0.05, 0.06, 0.07, 0.08, 0.12, 0.20, 0.22, 0.20, 0.18, 0.03, 0.14, 0.10],  # Shrubland
        [0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.10, 0.20, 0.20],  # Clouds
    ],
    dtype=np.float32,
)


def _generate_label_map(size: int, num_classes: int, rng: np.random.Generator) -> np.ndarray:
    """Create a spatially-correlated label map using smoothed random noise."""
    # Generate one noise field per class, smooth it, take argmax
    fields = np.stack(
        [
            gaussian_filter(rng.standard_normal((size, size)), sigma=size // 10)
            for _ in range(num_classes)
        ]
    )
    # Mild bias toward common Bhopal-region classes to ensure variety
    bias = np.zeros(num_classes)
    bias[1] = 0.2  # Forest
    bias[2] = 0.2  # Cropland
    bias[0] = 0.15  # Urban
    bias[6] = 0.15  # Water
    fields += bias[:, None, None]
    return fields.argmax(axis=0).astype(np.int64)


def _generate_multispectral(
    label_map: np.ndarray, num_bands: int, rng: np.random.Generator
) -> np.ndarray:
    """Render a multispectral image from a label map using spectral profiles."""
    h, w = label_map.shape
    n_profile_bands = SPECTRAL_PROFILES.shape[1]
    img = np.zeros((num_bands, h, w), dtype=np.float32)

    for cls_id in range(NUM_CLASSES):
        mask = label_map == cls_id
        if not mask.any():
            continue
        for b in range(min(num_bands, n_profile_bands)):
            base = SPECTRAL_PROFILES[cls_id, b]
            noise = rng.normal(0, 0.02, size=mask.sum()).astype(np.float32)
            img[b][mask] = base + noise

    # Fill remaining bands (indices NDVI/NDWI/NDBI pre-computed)
    if num_bands > n_profile_bands:
        eps = 1e-8
        b4, b8, b3, b11 = img[3], img[7], img[2], img[10]
        if num_bands > 12:
            img[12] = (b8 - b4) / (b8 + b4 + eps)  # NDVI
        if num_bands > 13:
            img[13] = (b3 - b8) / (b3 + b8 + eps)  # NDWI
        if num_bands > 14:
            img[14] = (b11 - b8) / (b11 + b8 + eps)  # NDBI
        # Any extra bands → zero (padding)

    return np.clip(img, 0.0, 1.0)


def generate_synthetic_patch(
    size: int = 256,
    num_bands: int = 16,  # ✅ FIXED: Change function signature to size: int = 256
    seed: int | None = None,
):
    """Returns (image, label) as numpy arrays."""
    rng = np.random.default_rng(seed)
    label = _generate_label_map(size, NUM_CLASSES, rng)
    image = _generate_multispectral(label, num_bands, rng)
    return image, label


def create_quickstart_dataset(
    output_dir: str = "data/quickstart",
    num_train: int = 50,
    num_val: int = 15,
    num_test: int = 15,
    patch_size: int = 256,
    num_bands: int = 16,
):
    """Creates train/val/test splits saved as .npy files."""
    out = Path(output_dir)

    # ✅ FIX: Explicitly remove old data to prevent mismatch bugs
    if out.exists():
        import shutil

        shutil.rmtree(out, ignore_errors=True)

    splits = {"train": num_train, "val": num_val, "test": num_test}
    total = sum(splits.values())

    # Create directory structure
    for split in splits:
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        (out / split / "labels").mkdir(parents=True, exist_ok=True)

    total_train_val = num_train + num_val
    for i in range(total_train_val):
        split = "train" if i < num_train else "val"

        img, lbl = generate_synthetic_patch(size=patch_size, num_bands=num_bands, seed=i)
        np.save(out / split / "images" / f"patch_{i:04d}.npy", img)
        np.save(out / split / "labels" / f"patch_{i:04d}.npy", lbl)

    # ✅ ADDED: Grid patching for test set to demonstrate stitching
    print("Generating contiguous test grid (256x256)...")
    split = "test"
    grid_size = int(np.ceil(np.sqrt(num_test)))
    full_h, full_w = grid_size * patch_size, grid_size * patch_size
    # Generate a large "scene" and then patch it (seed 42 for repeatable diversity)
    full_img, full_lbl = generate_synthetic_patch(size=full_h, num_bands=num_bands, seed=42)

    idx = 0
    for r in range(grid_size):
        for c in range(grid_size):
            y, x = r * patch_size, c * patch_size
            patch_img = full_img[:, y : y + patch_size, x : x + patch_size]
            patch_lbl = full_lbl[y : y + patch_size, x : x + patch_size]

            np.save(out / split / "images" / f"patch_{idx:04d}.npy", patch_img)
            np.save(out / split / "labels" / f"patch_{idx:04d}.npy", patch_lbl)
            idx += 1

    # Save original scene shape for stitching reference
    np.save(out / split / "scene_shape.npy", np.array([full_h, full_w]))
    print(f"  Generated {total} patches ({patch_size}×{patch_size}, {num_bands} bands)")
    _compute_and_save_stats(out / "train" / "images", out / "band_stats.npy", num_bands)
    print(f"  Dataset saved to {out.resolve()}")


def _compute_and_save_stats(img_dir: Path, stats_path: Path, num_bands: int):
    """Compute per-band mean and std from training images."""
    files = sorted(img_dir.glob("*.npy"))
    running_sum = np.zeros(num_bands, dtype=np.float64)
    running_sq = np.zeros(num_bands, dtype=np.float64)
    n_pixels = 0
    for f in files:
        img = np.load(f)  # (C, H, W)
        running_sum += img.sum(axis=(1, 2))
        running_sq += (img**2).sum(axis=(1, 2))
        n_pixels += img.shape[1] * img.shape[2]
    means = running_sum / n_pixels
    stds = np.sqrt(running_sq / n_pixels - means**2)
    np.save(stats_path, np.stack([means, stds]))
    print(f"  Band stats saved to {stats_path}")


if __name__ == "__main__":
    print("=" * 60)
    print("  Generating Quick-Start Synthetic Dataset")
    print("=" * 60)
    create_quickstart_dataset()
    print("  Done!")
