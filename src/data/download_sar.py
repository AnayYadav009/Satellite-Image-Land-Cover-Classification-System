"""
SAR data download and synthetic generation module.

Provides convenience wrappers for generating synthetic Sentinel-1 SAR
backscatter from label maps and a stub for real GEE-based Sentinel-1 download.
"""

import numpy as np

from .sar_preprocess import (
    compute_sar_indices,
    generate_synthetic_sar,
)


def generate_sar_for_patch(
    label_map: np.ndarray,
    seed: int | None = None,
) -> np.ndarray:
    """Generate a 3-channel SAR array from a label map.

    Uses ``generate_synthetic_sar`` for VV/VH backscatter generation and
    ``compute_sar_indices`` for the cross-polarization ratio. This is the
    primary entry point called by FusionDataset.

    Args:
        label_map: Integer label map of shape (H, W).
        seed: Optional random seed for reproducibility.

    Returns:
        Float32 array of shape (3, H, W) with channels [VV, VH, VV/VH ratio],
        all values clipped to [0, 1].
    """
    try:
        rng = np.random.default_rng(seed)
        sar_vv_vh = generate_synthetic_sar(label_map, rng)
        vv_vh_ratio = compute_sar_indices(sar_vv_vh)
        fused = np.concatenate([sar_vv_vh, vv_vh_ratio], axis=0)
        return fused.astype(np.float32)
    except Exception as e:
        print(f"  Warning: SAR patch generation failed: {e}. Returning zeros.")
        h, w = label_map.shape
        return np.zeros((3, h, w), dtype=np.float32)


def download_sentinel1_patch(
    lat: float,
    lon: float,
    date_start: str,
    date_end: str,
    output_path: str,
) -> np.ndarray | None:
    """Download a Sentinel-1 GRD patch from Google Earth Engine.

    **Stub** — real implementation requires GEE authentication. Would
    download VV+VH for the given AOI and date range, apply terrain
    correction, and return a (2, H, W) float32 array.

    Args:
        lat: Latitude of the area of interest center.
        lon: Longitude of the area of interest center.
        date_start: Start date in ``YYYY-MM-DD`` format.
        date_end: End date in ``YYYY-MM-DD`` format.
        output_path: Path to save the downloaded patch.

    Returns:
        None (stub implementation).
    """
    print("WARNING: Real SAR download requires GEE auth. Using synthetic SAR.")
    return None
