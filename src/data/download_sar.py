"""
✅ FEATURE 3 - SAR: SAR data download / generation module.
Provides convenience wrappers for generating synthetic SAR from label maps,
and a stub for real Sentinel-1 GEE download.
"""

import numpy as np

from .sar_preprocess import (  # ✅ FIXED: Relative import for robustness
    compute_sar_indices,
    generate_synthetic_sar,
)


def generate_sar_for_patch(
    label_map: np.ndarray,
    seed: int | None = None,
) -> np.ndarray:
    """
    Convenience wrapper: given a label map, returns a (3, H, W) SAR array
    with channels [VV, VH, VV_VH_ratio].

    Uses generate_synthetic_sar for VV/VH generation and compute_sar_indices
    for the cross-polarization ratio. This is the function called by the
    fusion dataset.

    Args:
        label_map: Integer label map of shape (H, W).
        seed: Optional random seed for reproducibility.

    Returns:
        Array of shape (3, H, W) with channels [VV, VH, VV_VH_ratio],
        all values clipped to [0, 1].
    """
    try:
        rng = np.random.default_rng(seed)
        # ✅ FEATURE 3 - SAR: Generate VV + VH bands
        sar_vv_vh = generate_synthetic_sar(label_map, rng)  # (2, H, W)
        # ✅ FEATURE 3 - SAR: Compute derived VV/VH ratio index
        vv_vh_ratio = compute_sar_indices(sar_vv_vh)  # (1, H, W)
        # ✅ FEATURE 3 - SAR: Concatenate to (3, H, W): [VV, VH, ratio]
        fused = np.concatenate([sar_vv_vh, vv_vh_ratio], axis=0)
        return fused.astype(np.float32)
    except Exception as e:
        print(f"  Warning: SAR patch generation failed: {e}. Returning zeros.")
        h, w = label_map.shape
        return np.zeros((3, h, w), dtype=np.float32)


# ── Stub for real GEE download (not implemented, documents the interface) ──


def download_sentinel1_patch(
    lat: float,
    lon: float,
    date_start: str,  # "YYYY-MM-DD"
    date_end: str,
    output_path: str,
) -> np.ndarray | None:
    """
    STUB — real implementation requires GEE authentication.

    Would download Sentinel-1 GRD IW VV+VH for the given AOI and date range,
    apply terrain correction, and return (2, H, W) float32 array.
    Currently prints a warning and returns None.

    Args:
        lat: Latitude of the area of interest center.
        lon: Longitude of the area of interest center.
        date_start: Start date in "YYYY-MM-DD" format.
        date_end: End date in "YYYY-MM-DD" format.
        output_path: Path to save the downloaded patch.

    Returns:
        None (stub implementation).
    """
    print("WARNING: Real SAR download requires GEE auth. Using synthetic SAR.")
    return None
