"""
SAR data download and synthetic generation module.

Provides convenience wrappers for generating synthetic Sentinel-1 SAR
backscatter from label maps and a stub for real GEE-based Sentinel-1 download.
"""

import numpy as np

from .download_ee import retry_ee
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


@retry_ee(max_retries=5)
def download_sentinel1_patch(
    lat: float,
    lon: float,
    date_start: str,
    date_end: str,
    output_path: str,
    size_px: int = 256,
    scale: int = 10,
) -> np.ndarray | None:
    """Download a Sentinel-1 GRD patch from Google Earth Engine.

    Queries the COPERNICUS/S1_GRD collection for VV and VH polarizations,
    computes the cross-polarization ratio, converts it to a numpy array, and saves it.

    Args:
        lat: Latitude of the area of interest center.
        lon: Longitude of the area of interest center.
        date_start: Start date in ``YYYY-MM-DD`` format.
        date_end: End date in ``YYYY-MM-DD`` format.
        output_path: Path to save the downloaded patch.
        size_px: Target pixel height/width.
        scale: Resolution in meters per pixel.

    Returns:
        Float32 array of shape (3, H, W) containing [VV, VH, VV/VH ratio], or None on failure.
    """
    try:
        import ee

        point = ee.Geometry.Point([lon, lat])
        region = point.buffer(size_px * scale / 2).bounds()

        s1_col = (
            ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(region)
            .filterDate(date_start, date_end)
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
            .filter(ee.Filter.eq("instrumentMode", "IW"))
        )

        if s1_col.size().getInfo() == 0:
            print(f"  [S1 Download] No Sentinel-1 imagery found for lat={lat}, lon={lon}")
            return None

        # Sort by time start to get the latest available image in date range
        s1 = s1_col.sort("system:time_start", False).first()
        s1_img = s1.select(["VV", "VH"])
        # Compute ratio index: VV / (VH + eps)
        ratio = s1_img.select("VV").divide(s1_img.select("VH").add(1e-8)).rename("VV_VH_ratio")
        fused_s1 = s1_img.addBands(ratio)

        import geemap

        sar_np = geemap.ee_to_numpy(fused_s1, region=region, scale=scale)

        if sar_np is not None:
            import cv2

            if sar_np.shape[:2] != (size_px, size_px):
                sar_np = cv2.resize(sar_np, (size_px, size_px), interpolation=cv2.INTER_LINEAR)

            # Map dB backscatter values (typically [-25.0, 0.0]) to [0.0, 1.0] linear scale proxy
            sar_np = np.clip((sar_np + 25.0) / 25.0, 0.0, 1.0)
            sar_np = sar_np.transpose(2, 0, 1).astype(np.float32)

            import os

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            np.save(output_path, sar_np)
            return sar_np

    except Exception as e:
        print(f"  [S1 Download] Real Sentinel-1 download failed for lat={lat}, lon={lon}: {e}")
        print("  Using synthetic SAR fallback.")
    return None
