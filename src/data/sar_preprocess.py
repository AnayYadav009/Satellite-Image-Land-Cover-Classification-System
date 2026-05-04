"""
SAR preprocessing utilities for Sentinel-1 data.

Generates synthetic SAR backscatter (VV, VH) from label maps, applies
Lee speckle filtering, and computes derived SAR indices for fusion
with optical bands.
"""

import numpy as np
from scipy.ndimage import uniform_filter

SAR_VV_IDX = -2
SAR_VH_IDX = -1

SAR_PROFILES = np.array(
    [
        [0.35, 0.25],
        [0.20, 0.15],
        [0.18, 0.10],
        [0.15, 0.08],
        [0.22, 0.12],
        [0.12, 0.08],
        [0.02, 0.01],
        [0.25, 0.18],
        [0.17, 0.10],
        [0.08, 0.05],
    ],
    dtype=np.float32,
)
"""Approximate C-band backscatter profiles (sigma0, linear scale, normalized to [0,1]).

Rows correspond to land-cover classes: Urban, Forest, Cropland, Grassland,
Bare Soil, Wetlands, Water, Snow, Shrubland, Clouds.
Columns correspond to polarizations: [VV, VH].
"""


def lee_filter(
    image: np.ndarray,
    window_size: int = 3,
) -> np.ndarray:
    """Apply Lee speckle filter to a single-band SAR image.

    Computes local mean and variance within a sliding window, estimates
    global noise variance, and applies an adaptive weighted average that
    preserves edges while reducing speckle noise.

    Args:
        image: Single-band SAR image of shape (H, W), float32.
        window_size: Size of the local window for filtering.

    Returns:
        Filtered array of same shape (H, W), float32.
    """
    try:
        image = image.astype(np.float64)
        local_mean = uniform_filter(image, size=window_size)
        local_sq_mean = uniform_filter(image**2, size=window_size)
        local_var = np.maximum(local_sq_mean - local_mean**2, 0.0)

        noise_var = np.mean(local_var)
        if noise_var < 1e-10:
            return image.astype(np.float32)

        weight = local_var / (local_var + noise_var)
        filtered = local_mean + weight * (image - local_mean)

        return filtered.astype(np.float32)
    except Exception as e:
        print(f"  Warning: Lee filter failed: {e}. Returning unfiltered image.")
        return image.astype(np.float32)


def apply_lee_filter_multichannel(
    sar_image: np.ndarray,
    window_size: int = 3,
) -> np.ndarray:
    """Apply Lee speckle filter independently to each SAR channel.

    Args:
        sar_image: SAR array of shape (2, H, W) with [VV, VH] bands.
        window_size: Size of the local window for Lee filtering.

    Returns:
        Filtered array of shape (2, H, W).
    """
    try:
        result = np.zeros_like(sar_image)
        for i in range(sar_image.shape[0]):
            result[i] = lee_filter(sar_image[i], window_size)
        return result
    except Exception as e:
        print(f"  Warning: Multichannel Lee filter failed: {e}. Returning unfiltered.")
        return sar_image


def generate_synthetic_sar(
    label_map: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate synthetic Sentinel-1 SAR backscatter from a label map.

    Assigns approximate C-band backscatter values per land-cover class,
    adds per-pixel Gaussian noise, and applies Lee speckle filtering.

    Args:
        label_map: Integer label map of shape (H, W).
        rng: NumPy random generator for reproducibility.

    Returns:
        Float32 array of shape (2, H, W) with values in [0, 1].
        Channel 0 = VV polarization, Channel 1 = VH polarization.
    """
    try:
        h, w = label_map.shape
        sar = np.zeros((2, h, w), dtype=np.float32)

        for cls_id in range(SAR_PROFILES.shape[0]):
            mask = label_map == cls_id
            if not mask.any():
                continue
            for band_idx in range(2):
                base_val = SAR_PROFILES[cls_id, band_idx]
                sar[band_idx][mask] = base_val

        noise = rng.normal(0, 0.02, size=(2, h, w)).astype(np.float32)
        sar += noise

        sar = apply_lee_filter_multichannel(sar, window_size=3)
        sar = np.clip(sar, 0.0, 1.0)

        return sar
    except Exception as e:
        print(f"  Warning: Synthetic SAR generation failed: {e}. Returning zeros.")
        return np.zeros((2, label_map.shape[0], label_map.shape[1]), dtype=np.float32)


def compute_sar_indices(
    sar_image: np.ndarray,
) -> np.ndarray:
    """Compute the VV/VH cross-polarization ratio index.

    Calculates ``VV / (VH + 1e-8)`` and normalizes to [0, 1] by
    dividing by 10. High values indicate urban double-bounce or bare soil.

    Args:
        sar_image: SAR array of shape (2, H, W) with [VV, VH] bands.

    Returns:
        Float32 array of shape (1, H, W) containing the normalized VV/VH ratio.
    """
    try:
        vv = sar_image[0]
        vh = sar_image[1]
        ratio = vv / (vh + 1e-8)
        ratio = np.clip(ratio / 10.0, 0.0, 1.0)
        return ratio[np.newaxis, :, :]
    except Exception as e:
        print(f"  Warning: SAR index computation failed: {e}. Returning zeros.")
        return np.zeros((1, sar_image.shape[1], sar_image.shape[2]), dtype=np.float32)
