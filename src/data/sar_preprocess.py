"""
✅ FEATURE 3 - SAR: SAR preprocessing utilities for Sentinel-1 data.
Generates synthetic SAR backscatter (VV, VH) from label maps,
applies Lee speckle filtering, and computes derived SAR indices.
"""

import numpy as np
from scipy.ndimage import uniform_filter  # ✅ FEATURE 3 - SAR: efficient windowed statistics

# ✅ FEATURE 3 - SAR: SAR band indices — added to NUM_BANDS when fusion is active
SAR_VV_IDX = -2  # second to last channel
SAR_VH_IDX = -1  # last channel

# ✅ FEATURE 3 - SAR: Approximate C-band backscatter profiles (sigma0 in linear scale)
# Normalized to [0,1] by dividing by 0.5
# Rows: [Urban, Forest, Cropland, Grassland, BareSoil, Wetlands, Water, Snow, Shrubland, Clouds]
# Cols: [VV, VH]
SAR_PROFILES = np.array(
    [
        [0.35, 0.25],  # Urban         — high double-bounce
        [0.20, 0.15],  # Forest        — volume scattering
        [0.18, 0.10],  # Cropland      — varies with growth stage
        [0.15, 0.08],  # Grassland     — low, smooth
        [0.22, 0.12],  # Bare Soil     — surface roughness
        [0.12, 0.08],  # Wetlands      — specular + vegetation mix
        [0.02, 0.01],  # Water         — specular, very low return
        [0.25, 0.18],  # Snow          — volume scattering
        [0.17, 0.10],  # Shrubland     — mixed
        [0.08, 0.05],  # Clouds        — minimal (radar sees through clouds)
    ],
    dtype=np.float32,
)


def lee_filter(
    image: np.ndarray,  # (H, W) single band float32
    window_size: int = 3,
) -> np.ndarray:
    """
    Lee speckle filter for SAR imagery.

    For each pixel window:
        mean = local mean in window
        var  = local variance in window
        noise_var = mean(var) across image (estimated noise variance)
        weight = var / (var + noise_var)
        filtered = mean + weight * (pixel - mean)

    Uses scipy.ndimage.uniform_filter for efficient windowed mean/variance.

    Args:
        image: Single-band SAR image of shape (H, W), float32.
        window_size: Size of the local window for filtering (default 3).

    Returns:
        Filtered array of same shape (H, W).
    """
    try:
        image = image.astype(np.float64)  # ✅ FEATURE 3 - SAR: use float64 for numerical stability
        # Local mean
        local_mean = uniform_filter(image, size=window_size)
        # Local variance: E[x^2] - E[x]^2
        local_sq_mean = uniform_filter(image**2, size=window_size)
        local_var = local_sq_mean - local_mean**2
        local_var = np.maximum(
            local_var, 0.0
        )  # ✅ FEATURE 3 - SAR: clamp negative variance from float errors

        # Estimated noise variance (global average of local variance)
        noise_var = np.mean(local_var)
        if noise_var < 1e-10:
            return image.astype(np.float32)  # ✅ FEATURE 3 - SAR: no filtering if no noise

        # Lee filter weight
        weight = local_var / (local_var + noise_var)
        filtered = local_mean + weight * (image - local_mean)

        return filtered.astype(np.float32)
    except Exception as e:
        print(f"  Warning: Lee filter failed: {e}. Returning unfiltered image.")
        return image.astype(np.float32)


def apply_lee_filter_multichannel(
    sar_image: np.ndarray,  # (2, H, W)
    window_size: int = 3,
) -> np.ndarray:
    """
    Applies lee_filter to each SAR channel independently.

    Args:
        sar_image: SAR array of shape (2, H, W) with [VV, VH] bands.
        window_size: Size of the local window for Lee filtering.

    Returns:
        Filtered array of shape (2, H, W).
    """
    try:
        result = np.zeros_like(sar_image)
        for i in range(sar_image.shape[0]):
            result[i] = lee_filter(
                sar_image[i], window_size
            )  # ✅ FEATURE 3 - SAR: filter each band independently
        return result
    except Exception as e:
        print(f"  Warning: Multichannel Lee filter failed: {e}. Returning unfiltered.")
        return sar_image


def generate_synthetic_sar(
    label_map: np.ndarray,  # (H, W) int, same labels as optical
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generates synthetic Sentinel-1 SAR backscatter (VV, VH) from a label map.

    Uses approximate C-band backscatter profiles per land-cover class,
    adds per-pixel Gaussian noise, and applies Lee speckle filtering.

    Args:
        label_map: Integer label map of shape (H, W).
        rng: NumPy random generator for reproducibility.

    Returns:
        Array of shape (2, H, W) with values in [0, 1].
        Channel 0 = VV polarization, Channel 1 = VH polarization.
    """
    try:
        h, w = label_map.shape
        sar = np.zeros((2, h, w), dtype=np.float32)

        # ✅ FEATURE 3 - SAR: Assign backscatter values based on class profiles
        for cls_id in range(SAR_PROFILES.shape[0]):
            mask = label_map == cls_id
            if not mask.any():
                continue
            for band_idx in range(2):
                base_val = SAR_PROFILES[cls_id, band_idx]
                sar[band_idx][mask] = base_val

        # ✅ FEATURE 3 - SAR: Add per-pixel Gaussian noise (sigma = 0.02)
        noise = rng.normal(0, 0.02, size=(2, h, w)).astype(np.float32)
        sar += noise

        # ✅ FEATURE 3 - SAR: Apply Lee speckle filter
        sar = apply_lee_filter_multichannel(sar, window_size=3)

        # ✅ FEATURE 3 - SAR: Clip to valid range
        sar = np.clip(sar, 0.0, 1.0)

        return sar
    except Exception as e:
        print(f"  Warning: Synthetic SAR generation failed: {e}. Returning zeros.")
        return np.zeros((2, label_map.shape[0], label_map.shape[1]), dtype=np.float32)


def compute_sar_indices(
    sar_image: np.ndarray,  # (2, H, W) — [VV, VH]
) -> np.ndarray:
    """
    Computes derived SAR index as an extra channel.

    VV/VH cross-ratio = VV / (VH + 1e-8), normalized to [0, 1] by dividing by 10.
    High values indicate urban double-bounce or bare soil.

    Args:
        sar_image: SAR array of shape (2, H, W) with [VV, VH] bands.

    Returns:
        Array of shape (1, H, W) containing the normalized VV/VH ratio.
    """
    try:
        vv = sar_image[0]  # (H, W)
        vh = sar_image[1]  # (H, W)
        # ✅ FEATURE 3 - SAR: Cross-polarization ratio, normalized
        ratio = vv / (vh + 1e-8)
        ratio = np.clip(ratio / 10.0, 0.0, 1.0)  # normalize to [0, 1]
        return ratio[np.newaxis, :, :]  # (1, H, W)
    except Exception as e:
        print(f"  Warning: SAR index computation failed: {e}. Returning zeros.")
        return np.zeros((1, sar_image.shape[1], sar_image.shape[2]), dtype=np.float32)
