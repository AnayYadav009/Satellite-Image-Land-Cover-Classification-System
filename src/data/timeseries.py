"""
Synthetic NDVI time-series generation, anomaly detection, and visualization.

Generates monthly NDVI maps from synthetic multispectral patches, detects
anomalous pixels via z-score analysis, saves animation frames, and computes
summary statistics for the seasonal vegetation cycle.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .download_quickstart import generate_synthetic_patch


def generate_monthly_ndvi_series(
    region_seed: int = 42, n_months: int = 12, patch_size: int = 256, num_bands: int = 16
) -> np.ndarray:
    """Generate a synthetic monthly NDVI time-series.

    Produces one NDVI map per month by applying a seasonal growth factor
    (Kharif cycle peaking in Jul/Aug) to agricultural pixels identified
    from the base patch, then adding Gaussian noise for realism.

    Args:
        region_seed: Seed for base patch generation and noise RNG.
        n_months: Number of months to simulate.
        patch_size: Spatial dimension of each NDVI frame (H and W).
        num_bands: Number of spectral bands in the synthetic base patch.

    Returns:
        Float32 array of shape (n_months, patch_size, patch_size) with values clipped to [-1, 1].
    """
    try:
        ndvi_series = []

        base_img, _ = generate_synthetic_patch(
            size=patch_size, num_bands=num_bands, seed=region_seed
        )
        red_base = base_img[3]
        nir_base = base_img[7]
        ndvi_template = (nir_base - red_base) / (nir_base + red_base + 1e-8)
        ag_mask = ndvi_template > 0.2

        rng = np.random.default_rng(region_seed)
        for m in range(n_months):
            ndvi = ndvi_template.copy()
            seasonal_factor = 0.5 + 0.5 * np.sin(2 * np.pi * (m - 2) / 12)
            ndvi[ag_mask] *= seasonal_factor
            noise = rng.normal(0, 0.02, size=ndvi.shape)
            ndvi = ndvi + noise
            ndvi_series.append(np.clip(ndvi, -1, 1))

        return np.stack(ndvi_series).astype(np.float32)
    except Exception as e:
        print(f"Error in generate_monthly_ndvi_series: {e}")
        raise


def detect_ndvi_anomalies(
    ndvi_series: np.ndarray,
) -> np.ndarray:
    """Detect anomalous pixels using temporal z-score analysis.

    For each pixel, computes the z-score deviation from its temporal mean
    across all months and returns the maximum absolute z-score. Pixels
    with values exceeding 2.0 are considered anomalous.

    Args:
        ndvi_series: NDVI time-series of shape (T, H, W).

    Returns:
        Float32 anomaly map of shape (H, W) containing max |z-score| per pixel.
    """
    try:
        mean_t = np.mean(ndvi_series, axis=0)
        std_t = np.std(ndvi_series, axis=0)
        z_scores = np.abs((ndvi_series - mean_t) / (std_t + 1e-8))
        anomaly_map = np.max(z_scores, axis=0)
        return anomaly_map.astype(np.float32)
    except Exception as e:
        print(f"Error in detect_ndvi_anomalies: {e}")
        raise


def save_ndvi_animation_frames(
    ndvi_series: np.ndarray,
    output_dir: str,
    month_names: list[str],
) -> list[str]:
    """Save each month's NDVI map as a heatmap PNG frame.

    Args:
        ndvi_series: NDVI time-series of shape (T, H, W).
        output_dir: Directory to save PNG frames into.
        month_names: List of month name strings for frame titles.

    Returns:
        List of saved file paths.
    """
    try:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        saved_paths = []
        for t in range(ndvi_series.shape[0]):
            plt.figure(figsize=(8, 8))
            plt.imshow(ndvi_series[t], cmap="RdYlGn", vmin=-0.2, vmax=0.8)
            plt.title(f"NDVI - {month_names[t]}")
            plt.axis("off")
            plt.colorbar(label="NDVI")

            filename = f"ndvi_frame_{t:02d}.png"
            filepath = out_path / filename
            plt.savefig(filepath, dpi=100, bbox_inches="tight")
            plt.close()
            saved_paths.append(str(filepath))

        return saved_paths
    except Exception as e:
        print(f"Error in save_ndvi_animation_frames: {e}")
        raise


def compute_ndvi_stats(
    ndvi_series: np.ndarray,
) -> dict:
    """Compute summary statistics from an NDVI time-series.

    Args:
        ndvi_series: NDVI time-series of shape (T, H, W).

    Returns:
        Dict with keys:
            monthly_mean: List of T spatial mean values.
            monthly_std: List of T spatial std values.
            peak_month: 0-indexed month with the highest mean NDVI.
            trough_month: 0-indexed month with the lowest mean NDVI.
            pct_anomalous: Percentage of pixels flagged as anomalous (z > 2.0).
    """
    try:
        monthly_means = np.mean(ndvi_series, axis=(1, 2))
        monthly_stds = np.std(ndvi_series, axis=(1, 2))

        peak_month = int(np.argmax(monthly_means))
        trough_month = int(np.argmin(monthly_means))

        anomaly_map = detect_ndvi_anomalies(ndvi_series)
        pct_anomalous = float(np.mean(anomaly_map > 2.0) * 100)

        return {
            "monthly_mean": [float(m) for m in monthly_means],
            "monthly_std": [float(s) for s in monthly_stds],
            "peak_month": peak_month,
            "trough_month": trough_month,
            "pct_anomalous": pct_anomalous,
        }
    except Exception as e:
        print(f"Error in compute_ndvi_stats: {e}")
        raise
