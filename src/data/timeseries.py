from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .download_quickstart import generate_synthetic_patch  # ✅ FIXED: Relative import


# ✅ FEATURE 1 - TIMESERIES: Synthetic NDVI series generator
def generate_monthly_ndvi_series(
    region_seed: int = 42, n_months: int = 12, patch_size: int = 256, num_bands: int = 16
) -> np.ndarray:
    """Generates a synthetic monthly NDVI time-series of shape (n_months, H, W).

    For each month m (0=Jan, 11=Dec):
      1. Generate a base patch using generate_synthetic_patch(seed=region_seed)
      2. Apply a seasonal NDVI multiplier to simulate crop growth cycles:
           seasonal_factor = 0.5 + 0.5 * sin(2π * (m - 2) / 12)
           (peaks in July/August, lowest in Jan/Feb — Bhopal Kharif cycle)
      3. Compute NDVI from the patch's NIR (band index 7) and Red (band index 3):
           ndvi = (nir - red) / (nir + red + 1e-8)
      4. Multiply agricultural pixels (NDVI > 0.2 in base) by seasonal_factor
      5. Add Gaussian noise: np.random.normal(0, 0.02, shape)

    Returns array of shape (12, 256, 256) with values in [-1, 1]."""
    try:
        ndvi_series = []

        # ✅ FIXED: Generate base patch ONCE to avoid redundant calls in the loop
        base_img, _ = generate_synthetic_patch(
            size=patch_size, num_bands=num_bands, seed=region_seed
        )
        red_base = base_img[3]
        nir_base = base_img[7]
        ndvi_template = (nir_base - red_base) / (nir_base + red_base + 1e-8)
        ag_mask = ndvi_template > 0.2

        # ✅ FIXED WARNING 2: seed RNG from region_seed for reproducible
        #                     monthly noise across pipeline runs
        rng = np.random.default_rng(region_seed)
        for m in range(n_months):
            # 1. Start with a copy of the base NDVI to avoid mutating original
            ndvi = ndvi_template.copy()

            # 2. Seasonal factor
            seasonal_factor = 0.5 + 0.5 * np.sin(2 * np.pi * (m - 2) / 12)

            # 3. Apply to agricultural pixels
            ndvi[ag_mask] *= seasonal_factor

            # 4. Add month-specific noise for realism
            noise = rng.normal(0, 0.02, size=ndvi.shape)
            ndvi = ndvi + noise

            ndvi_series.append(np.clip(ndvi, -1, 1))

        return np.stack(ndvi_series).astype(np.float32)
    except Exception as e:
        print(f"Error in generate_monthly_ndvi_series: {e}")
        raise


# ✅ FEATURE 1 - TIMESERIES: Anomaly detection using z-scores
def detect_ndvi_anomalies(
    ndvi_series: np.ndarray,  # (T, H, W)
) -> np.ndarray:
    """For each pixel, computes z-score deviation from its own mean across T months.
    Returns anomaly_map of shape (H, W) float32.
    Pixels with |z-score| > 2.0 are flagged as anomalous.
    Uses: z = (ndvi[t] - mean_t) / (std_t + 1e-8) for each timestep,
    returns max |z| across all timesteps per pixel."""
    try:
        mean_t = np.mean(ndvi_series, axis=0)
        std_t = np.std(ndvi_series, axis=0)

        # Compute z-scores for all timesteps
        z_scores = np.abs((ndvi_series - mean_t) / (std_t + 1e-8))

        # Max z-score per pixel
        anomaly_map = np.max(z_scores, axis=0)
        return anomaly_map.astype(np.float32)
    except Exception as e:
        print(f"Error in detect_ndvi_anomalies: {e}")
        raise


# ✅ FEATURE 1 - TIMESERIES: Save monthly heatmap frames
def save_ndvi_animation_frames(
    ndvi_series: np.ndarray,  # (T, H, W)
    output_dir: str,
    month_names: list[str],
) -> list[str]:
    """Saves T PNG frames (one per month) as a heatmap using RdYlGn colormap,
    vmin=-0.2, vmax=0.8. Each frame has the month name as title.
    Returns list of saved file paths."""
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


# ✅ FEATURE 1 - TIMESERIES: Compute summary statistics
def compute_ndvi_stats(
    ndvi_series: np.ndarray,  # (T, H, W)
) -> dict:
    """Returns dict with keys:
    'monthly_mean': list of T floats (spatial mean per month)
    'monthly_std':  list of T floats (spatial std per month)
    'peak_month':   int (0-indexed month with highest mean NDVI)
    'trough_month': int (0-indexed month with lowest mean NDVI)
    'pct_anomalous': float (% of pixels flagged as anomalous)"""
    try:
        monthly_means = np.mean(ndvi_series, axis=(1, 2))
        monthly_stds = np.std(ndvi_series, axis=(1, 2))

        peak_month = int(np.argmax(monthly_means))
        trough_month = int(np.argmin(monthly_means))

        # Anomaly calculation
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
