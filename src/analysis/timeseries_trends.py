"""
Multi-temporal NDVI Trend Analysis Module.

Loads the monthly NDVI time-series array of shape (12, H, W) and calculates
pixel-wise linear regression slopes to identify vegetation growth/degradation trends.
Saves the spatial trend map and reports stats.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def analyze_ndvi_trends(
    series_path="outputs/reports/ndvi_series.npy",
    output_png="outputs/maps/ndvi_trends.png",
    output_json="outputs/reports/ndvi_trends.json",
):
    """Calculate pixel-wise linear regression slope over the 12-month NDVI series.

    Args:
        series_path: Path to the 12-month (12, H, W) NDVI numpy array.
        output_png: Path to save the visual trend heatmap.
        output_json: Path to save trend statistics.
    """
    series_file = Path(series_path)
    if not series_file.exists():
        print(f"[ERROR] Time-series file {series_path} does not exist. Run the pipeline first.")
        return

    # Load series, shape is (12, H, W)
    ndvi_series = np.load(series_path).astype(np.float32)
    T, H, W = ndvi_series.shape

    print(f"[INFO] Analyzing NDVI time-series trends (Dimensions: {T} months, size {H}x{W})...")

    # Time indices (x-values for linear regression: 0 to 11)
    t = np.arange(T, dtype=np.float32)
    t_mean = np.mean(t)
    t_dev = t - t_mean
    t_variance = np.sum(t_dev**2)

    # Vectorized computation of the linear regression slope m:
    # m = Cov(t, ndvi) / Var(t)
    # ndvi_series shape: (12, H, W)
    ndvi_mean = np.mean(ndvi_series, axis=0)  # (H, W)

    # Compute covariance: sum_{t} (t - t_mean) * (ndvi_t - ndvi_mean)
    # Using broadcasting: t_dev shape is (12,), ndvi_series shape is (12, H, W)
    t_dev_expanded = t_dev[:, np.newaxis, np.newaxis]  # (12, 1, 1)
    covariance = np.sum(
        t_dev_expanded * (ndvi_series - ndvi_mean[np.newaxis, :, :]), axis=0
    )  # (H, W)

    slopes = covariance / t_variance  # (H, W)

    # Classify trends
    # Slope > 0.015 -> Improving Vegetation (Greening)
    # Slope < -0.015 -> Degrading Vegetation (Brown-down/Clearing)
    # Otherwise -> Stable
    greening_mask = slopes > 0.015
    degrading_mask = slopes < -0.015
    stable_mask = (~greening_mask) & (~degrading_mask)

    total_pixels = H * W
    pct_greening = float(np.sum(greening_mask) / total_pixels * 100)
    pct_degrading = float(np.sum(degrading_mask) / total_pixels * 100)
    pct_stable = float(np.sum(stable_mask) / total_pixels * 100)

    # Save statistics
    trend_stats = {
        "pct_greening": pct_greening,
        "pct_degrading": pct_degrading,
        "pct_stable": pct_stable,
        "max_slope": float(np.max(slopes)),
        "min_slope": float(np.min(slopes)),
        "mean_slope": float(np.mean(slopes)),
    }

    # Save JSON stats
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w") as f:
        json.dump(trend_stats, f, indent=2)
    print(f"  Saved trend statistics to {output_json}")

    # Plot and save spatial trend map
    Path(output_png).parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 8))
    # Using 'RdYlGn' colormap (Red = degrading, Yellow = stable, Green = greening)
    im = plt.imshow(slopes, cmap="RdYlGn", vmin=-0.05, vmax=0.05)
    plt.colorbar(im, label="NDVI Change Slope (NDVI / Month)", extend="both")
    plt.title("🌿 Spatial NDVI Change Trend Map (12-Month Slope)", pad=15)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.axis("on")
    plt.tight_layout()
    plt.savefig(output_png, dpi=150)
    plt.close()

    print(f"  Saved trend map visualization to {output_png}")
    print(
        f"  Summary: Greening={pct_greening:.2f}%, "
        f"Degrading={pct_degrading:.2f}%, Stable={pct_stable:.2f}%"
    )


if __name__ == "__main__":
    analyze_ndvi_trends()
