"""
NDVI seasonal trend visualization.

Plots monthly mean NDVI with standard deviation bands and marks
peak/trough months for vegetation health monitoring.
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_ndvi_curve(
    stats: dict,
    month_names: list[str],
    output_path: str,
) -> None:
    """Plot a seasonal NDVI trend line with standard deviation shading.

    Draws a line chart of monthly mean NDVI values with ±1 std as a
    shaded band, and marks the peak and trough months with vertical
    dashed lines.

    Args:
        stats: Dict from compute_ndvi_stats with keys 'monthly_mean',
               'monthly_std', 'peak_month', and 'trough_month'.
        month_names: List of month name strings for x-axis labels.
        output_path: File path to save the plot as PNG (150 dpi).
    """
    try:
        means = np.array(stats["monthly_mean"])
        stds = np.array(stats["monthly_std"])
        x = np.arange(len(month_names))

        plt.figure(figsize=(10, 6))
        plt.plot(x, means, "g-o", label="Mean NDVI", linewidth=2)
        plt.fill_between(
            x, means - stds, means + stds, color="green", alpha=0.2, label="±1 Std Dev"
        )

        peak = stats["peak_month"]
        trough = stats["trough_month"]

        plt.axvline(
            x=peak, color="blue", linestyle="--", alpha=0.6, label=f"Peak ({month_names[peak]})"
        )
        plt.axvline(
            x=trough,
            color="red",
            linestyle="--",
            alpha=0.6,
            label=f"Trough ({month_names[trough]})",
        )

        plt.xticks(x, month_names)
        plt.xlabel("Month")
        plt.ylabel("NDVI")
        plt.title("Seasonal NDVI Time-Series Monitoring")
        plt.grid(True, alpha=0.3)
        plt.legend()

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
    except Exception as e:
        print(f"Error in plot_ndvi_curve: {e}")
        raise
