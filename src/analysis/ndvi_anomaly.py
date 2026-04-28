import matplotlib.pyplot as plt
import numpy as np


# ✅ FEATURE 1 - TIMESERIES: Plot seasonal NDVI trend
def plot_ndvi_curve(
    stats: dict,  # from compute_ndvi_stats
    month_names: list[str],
    output_path: str,
) -> None:
    """Plots a line chart of monthly mean NDVI ± 1 std as a shaded band.
    Marks peak and trough months with vertical dashed lines.
    Saves to output_path as PNG at 150 dpi."""
    try:
        means = np.array(stats["monthly_mean"])
        stds = np.array(stats["monthly_std"])
        x = np.arange(len(month_names))

        plt.figure(figsize=(10, 6))
        plt.plot(x, means, "g-o", label="Mean NDVI", linewidth=2)
        plt.fill_between(
            x, means - stds, means + stds, color="green", alpha=0.2, label="±1 Std Dev"
        )

        # Mark peak and trough
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
