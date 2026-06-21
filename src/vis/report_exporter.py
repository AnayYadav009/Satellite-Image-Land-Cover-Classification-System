"""
PDF and Markdown Report Exporter.

Parses evaluation metrics, NDVI stats, and transition matrix outputs,
and generates a PDF report sheet and a Markdown summary.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def generate_reports(
    metrics_path="outputs/reports/metrics.json",
    trans_path="outputs/reports/transition_area_ha.csv",
    ndvi_path="outputs/reports/ndvi_stats.json",
    output_pdf="outputs/reports/classification_report.pdf",
    output_md="outputs/reports/project_summary_report.md",
):
    """Compile results and generate PDF/Markdown reports."""
    metrics_file = Path(metrics_path)
    if not metrics_file.exists():
        print(f"[ERROR] Metrics file {metrics_path} does not exist. Run the pipeline first.")
        return

    # 1. Load Data
    with open(metrics_file, "r") as f:
        metrics = json.load(f)

    trans_data = None
    if Path(trans_path).exists():
        trans_data = pd.read_csv(trans_path, index_col=0)

    ndvi_stats = None
    if Path(ndvi_path).exists():
        with open(ndvi_path, "r") as f:
            ndvi_stats = json.load(f)

    print("[INFO] Compiling report data and generating PDF/Markdown exports...")

    # 2. Write Markdown Report
    md_content = [
        "# Bhopal Land-Cover Classification System - Project Audit Report\n",
        "Generated automatically on completion of evaluation.\n",
        "## 📊 Model Performance Metrics\n",
        f"- **Overall Pixel Accuracy:** {metrics['overall_accuracy'] * 100:.2f}%\n",
        f"- **Mean Intersection-over-Union (mIoU):** {metrics['mean_iou'] * 100:.2f}%\n",
    ]
    if "mean_confidence" in metrics:
        md_content.append(f"- **Mean Model Confidence:** {metrics['mean_confidence'] * 100:.2f}%\n")

    md_content.append("\n### Per-Class Intersection-over-Union (IoU) & F1-Score\n")
    md_content.append("| Class Name | IoU Score | F1 Score |\n")
    md_content.append("|---|---|---|\n")
    for name in metrics["per_class_iou"].keys():
        iou = metrics["per_class_iou"][name] * 100
        f1 = metrics["per_class_f1"].get(name, 0.0) * 100
        md_content.append(f"| {name} | {iou:.2f}% | {f1:.2f}% |\n")

    if ndvi_stats:
        MONTH_NAMES = [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]
        md_content.append("\n## 🌿 Seasonal NDVI Vegetation Health Stats\n")
        md_content.append(f"- **Peak Greenness Month:** {MONTH_NAMES[ndvi_stats['peak_month']]}\n")
        md_content.append(
            f"- **Trough (Lowest) Month:** {MONTH_NAMES[ndvi_stats['trough_month']]}\n"
        )
        md_content.append(
            f"- **Vegetation Anomaly Rate:** {ndvi_stats['pct_anomalous']:.2f}% of pixels\n"
        )

    if trans_data is not None:
        md_content.append("\n## 🔄 Land Cover Class Transitions (Hectares)\n")
        md_content.append("| Class | " + " | ".join(trans_data.columns) + " |\n")
        md_content.append("|---" + "|---" * len(trans_data.columns) + "|\n")
        for idx, row in trans_data.iterrows():
            row_str = " | ".join([f"{val:.2f}" for val in row])
            md_content.append(f"| **{idx}** | {row_str} |\n")

    with open(output_md, "w", encoding="utf-8") as f:
        f.writelines(md_content)
    print(f"  Saved Markdown summary to {output_md}")

    # 3. Create PDF Report using Matplotlib Vector PDF Backend
    # Set up a portrait 8.5x11 page layout (using 8.5x11 inches)
    fig = plt.figure(figsize=(8.5, 11))

    # Title Section
    fig.text(
        0.1,
        0.93,
        "Bhopal Land-Cover Segmentation Report",
        fontsize=18,
        fontweight="bold",
        color="#1e3d59",
    )
    fig.text(
        0.1,
        0.905,
        "Sentinel-2 Multi-spectral Geospatial Intelligence Pipeline",
        fontsize=11,
        color="#17b978",
        fontweight="bold",
    )
    fig.text(0.1, 0.885, "System status: Evaluation complete", fontsize=9, color="grey")
    fig.text(0.1, 0.865, "-" * 90, color="grey", alpha=0.5)

    # Core Stats block
    fig.text(
        0.1, 0.82, "📊 Overall Performance Metrics", fontsize=13, fontweight="bold", color="#1e3d59"
    )
    fig.text(
        0.12, 0.795, f"Overall Accuracy : {metrics['overall_accuracy'] * 100:.2f}%", fontsize=10
    )
    fig.text(0.12, 0.775, f"Mean IoU (mIoU)  : {metrics['mean_iou'] * 100:.2f}%", fontsize=10)
    if "mean_confidence" in metrics:
        fig.text(
            0.12, 0.755, f"Model Confidence : {metrics['mean_confidence'] * 100:.2f}%", fontsize=10
        )

    if ndvi_stats:
        MONTH_NAMES = [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]
        fig.text(
            0.5,
            0.82,
            "🌿 Seasonal NDVI Health Stats",
            fontsize=13,
            fontweight="bold",
            color="#1e3d59",
        )
        fig.text(
            0.52, 0.795, f"Peak Month        : {MONTH_NAMES[ndvi_stats['peak_month']]}", fontsize=10
        )
        fig.text(
            0.52,
            0.775,
            f"Trough Month      : {MONTH_NAMES[ndvi_stats['trough_month']]}",
            fontsize=10,
        )
        fig.text(
            0.52, 0.755, f"Anomaly Area Rate : {ndvi_stats['pct_anomalous']:.2f}%", fontsize=10
        )

    fig.text(0.1, 0.73, "-" * 90, color="grey", alpha=0.5)

    # Add Per-Class IoU Bar Chart as an embedded subplot
    ax = fig.add_axes([0.12, 0.38, 0.75, 0.30])  # [left, bottom, width, height]
    per_class_iou = metrics["per_class_iou"]
    classes = list(per_class_iou.keys())
    scores = [val * 100 for val in per_class_iou.values()]

    # Sort for plotting
    sorted_idx = np.argsort(scores)
    classes_s = [classes[i] for i in sorted_idx]
    scores_s = [scores[i] for i in sorted_idx]

    CLASS_COLORS = [
        "#FF0000",
        "#006400",
        "#FFD700",
        "#7CFC00",
        "#D2B48C",
        "#00CED1",
        "#0000FF",
        "#FFFFFF",
        "#8B4513",
        "#808080",
    ]
    colors_dict = dict(zip(classes, CLASS_COLORS))
    bar_colors = [colors_dict.get(cls, "#808080") for cls in classes_s]

    bars = ax.barh(classes_s, scores_s, color=bar_colors, edgecolor="grey", alpha=0.85)
    for bar in bars:
        w = bar.get_width()
        ax.text(
            w + 1,
            bar.get_y() + bar.get_height() / 2,
            f"{w:.1f}%",
            va="center",
            ha="left",
            fontsize=8,
            fontweight="bold",
        )

    ax.set_title(
        "Per-Class Intersection-over-Union (IoU) Scores",
        fontsize=11,
        fontweight="bold",
        pad=10,
        color="#1e3d59",
    )
    ax.set_xlabel("IoU (%)", fontsize=9)
    ax.set_xlim(0, 115)
    ax.tick_params(axis="both", which="major", labelsize=8)
    ax.grid(axis="x", linestyle="--", alpha=0.5)

    # Add Class Metrics Table at the bottom
    fig.text(
        0.1, 0.31, "📋 Per-Class Performance Table", fontsize=13, fontweight="bold", color="#1e3d59"
    )

    table_data = []
    for name in classes:
        iou_pct = f"{per_class_iou[name] * 100:.2f}%"
        f1_pct = f"{metrics['per_class_f1'].get(name, 0.0) * 100:.2f}%"
        table_data.append([name, iou_pct, f1_pct])

    # Table layout
    col_labels = ["Class Name", "Intersection-over-Union (IoU)", "F1-Score"]
    table_ax = fig.add_axes([0.1, 0.08, 0.8, 0.20])
    table_ax.axis("off")

    table = table_ax.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
        colColours=["#1e3d59", "#1e3d59", "#1e3d59"],
    )
    # Set text colors and styling for headers
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(color="white", weight="bold")
        cell.set_fontsize(8)

    # Save PDF
    Path(output_pdf).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_pdf, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"  Saved PDF report sheet to {output_pdf}")


if __name__ == "__main__":
    generate_reports()
