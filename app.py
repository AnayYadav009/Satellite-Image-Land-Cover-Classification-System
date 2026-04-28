import json
import os
import subprocess
import sys  # ✅ FIXED: Moved to top
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

# --- Configuration ---
ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
MAP_DIR = OUTPUT_DIR / "maps"
REPORT_DIR = OUTPUT_DIR / "reports"
METRICS_FILE = REPORT_DIR / "metrics.json"
LOG_FILE = OUTPUT_DIR / "pipeline.log"

st.set_page_config(
    page_title="Sentinel-2 Segmentation Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ✅ FIXED: Removed broken Streamlit CSS block

# --- Sidebar ---
st.sidebar.title("🌍 Sentinel-2 ML Pipeline")
st.sidebar.markdown("---")
st.sidebar.info(
    "This dashboard controls the multispectral land-cover segmentation pipeline for the Bhopal region."
)

st.sidebar.subheader("Configuration")
data_mode = st.sidebar.radio(
    "Data Source",
    ["Synthetic (Quickstart)", "Real (Earth Engine)"],
    index=0,
    help="Select 'Real' to fetch live Bhopal imagery from Google Earth Engine (requires authentication).",
)
mode_arg = "gee" if "Real" in data_mode else "quickstart"

# ✅ FEATURE 5 - SEGFORMER: Model selection radio
model_choice = st.sidebar.radio(
    "Model",
    ["UNet only", "SegFormer only", "Both (Benchmark)"],
    index=0,
    help="Select model architecture. 'Both' runs a side-by-side benchmark comparison.",
)
model_arg = {"UNet only": "unet", "SegFormer only": "segformer", "Both (Benchmark)": "both"}[
    model_choice
]

# ✅ FEATURE 3 - SAR: Fusion checkbox in sidebar
use_fusion = st.sidebar.checkbox(
    "SAR + Optical Fusion",
    value=False,
    help="Adds synthetic Sentinel-1 VV+VH bands. Shows cloud recovery demo.",
)

if mode_arg == "gee":
    st.sidebar.warning(
        "⚠️ Ensure you have run 'ee.Authenticate()' in your terminal before starting."
    )

if "running" not in st.session_state:
    st.session_state.running = False

if st.sidebar.button(
    "🚀 Run Full Pipeline", use_container_width=True, disabled=st.session_state.running
):
    st.session_state.running = True
    with st.status(f"Running Pipeline ({mode_arg.upper()} mode)...", expanded=True) as status:
        st.write("Initializing environment...")
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        # ✅ FIXED: Removed hardcoded PROJ_DATA env override

        # Open log file
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        # ✅ FEATURE 5 - SEGFORMER: Build command with model arg
        cmd = [sys.executable, "run_pipeline.py", "--mode", mode_arg, "--model", model_arg]
        # ✅ FEATURE 3 - SAR: Add fusion flag if enabled
        if use_fusion:
            cmd.append("--fusion")
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            process = subprocess.Popen(
                cmd,  # ✅ FEATURE 5 - SEGFORMER: use dynamic command
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
                cwd=str(ROOT),
            )

            log_container = st.empty()
            full_log = ""

            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                if line:
                    f.write(line)
                    full_log += line
                    log_container.code("\n".join(full_log.splitlines()[-10:]))

        st.session_state.running = False
        if process.returncode == 0:
            status.update(
                label="✅ Pipeline Completed Successfully!", state="complete", expanded=False
            )
            st.balloons()
            st.rerun()
        else:
            status.update(label="❌ Pipeline Failed", state="error")
            st.error(f"Pipeline exited with code {process.returncode}")
            st.session_state.running = False

st.sidebar.markdown("---")
st.sidebar.subheader("System Status")
st.sidebar.success("Environment: .venv (Python 3.13)")
st.sidebar.info("Device: CPU (Training)")

# --- Main Dashboard ---
st.title("📊 Land-Cover Segmentation Dashboard")

# Load metrics if available
metrics = None
if METRICS_FILE.exists():
    try:
        with open(METRICS_FILE, "r") as f:
            metrics = json.load(f)
    except (json.JSONDecodeError, ValueError) as e:
        st.error(f"Corrupted metrics file detected: {e}. Please rerun the pipeline.")
        metrics = None

# Top Metrics Row
if metrics:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Overall Accuracy", f"{metrics['overall_accuracy'] * 100:.1f}%")
    with col2:
        st.metric("Mean IoU", f"{metrics['mean_iou'] * 100:.1f}%")
    with col3:
        st.metric("Classes", "10")
    with col4:
        st.metric("Resolution", "10m / px")
else:
    st.warning("No metrics found. Please run the pipeline to generate results.")

# Tabs for detailed view
tabs = st.tabs(
    [
        "🖼️ Predictions",
        "🗺️ Map View",
        "📈 Performance",
        "🔄 Change Detection",
        "🌿 NDVI Time-Series",
        "📡 SAR Fusion",
        "📄 Reports",
    ]
)  # ✅ FEATURE 3 - SAR: added SAR Fusion tab

with tabs[0]:
    st.header("Segmentation Results")
    pred_img_path = MAP_DIR / "sample_predictions.png"
    if pred_img_path.exists():
        st.image(
            Image.open(pred_img_path),
            caption="Sentinel-2 RGB | Ground Truth | UNet Prediction",
            use_container_width=True,
        )

    st.markdown("---")
    st.subheader("🖼️ Stitched Full Scene (Test Grid)")
    stitched_img_path = MAP_DIR / "full_stitched_scene.png"

    if stitched_img_path.exists():
        # Check if the image matches the current mode (stitching is quickstart-only)
        if mode_arg == "quickstart":
            import datetime

            mtime = os.path.getmtime(stitched_img_path)
            dt_str = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
            st.image(
                Image.open(stitched_img_path),
                caption=f"Full Scene Reassembled (Generated: {dt_str})",
                use_container_width=True,
            )
        else:
            st.info(
                "💡 The stitched scene from the previous Quickstart run is available "
                "in the outputs, but stitching is disabled for the current 'Real (GEE)' mode."
            )
            if st.checkbox("View stale Quickstart stitched scene anyway?"):
                st.image(
                    Image.open(stitched_img_path),
                    caption="Stale Quickstart Scene",
                    use_container_width=True,
                )
    else:
        if mode_arg == "gee":
            st.info(
                "ℹ️ Full scene stitching is only available in **Quickstart** mode "
                "(random Earth Engine patches cannot be perfectly stitched without a contiguous grid)."
            )
        else:
            st.info(
                "⏳ Stitched results will appear here after you run the pipeline in Quickstart mode."
            )

    # ✅ FEATURE 2 - CONFIDENCE: Expander in Predictions tab
    with st.expander("Confidence & Uncertainty Analysis"):
        uncertainty_img = MAP_DIR / "uncertainty_maps.png"
        if uncertainty_img.exists():
            st.image(
                str(uncertainty_img),
                caption="Green = high confidence, Red = uncertain. MC Dropout over 20 passes.",
                use_container_width=True,
            )

        if metrics and "mean_confidence" in metrics:
            st.metric("Mean Model Confidence", f"{metrics['mean_confidence'] * 100:.2f}%")
        else:
            st.info("Run the pipeline to generate uncertainty analysis.")

# ✅ FEATURE 4 - MAP: Tab implementation
with tabs[1]:
    st.header("Interactive GIS Web Map")
    overlay_path = MAP_DIR / "segmentation_overlay.png"
    bounds_path = REPORT_DIR / "map_bounds.json"

    if overlay_path.exists() and bounds_path.exists():
        try:
            import json

            from streamlit_folium import st_folium

            try:
                from src.vis.map_export import build_folium_map
            except (ImportError, ModuleNotFoundError):
                from vis.map_export import build_folium_map

            with open(bounds_path, "r") as f:
                bounds = json.load(f)

            m = build_folium_map(str(overlay_path), bounds)

            st.markdown(
                "Segmentation overlay on OpenStreetMap. "
                "Toggle the layer using the control in the top-right corner of the map."
            )
            # ✅ FIXED BUG 2: use_container_width=True instead of width="100%"
            #                 st_folium does not accept string widths
            st_folium(m, use_container_width=True, height=520)
        except Exception as e:
            st.error(f"Failed to load map: {e}")
    else:
        st.info("Run the pipeline in Quickstart mode to generate the map overlay.")

with tabs[2]:
    st.header("Model Performance")
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Confusion Matrix")
        conf_img_path = MAP_DIR / "confusion_matrix.png"
        if conf_img_path.exists():
            st.image(str(conf_img_path), use_container_width=True)
        else:
            st.info("Confusion matrix will appear after evaluation.")

    with col2:
        st.subheader("Per-Class IoU")
        if metrics:
            df_iou = pd.DataFrame(
                {
                    "Class": list(metrics["per_class_iou"].keys()),
                    "IoU (%)": [v * 100 for v in metrics["per_class_iou"].values()],
                }
            )
            st.bar_chart(df_iou.set_index("Class"))
        else:
            st.info("Class metrics will appear after training.")

    st.markdown("---")
    st.subheader("Training History & Learning Rate")
    log_dir = Path("outputs/lightning_logs")
    if log_dir.exists():
        versions = sorted(log_dir.glob("version_*"), key=os.path.getmtime, reverse=True)
        if versions:
            metrics_csv = versions[0] / "metrics.csv"
            if metrics_csv.exists():
                try:
                    df = pd.read_csv(metrics_csv)
                    col3, col4 = st.columns([1, 1])

                    with col3:
                        if "train_loss" in df.columns:
                            df_loss = df.groupby("epoch")[["train_loss"]].mean().dropna()
                            if "val_loss" in df.columns:
                                df_loss["val_loss"] = (
                                    df.groupby("epoch")["val_loss"].mean().dropna()
                                )
                            st.line_chart(df_loss)

                    with col4:
                        lr_cols = [c for c in df.columns if "lr" in c.lower()]
                        if lr_cols:
                            # Lightning logs LR on separate rows without epoch, so we forward-fill
                            df["epoch"] = df["epoch"].ffill().bfill()
                            df_lr = df.groupby("epoch")[lr_cols].mean().dropna()
                            st.line_chart(df_lr)
                        else:
                            st.info("No learning rate metrics found in the current log.")
                except Exception as e:
                    st.warning(f"Could not load training history: {e}")
    else:
        st.info("Training history will appear here after the first run.")

    # ✅ FEATURE 5 - SEGFORMER: Benchmark comparison section
    st.markdown("---")
    st.subheader("Model Benchmark")
    bench_img = MAP_DIR / "benchmark_comparison.png"
    if bench_img.exists():
        st.subheader("🏆 UNet vs SegFormer Benchmark")
        st.image(str(bench_img), use_container_width=True)

        bench_json = REPORT_DIR / "benchmark_results.json"
        if bench_json.exists():
            try:
                with open(bench_json) as f:
                    bench = json.load(f)
                cols = st.columns(len(bench))
                for col, result in zip(cols, bench):
                    col.metric(
                        result["model"].upper(),
                        f"{result['mean_iou'] * 100:.1f}% mIoU",
                        f"{result['train_time_sec'] / 60:.1f} min training",
                    )
            except Exception as e:
                st.warning(f"Could not load benchmark results: {e}")
    else:
        st.info("Run the pipeline with 'Both (Benchmark)' to compare models.")

with tabs[3]:
    st.header("Temporal Change Detection")
    st.markdown("Comparison between T1 (Pre-monsoon) and T2 (Post-monsoon) simulated patches.")

    change_img_path = MAP_DIR / "change_detection_maps.png"
    if change_img_path.exists():
        st.image(str(change_img_path), use_container_width=True)

        st.subheader("Transition Matrix (Hectares)")
        trans_file = REPORT_DIR / "transition_area_ha.csv"
        if trans_file.exists():
            df_trans = pd.read_csv(trans_file, index_col=0)
            st.dataframe(
                df_trans.style.background_gradient(cmap="YlOrRd"), use_container_width=True
            )
    else:
        st.info("Run the pipeline to generate change detection maps.")

with tabs[4]:
    st.header("🌿 NDVI Time-Series Monitoring")
    st.markdown("Analysis of seasonal vegetation health and anomaly detection.")

    # Section 1 — Monthly NDVI Curve
    st.subheader("Monthly NDVI Curve")
    curve_path = MAP_DIR / "ndvi_curve.png"
    stats_path = REPORT_DIR / "ndvi_stats.json"

    if curve_path.exists():
        st.image(str(curve_path), use_container_width=True)
        if stats_path.exists():
            with open(stats_path, "r") as f:
                stats = json.load(f)

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

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Peak Month", MONTH_NAMES[stats["peak_month"]])
            with col2:
                st.metric("Trough Month", MONTH_NAMES[stats["trough_month"]])
            with col3:
                st.metric("Anomalous Pixels", f"{stats['pct_anomalous']:.2f}%")
    else:
        st.info("Run the pipeline to generate the NDVI curve.")

    st.markdown("---")

    # Section 2 — Monthly Maps (Animation)
    st.subheader("Monthly Maps (Animation)")
    ts_dir = MAP_DIR / "timeseries"
    if ts_dir.exists():
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
        m_idx = st.slider("Select Month", 0, 11, 6, format="")
        st.markdown(f"**Viewing: {MONTH_NAMES[m_idx]}**")
        frame_path = ts_dir / f"ndvi_frame_{m_idx:02d}.png"
        if frame_path.exists():
            st.image(
                str(frame_path),
                use_container_width=True,
                caption="NDVI heatmap — green=vegetation, red=bare/urban",
            )
    else:
        st.info("Monthly NDVI frames will appear here after the run.")

    st.markdown("---")

    # Section 3 — Anomaly Detection Map
    st.subheader("Anomaly Detection Map")
    anomaly_path = MAP_DIR / "ndvi_anomaly_map.png"
    if anomaly_path.exists():
        st.image(
            str(anomaly_path),
            use_container_width=True,
            caption=(
                "Pixels with NDVI z-score > 2.0 across the year. "
                "Bright = high anomaly. Likely land-cover change or crop failure."
            ),
        )
    else:
        st.info("Anomaly detection results will appear here after the run.")

with tabs[5]:
    # ✅ FEATURE 3 - SAR: SAR Fusion tab
    st.header("📡 SAR + Optical Fusion Analysis")
    st.markdown("Sentinel-1 SAR (VV + VH polarization) combined with Sentinel-2 optical bands.")

    # Section 1 — SAR vs Optical Comparison
    st.subheader("SAR vs Optical Comparison")
    fusion_img_path = MAP_DIR / "fusion_vs_optical.png"
    fusion_json_path = REPORT_DIR / "fusion_results.json"

    if fusion_img_path.exists():
        st.image(str(fusion_img_path), use_container_width=True)
        if fusion_json_path.exists():
            try:
                with open(fusion_json_path) as f:
                    fusion_data = json.load(f)
                if len(fusion_data) >= 2:
                    col1, col2, col3 = st.columns(3)
                    optical_miou = fusion_data[0]["mean_iou"]
                    fusion_miou = fusion_data[1]["mean_iou"]
                    delta = fusion_miou - optical_miou
                    with col1:
                        st.metric("Optical mIoU", f"{optical_miou * 100:.1f}%")
                    with col2:
                        st.metric("Fusion mIoU", f"{fusion_miou * 100:.1f}%")
                    with col3:
                        st.metric("SAR Improvement", f"{delta * 100:+.1f}%")
            except Exception as e:
                st.warning(f"Could not load fusion results: {e}")
    else:
        st.info("Run the pipeline with 'SAR + Optical Fusion' enabled to see comparison.")

    st.markdown("---")

    # Section 2 — Cloud Recovery Demo
    st.subheader("Cloud Recovery Demo")
    cloud_img_path = MAP_DIR / "cloud_recovery.png"
    if cloud_img_path.exists():
        st.image(
            str(cloud_img_path),
            use_container_width=True,
            caption=(
                "Columns 3 and 4 show segmentation on the same patch with a simulated "
                "cloud blocking 100×100 pixels. The fusion model uses SAR (radar) data "
                "to partially recover accuracy in the occluded region."
            ),
        )
    else:
        st.info("Run the pipeline with 'SAR + Optical Fusion' enabled to see cloud recovery demo.")

    st.markdown("---")

    # Section 3 — SAR Band Visualization
    st.subheader("SAR Band Explanation")
    if fusion_json_path.exists():
        st.markdown("""
        **Understanding SAR Polarization Channels:**

        - **VV (Vertical-Vertical):** Sensitive to surface roughness and moisture content.
          High values indicate rough surfaces (urban, bare soil) or high moisture.
        - **VH (Vertical-Horizontal):** Sensitive to volume scattering from vegetation canopy.
          High values indicate dense vegetation (forest, cropland).
        - **VV/VH Ratio:** High ratio indicates urban double-bounce or bare soil;
          low ratio indicates volume-dominated scattering (forest canopy).

        **Why SAR helps under clouds:**
        Synthetic Aperture Radar operates at C-band (~5.4 GHz) which penetrates
        clouds, rain, and smoke. When optical imagery is obscured, SAR provides
        complementary structural and moisture information for classification.
        """)
    else:
        st.info("SAR band details will appear after running with fusion enabled.")

with tabs[6]:
    st.header("Project Reports & Exports")
    if metrics:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Detailed Classification Report")
            df_report = pd.DataFrame(
                {
                    "Metric": ["Overall Accuracy", "Mean IoU"],
                    "Value": [
                        f"{metrics['overall_accuracy'] * 100:.2f}%",
                        f"{metrics['mean_iou'] * 100:.2f}%",
                    ],
                }
            )
            st.table(df_report)

        with col2:
            st.subheader("Generated Artifacts")
            st.markdown(f"- **GeoTIFF Prediction:** `{REPORT_DIR / 'final_segmentation.tif'}`")
            st.markdown(f"- **Transition Matrix:** `{REPORT_DIR / 'transition_area_ha.csv'}`")
            st.markdown(f"- **Metrics JSON:** `{METRICS_FILE}`")

        with st.expander("View Raw Metrics JSON"):
            st.json(metrics)

        st.markdown("---")
        st.subheader("🔍 Dataset Visual Audit")
        audit_img_path = MAP_DIR / "dataset_audit.png"
        if audit_img_path.exists():
            st.image(
                Image.open(audit_img_path),
                caption="Raw Patches vs Labels (Phase 2 Diagnostic)",
                use_container_width=True,
            )
            st.info(
                "Check this audit to ensure that different land-cover types "
                "actually look distinct in the multispectral data."
            )
        else:
            st.info("Visual audit results will appear after the next pipeline run.")
    else:
        st.info("Reports will be generated after the pipeline run.")

# Footer
st.markdown("---")
st.markdown("Sentinel-2 Geospatial Intelligence")
