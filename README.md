# 🌍 Satellite Image Land-Cover Classification System

An advanced, production-ready geospatial machine learning pipeline for automated land-cover classification using **Sentinel-2 multispectral imagery** and **Sentinel-1 SAR data fusion**. This system integrates state-of-the-art architectures (UNet, SegFormer), uncertainty analysis, and temporal monitoring into a unified Streamlit dashboard.

---

## 🚀 Features

### 1. **Multi-Architecture Benchmarking**
- Compare the classic **UNet** (ResNet34 backbone) against the state-of-the-art **SegFormer** (`mit-b0`).
- Automated side-by-side performance reports including per-class IoU, F1-score, and inference speed.

### 2. **SAR-Optical Data Fusion**
- Integrates **Sentinel-1 (VV, VH)** radar backscatter with Sentinel-2 optical bands.
- Includes a **Cloud Recovery Simulation** demonstrating the model's ability to maintain high accuracy even under 100% cloud occlusion by leveraging radar data.

### 3. **Uncertainty & Confidence Analysis**
- Implements **MC Dropout** for Bayesian uncertainty estimation.
- Generates **Entropy Heatmaps** to visualize where the model is uncertain, enabling human-in-the-loop validation.

### 4. **NDVI Time-Series Monitoring**
- Automated vegetation health monitoring across a 12-month cycle.
- **Anomaly Detection**: Flagging significant deviations in vegetation growth (e.g., deforestation or drought) using pixel-wise z-scores.

### 5. **Interactive GIS Web Map**
- Seamless transition from model predictions to interactive maps.
- Supports **Reprojection to WGS84** and automated generation of **RGBA Overlays** for integration with Folium/Leaflet.

---

## 🛠️ Tech Stack

- **Framework**: PyTorch Lightning
- **Architectures**: Segmentation Models PyTorch (SMP), HuggingFace Transformers (SegFormer)
- **Geospatial**: Rasterio, Geopandas, Google Earth Engine (GEE)
- **Visualization**: Streamlit, Matplotlib, Seaborn, Folium
- **Data**: Multispectral (16 bands), Synthetic C-band SAR

---

## 📦 Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/AnayYadav009/Satellite-Image-Land-Cover-Classification-System.git
   cd Satellite-Image-Land-Cover-Classification-System
   ```

2. **Set up Virtual Environment**:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   source .venv/bin/activate  # Linux/Mac
   ```

3. **Install Dependencies**:
   ```bash
   pip install -e .
   ```

4. **Earth Engine Setup (Optional for Real Data)**:
   ```bash
   python -c "import ee; ee.Authenticate()"
   ```

---

## 🚦 Usage

### **1. Streamlit Dashboard (Recommended)**
The easiest way to interact with the system is through the dashboard:
```bash
streamlit run app.py
```

### **2. Command Line Interface**
For large-scale processing or CI/CD integration:
```bash
# Run baseline Quickstart
python run_pipeline.py --mode quickstart

# Run Benchmark (UNet vs SegFormer) with SAR Fusion
python run_pipeline.py --mode quickstart --model both --fusion
```

---

## 📂 Project Structure

```text
├── app.py                  # Streamlit dashboard interface
├── run_pipeline.py         # Master CLI execution script
├── src/
│   ├── data/               # Datasets, GEE loaders, SAR preprocessing
│   ├── models/             # SegFormer & UNet module definitions
│   ├── eval/               # Uncertainty & Accuracy metrics
│   ├── analysis/           # NDVI Time-series & Anomaly detection
│   ├── training/           # Data augmentations & Loss functions
│   └── vis/                # GIS map exporting & Visualization
└── pyproject.toml          # Dependency & Metadata management
```

---

## 🧪 Technical Details

- **Input Channels**: 16 bands (S2 B1-B12 + NDVI/NDWI/NDBI + Padding). 19 bands when SAR Fusion is active.
- **Loss Function**: Combined **Focal Loss** + **Dice Loss** for handling class imbalance.
- **Speckle Filtering**: Lee filter implementation for SAR noise reduction.
- **Post-Processing**: Morphological cleanup (opening/closing) to reduce noise in high-resolution masks.

---

## 📄 License
Distributed under the MIT License. See `LICENSE` for more information.
