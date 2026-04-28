from pathlib import Path

import ee
import numpy as np
from tqdm import tqdm

# --- Bhopal Region Defaults ---
BHOPAL_BBOX = [77.30, 23.15, 77.55, 23.35]  # [min_lon, min_lat, max_lon, max_lat]
DATE_START = "2023-01-01"
DATE_END = "2023-12-31"
PATCH_SIZE = 256  # ✅ UPDATED to 256x256 as per requirement


def initialize_ee(project_id=None):
    """Initializes Earth Engine. Requires previous ee.Authenticate()."""
    import os

    if project_id is None:
        project_id = os.environ.get("GEE_PROJECT")

    try:
        if project_id:
            ee.Initialize(project=project_id)
        else:
            ee.Initialize()
        return True
    except Exception as e:
        print(f"Error initializing Earth Engine: {e}")
        print('Please run: .venv\\Scripts\\python.exe -c "import ee; ee.Authenticate()"')
        print(
            "If using Google Cloud, ensure you specify a project ID via GEE_PROJECT environment variable."
        )
        return False


def get_real_data_patch(lat, lon, size_px=256, scale=10):
    """
    Fetches a single multispectral patch and its label from GEE.
    Ensures fixed output size (size_px, size_px).
    """
    point = ee.Geometry.Point([lon, lat])
    region = point.buffer(size_px * scale / 2).bounds()

    # 1. Sentinel-2 L2A (Multispectral)
    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(region)
        .filterDate(DATE_START, DATE_END)
        .sort("CLOUDY_PIXEL_PERCENTAGE")
        .first()
    )

    if s2 is None:
        return None, None

    # Select standard 12 bands + compute 3 indices (NDVI, NDWI, NDBI)
    # Bands: B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B11, B12
    img = s2.select(["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"])

    # Add Indices (matching our 16-band requirement)
    ndvi = img.normalizedDifference(["B8", "B4"]).rename("NDVI")
    ndwi = img.normalizedDifference(["B3", "B8"]).rename("NDWI")
    ndbi = img.normalizedDifference(["B11", "B8"]).rename("NDBI")
    # Add a dummy 16th band for padding if needed
    dummy = ee.Image.constant(0).rename("dummy")

    img = img.addBands([ndvi, ndwi, ndbi, dummy])

    # Get the exact date of the selected S2 image
    s2_date = ee.Date(s2.get("system:time_start"))

    # 2. Dynamic World Labels (Ground Truth)
    # Find the matching Dynamic World label within a 5-day window
    dw = (
        ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
        .filterBounds(region)
        .filterDate(s2_date.advance(-5, "day"), s2_date.advance(5, "day"))
        .first()
    )

    if dw is None:
        return None, None
    label = dw.select("label")

    # 3. Download as Numpy
    # Note: getThumbURL or sampleRectangle are options. geemap.ee_to_numpy is best.
    try:
        import geemap

        img_np = geemap.ee_to_numpy(img, region=region, scale=scale)
        lbl_np = geemap.ee_to_numpy(label, region=region, scale=scale)

        # Reshape to (C, H, W) and (H, W)
        if img_np is not None:
            import cv2

            # ✅ FIXED: Ensure uniform size before stacking
            if img_np.shape[:2] != (size_px, size_px):
                img_np = cv2.resize(img_np, (size_px, size_px), interpolation=cv2.INTER_LINEAR)
            if lbl_np.shape[:2] != (size_px, size_px):
                lbl_np = cv2.resize(lbl_np, (size_px, size_px), interpolation=cv2.INTER_NEAREST)

            # Label might come back with extra channel
            if len(lbl_np.shape) == 3:
                lbl_np = lbl_np[:, :, 0]

            # Remap Dynamic World labels to our CLASS_NAMES
            dw_to_our_classes = {
                0: 6,  # Water -> Water
                1: 1,  # Trees -> Forest
                2: 3,  # Grass -> Grassland
                3: 5,  # Flooded veg -> Wetlands
                4: 2,  # Crops -> Cropland
                5: 8,  # Shrub & Scrub -> Shrubland
                6: 0,  # Built -> Urban
                7: 4,  # Bare -> Bare Soil
                8: 7,  # Snow -> Snow
            }
            remapped_lbl = np.zeros_like(lbl_np)
            for dw_c, our_c in dw_to_our_classes.items():
                remapped_lbl[lbl_np == dw_c] = our_c
            lbl_np = remapped_lbl

            img_np = img_np.transpose(2, 0, 1)  # (H,W,C) -> (C,H,W)
            img_np = img_np / 10000.0  # Scale S2 DN to 0-1
            img_np = np.clip(img_np, 0, 1)

        return img_np, lbl_np
    except Exception as e:
        print(f"Download failed for {lat}, {lon}: {e}")
        return None, None


def download_bhopal_dataset(
    output_dir="data/real", num_patches=20, patch_size=256, project_id=None
):
    """Downloads a set of real patches around Bhopal."""
    if not initialize_ee(project_id=project_id):
        raise RuntimeError(
            "Earth Engine initialization failed. Check your GEE_PROJECT_ID and authentication."
        )

    out = Path(output_dir)
    # Check if we should skip download
    if out.exists() and (out / "band_stats.npy").exists():
        print(f"  [INFO] Data already exists in {out}. Skipping download.")
        return

    # If missing, clean and start over
    if out.exists():
        import shutil

        print(f"  [INFO] Incomplete data in {out}. Cleaning up...")
        shutil.rmtree(out, ignore_errors=True)

    for split in ["train", "val", "test"]:
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        (out / split / "labels").mkdir(parents=True, exist_ok=True)

    print(f"Downloading {num_patches} real Sentinel-2 patches for Bhopal...")

    # 1. Generate random points for Train and Val
    num_random = int(num_patches * 0.85)
    rng = np.random.default_rng(42)
    rand_lons = rng.uniform(BHOPAL_BBOX[0], BHOPAL_BBOX[2], num_random)
    rand_lats = rng.uniform(BHOPAL_BBOX[1], BHOPAL_BBOX[3], num_random)

    # 2. Generate contiguous grid for Test (4x4 grid = 16 patches)
    grid_size = 4
    center_lon, center_lat = 77.41, 23.25  # Central Bhopal
    # ~2.56km steps in degrees
    step_lon, step_lat = 0.025, 0.023

    grid_points = []
    for r in range(grid_size):
        for c in range(grid_size):
            lat = center_lat + (r - grid_size // 2) * step_lat
            lon = center_lon + (c - grid_size // 2) * step_lon
            grid_points.append((lat, lon))

    # Combine all points
    all_points = []
    for i in range(num_random):
        all_points.append((rand_lats[i], rand_lons[i], "train" if i < num_patches * 0.7 else "val"))
    for lat, lon in grid_points:
        all_points.append((lat, lon, "test"))

    count = 0
    all_images = []
    for lat, lon, split in tqdm(all_points, desc="Downloading GEE patches"):
        img, lbl = get_real_data_patch(lat, lon, size_px=patch_size)

        if img is not None and lbl is not None:
            np.save(out / split / "images" / f"patch_{count:04d}.npy", img)
            np.save(out / split / "labels" / f"patch_{count:04d}.npy", lbl)

            if split == "train":
                all_images.append(img)
            count += 1

    # Save original scene shape for test grid (3x3 grid of 256x256)
    full_h, full_w = grid_size * patch_size, grid_size * patch_size
    np.save(out / "test" / "scene_shape.npy", np.array([full_h, full_w]))

    # Compute and save band statistics
    if all_images:
        stacked = np.stack(all_images)
        means = np.mean(stacked, axis=(0, 2, 3))
        stds = np.std(stacked, axis=(0, 2, 3))
        np.save(out / "band_stats.npy", np.stack([means, stds]))
        print(f"  [DEBUG] Saved band_stats.npy to {out / 'band_stats.npy'}")
        print(f"  Band statistics saved to {out / 'band_stats.npy'}")

    print(f"Successfully downloaded {count} real patches to {out}")


if __name__ == "__main__":
    download_bhopal_dataset()
