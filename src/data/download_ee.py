"""
Google Earth Engine data download module for Sentinel-2 imagery.

Fetches real Sentinel-2 L2A multispectral patches and Dynamic World
land-cover labels for the Bhopal region.
"""

from pathlib import Path

import ee
import numpy as np
from tqdm import tqdm

BHOPAL_BBOX = [77.30, 23.15, 77.55, 23.35]
DATE_START = "2023-01-01"
DATE_END = "2023-12-31"
PATCH_SIZE = 256


def initialize_ee(project_id=None):
    """Initialize Google Earth Engine with optional project ID.

    Args:
        project_id: Google Cloud project ID. Falls back to ``GEE_PROJECT`` env var.

    Returns:
        True if initialization succeeded, False otherwise.
    """
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
        print("If using Google Cloud, specify a project ID via GEE_PROJECT env variable.")
        return False


def get_real_data_patch(lat, lon, size_px=256, scale=10):
    """Fetch a single multispectral patch and label from GEE.

    Args:
        lat: Latitude of the area of interest center.
        lon: Longitude of the area of interest center.
        size_px: Target spatial dimension (H and W).
        scale: Spatial resolution in meters per pixel.

    Returns:
        Tuple of (image, label) as numpy arrays, or (None, None) on failure.
    """
    point = ee.Geometry.Point([lon, lat])
    region = point.buffer(size_px * scale / 2).bounds()

    s2_col = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(region)
        .filterDate(DATE_START, DATE_END)
    )

    # Server-side validation of the collection size
    if s2_col.size().getInfo() == 0:
        return None, None

    s2 = s2_col.sort("CLOUDY_PIXEL_PERCENTAGE").first()

    img = s2.select(["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"])
    ndvi = img.normalizedDifference(["B8", "B4"]).rename("NDVI")
    ndwi = img.normalizedDifference(["B3", "B8"]).rename("NDWI")
    ndbi = img.normalizedDifference(["B11", "B8"]).rename("NDBI")
    dummy = ee.Image.constant(0).rename("dummy")
    img = img.addBands([ndvi, ndwi, ndbi, dummy])

    s2_date = ee.Date(s2.get("system:time_start"))
    dw_col = (
        ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
        .filterBounds(region)
        .filterDate(s2_date.advance(-5, "day"), s2_date.advance(5, "day"))
    )

    # Server-side validation of the Dynamic World labels collection
    if dw_col.size().getInfo() == 0:
        return None, None

    dw = dw_col.first()
    label = dw.select("label")

    try:
        import geemap

        img_np = geemap.ee_to_numpy(img, region=region, scale=scale)
        lbl_np = geemap.ee_to_numpy(label, region=region, scale=scale)

        if img_np is not None:
            import cv2

            if img_np.shape[:2] != (size_px, size_px):
                img_np = cv2.resize(img_np, (size_px, size_px), interpolation=cv2.INTER_LINEAR)
            if lbl_np.shape[:2] != (size_px, size_px):
                lbl_np = cv2.resize(lbl_np, (size_px, size_px), interpolation=cv2.INTER_NEAREST)

            if len(lbl_np.shape) == 3:
                lbl_np = lbl_np[:, :, 0]

            dw_to_our_classes = {0: 6, 1: 1, 2: 3, 3: 5, 4: 2, 5: 8, 6: 0, 7: 4, 8: 7}
            remapped_lbl = np.zeros_like(lbl_np)
            for dw_c, our_c in dw_to_our_classes.items():
                remapped_lbl[lbl_np == dw_c] = our_c
            lbl_np = remapped_lbl

            img_np = img_np.transpose(2, 0, 1)
            img_np = img_np / 10000.0
            img_np = np.clip(img_np, 0, 1)

        return img_np, lbl_np
    except Exception as e:
        print(f"Download failed for {lat}, {lon}: {e}")
        return None, None


def download_bhopal_dataset(
    output_dir="data/real", num_patches=20, patch_size=256, project_id=None
):
    """Download a dataset of real Sentinel-2 patches around Bhopal.

    Args:
        output_dir: Root directory for the downloaded dataset.
        num_patches: Total number of random patches to download.
        patch_size: Spatial dimension of each patch.
        project_id: Google Cloud project ID for Earth Engine.
    """
    if not initialize_ee(project_id=project_id):
        raise RuntimeError("Earth Engine initialization failed.")

    out = Path(output_dir)
    train_img_dir = out / "train" / "images"
    if (
        out.exists()
        and (out / "band_stats.npy").exists()
        and train_img_dir.exists()
        and list(train_img_dir.glob("*.npy"))
    ):
        print(f"  [INFO] Data already exists in {out}. Skipping download.")
        return

    if out.exists():
        import shutil

        print(f"  [INFO] Incomplete data in {out}. Cleaning up...")
        shutil.rmtree(out, ignore_errors=True)

    for split in ["train", "val", "test"]:
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        (out / split / "labels").mkdir(parents=True, exist_ok=True)

    print(f"Downloading {num_patches} real Sentinel-2 patches for Bhopal...")

    num_random = int(num_patches * 0.85)
    rng = np.random.default_rng(42)
    rand_lons = rng.uniform(BHOPAL_BBOX[0], BHOPAL_BBOX[2], num_random)
    rand_lats = rng.uniform(BHOPAL_BBOX[1], BHOPAL_BBOX[3], num_random)

    grid_size = 4
    center_lon, center_lat = 77.41, 23.25
    step_lon, step_lat = 0.025, 0.023

    grid_points = []
    for r in range(grid_size):
        for c in range(grid_size):
            lat = center_lat + (r - grid_size // 2) * step_lat
            lon = center_lon + (c - grid_size // 2) * step_lon
            grid_points.append((lat, lon))

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

    full_h, full_w = grid_size * patch_size, grid_size * patch_size
    np.save(out / "test" / "scene_shape.npy", np.array([full_h, full_w]))

    if all_images:
        stacked = np.stack(all_images)
        means = np.mean(stacked, axis=(0, 2, 3))
        stds = np.std(stacked, axis=(0, 2, 3))
        np.save(out / "band_stats.npy", np.stack([means, stds]))
        print(f"  Band statistics saved to {out / 'band_stats.npy'}")

    print(f"Successfully downloaded {count} real patches to {out}")


if __name__ == "__main__":
    download_bhopal_dataset()
