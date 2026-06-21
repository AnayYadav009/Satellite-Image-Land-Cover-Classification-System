"""
FastAPI GIS Map Server.

Serves dynamic Web Map Service (WMS/TMS) XYZ tiles from the reprojected
final_segmentation_wgs84.tif GeoTIFF. Allows loading model predictions directly into
QGIS, ArcGIS, or Leaflet maps.
"""

import math
from io import BytesIO
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.windows import from_bounds

try:
    from fastapi import FastAPI, HTTPException, Response
    from fastapi.responses import HTMLResponse
except ImportError:
    print("❌ FastAPI is required to run the GIS server. Installing dependencies...")
    import subprocess
    import sys

    subprocess.check_call([sys.executable, "-m", "pip", "install", "fastapi", "uvicorn"])
    from fastapi import FastAPI, HTTPException, Response
    from fastapi.responses import HTMLResponse

app = FastAPI(
    title="Sentinel-2 GIS Map Server",
    description=(
        "Dynamic tile server (XYZ/TMS) serving Sentinel-2 land cover "
        "classification predictions."
    ),
)

GEOTIFF_PATH = Path("outputs/reports/final_segmentation_wgs84.tif")

CLASS_COLORS = [
    "#FF0000",  # Urban - Red
    "#006400",  # Forest - Dark Green
    "#FFD700",  # Cropland - Gold
    "#7CFC00",  # Grassland - Lawn Green
    "#D2B48C",  # Bare Soil - Tan
    "#00CED1",  # Wetlands - Dark Turquoise
    "#0000FF",  # Water - Blue
    "#FFFFFF",  # Snow - White
    "#8B4513",  # Shrubland - Saddle Brown
    "#808080",  # Clouds - Grey
]


def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


RGB_COLORS = [hex_to_rgb(c) for c in CLASS_COLORS]


def get_tile_bounds(x: int, y: int, z: int) -> tuple[float, float, float, float]:
    """Calculate the bounding box of an XYZ tile in EPSG:4326 coordinates.

    Returns:
        (west, south, east, north) in degrees.
    """
    n = 2.0**z
    lon_min = x / n * 360.0 - 180.0
    lon_max = (x + 1) / n * 360.0 - 180.0

    lat_rad_min = math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n)))
    lat_min = math.degrees(lat_rad_min)

    lat_rad_max = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    lat_max = math.degrees(lat_rad_max)

    return (lon_min, lat_min, lon_max, lat_max)


@app.get("/tile/{z}/{x}/{y}.png")
def get_tile(z: int, x: int, y: int):
    """Serve dynamic PNG tiles by reading from the reprojected WGS84 GeoTIFF."""
    if not GEOTIFF_PATH.exists():
        raise HTTPException(
            status_code=404, detail="GeoTIFF prediction file not found. Run the pipeline first."
        )

    # 1. Compute bounds of requested slippy tile
    w, s, e, n = get_tile_bounds(x, y, z)

    try:
        with rasterio.open(GEOTIFF_PATH) as src:
            # 2. Check if requested bounds overlap with GeoTIFF bounds
            gt_w, gt_s, gt_e, gt_n = src.bounds

            # No overlap -> return fully transparent tile
            if w > gt_e or e < gt_w or s > gt_n or n < gt_s:
                img = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
                buf = BytesIO()
                img.save(buf, format="PNG")
                return Response(content=buf.getvalue(), media_type="image/png")

            # 3. Read window from GeoTIFF corresponding to tile bounds
            window = from_bounds(w, s, e, n, src.transform)
            # Read first band, clip window sizes to avoid floating point index issues
            mask = src.read(1, window=window, boundless=True, fill_value=9)

        # 4. Map class index to RGBA color map
        h, w = mask.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)

        # Populate RGB and Alpha channels
        for c_idx, color in enumerate(RGB_COLORS):
            class_pixels = mask == c_idx
            rgba[class_pixels, 0] = color[0]
            rgba[class_pixels, 1] = color[1]
            rgba[class_pixels, 2] = color[2]
            rgba[class_pixels, 3] = 180  # Alpha opacity

        # Default fill transparent for ignore_index or no data
        rgba[mask == 9, 3] = 0

        # Resize to standard tile dimensions (256x256)
        img = Image.fromarray(rgba)
        if img.size != (256, 256):
            img = img.resize((256, 256), Image.NEAREST)

        # 5. Output PNG bytes
        buf = BytesIO()
        img.save(buf, format="PNG")
        return Response(content=buf.getvalue(), media_type="image/png")

    except Exception:
        # Fallback to empty transparent tile on read error
        img = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
        buf = BytesIO()
        img.save(buf, format="PNG")
        return Response(content=buf.getvalue(), media_type="image/png")


@app.get("/", response_class=HTMLResponse)
def get_map_viewer():
    """Returns a simple web page containing a Leaflet map demonstrating the TMS endpoint."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Bhopal Land Cover GIS Map Viewer</title>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
        <style>
            html, body, #map { height: 100%; margin: 0; padding: 0; }
            #title-box {
                position: absolute; top: 10px; left: 50px; z-index: 1000;
                background: white; padding: 10px; border-radius: 5px;
                box-shadow: 0 0 15px rgba(0,0,0,0.2); font-family: sans-serif;
            }
        </style>
    </head>
    <body>
        <div id="title-box">
            <h3 style="margin:0;">🌍 Bhopal Land-Cover Dynamic GIS Server</h3>
            <p style="margin:5px 0 0; font-size:12px; color:grey;">
                Serving WMS/TMS Tiles at <code>/tile/{z}/{x}/{y}.png</code>
            </p>
        </div>
        <div id="map"></div>
        <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
        <script>
            var map = L.map('map').setView([23.25, 77.41], 13);
            
            L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                maxZoom: 19,
                attribution: '© OpenStreetMap'
            }).addTo(map);

            // Add our model prediction XYZ TMS tile layer
            L.tileLayer('/tile/{z}/{x}/{y}.png', {
                maxZoom: 18,
                opacity: 0.8,
                attribution: '© Sentinel-2 ML Pipeline Model Predictions'
            }).addTo(map);
        </script>
    </body>
    </html>
    """
    return html_content


if __name__ == "__main__":
    import uvicorn

    print("🚀 Starting FastAPI GIS Server on http://localhost:8000 ...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
