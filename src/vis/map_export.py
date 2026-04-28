"""
Visualization utilities for creating interactive map overlays.
"""

import folium
import numpy as np
import rasterio
import rasterio.warp
from PIL import Image


# ✅ FEATURE 4 - MAP: Function to reproject a GeoTIFF to WGS84 for Leaflet map integration
def reproject_to_wgs84(src_path: str, dst_path: str) -> None:
    """Reprojects any GeoTIFF to EPSG:4326 using rasterio.warp.reproject.
    Required so Leaflet (which uses WGS84) can position the overlay correctly."""
    try:
        with rasterio.open(src_path) as src:
            dst_crs = "EPSG:4326"

            # Calculate transform and dimensions for the destination CRS
            transform, width, height = rasterio.warp.calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds
            )

            kwargs = src.meta.copy()
            kwargs.update(
                {"crs": dst_crs, "transform": transform, "width": width, "height": height}
            )

            with rasterio.open(dst_path, "w", **kwargs) as dst:
                for i in range(1, src.count + 1):
                    rasterio.warp.reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=dst_crs,
                        resampling=rasterio.warp.Resampling.nearest,
                    )
    except Exception as e:
        print(f"Error in reproject_to_wgs84: {e}")
        raise


# ✅ FEATURE 4 - MAP: Convert a segmentation mask to RGBA PNG and return WGS84 bounds
def segmentation_mask_to_rgba_png(
    mask_path: str,
    output_png_path: str,
    class_colors: list[str],  # hex strings, one per class
    alpha: int = 180,  # 0-255 transparency of overlay
) -> tuple[float, float, float, float]:
    """Converts a single-band uint8 segmentation GeoTIFF into an RGBA PNG
    where each pixel is colored by its class. Returns (west, south, east, north)
    bounds in WGS84 degrees for use as Leaflet ImageOverlay bounds."""
    try:
        with rasterio.open(mask_path) as src:
            mask = src.read(1)
            bounds = src.bounds

            # Calculate WGS84 bounds
            west, south, east, north = bounds.left, bounds.bottom, bounds.right, bounds.top

        # Convert hex colors to RGB tuples
        def hex_to_rgb(hex_color):
            hex_color = hex_color.lstrip("#")
            return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))

        rgb_colors = [hex_to_rgb(c) for c in class_colors]

        # Create RGBA image
        h, w = mask.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)

        for c_idx, color in enumerate(rgb_colors):
            class_pixels = mask == c_idx
            rgba[class_pixels, 0] = color[0]
            rgba[class_pixels, 1] = color[1]
            rgba[class_pixels, 2] = color[2]
            rgba[class_pixels, 3] = alpha

        # Save to PNG
        img = Image.fromarray(rgba)
        img.save(output_png_path)

        return (west, south, east, north)
    except Exception as e:
        print(f"Error in segmentation_mask_to_rgba_png: {e}")
        raise


# ✅ FEATURE 4 - MAP: Build the Folium map with the image overlay
def build_folium_map(
    rgba_png_path: str,
    bounds: tuple[float, float, float, float],  # (west, south, east, north)
    center: tuple[float, float] | None = None,
) -> folium.Map:
    """Creates a folium.Map centered on the scene with:
      - OpenStreetMap as base layer
      - The RGBA PNG as a folium.raster_layers.ImageOverlay with the given bounds
      - A folium.LayerControl so the overlay can be toggled
    Returns the folium.Map object."""
    try:
        west, south, east, north = bounds

        if center is None:
            center = [(south + north) / 2, (west + east) / 2]

        m = folium.Map(location=center, zoom_start=13, tiles="OpenStreetMap")

        # Image bounds for Folium is [[lat_min, lon_min], [lat_max, lon_max]]
        # Or in south, west, north, east order: [[south, west], [north, east]]
        folium_bounds = [[south, west], [north, east]]

        folium.raster_layers.ImageOverlay(
            image=rgba_png_path,
            bounds=folium_bounds,
            opacity=1.0,
            name="Segmentation Overlay",
            interactive=True,
            cross_origin=False,
            zindex=1,
        ).add_to(m)

        folium.LayerControl().add_to(m)

        return m
    except Exception as e:
        print(f"Error in build_folium_map: {e}")
        raise
