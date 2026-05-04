"""
Visualization utilities for creating interactive map overlays.

Provides GeoTIFF reprojection, segmentation mask to RGBA conversion,
and Folium map construction for Streamlit dashboard integration.
"""

import folium
import numpy as np
import rasterio
import rasterio.warp
from PIL import Image


def reproject_to_wgs84(src_path: str, dst_path: str) -> None:
    """Reproject a GeoTIFF to EPSG:4326 for Leaflet map integration.

    Args:
        src_path: Path to the source GeoTIFF.
        dst_path: Path for the reprojected output GeoTIFF.
    """
    try:
        with rasterio.open(src_path) as src:
            dst_crs = "EPSG:4326"
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


def segmentation_mask_to_rgba_png(
    mask_path: str,
    output_png_path: str,
    class_colors: list[str],
    alpha: int = 180,
) -> tuple[float, float, float, float]:
    """Convert a segmentation GeoTIFF to an RGBA PNG overlay.

    Args:
        mask_path: Path to a single-band uint8 segmentation GeoTIFF.
        output_png_path: Destination file path for the RGBA PNG.
        class_colors: List of hex color strings, one per class.
        alpha: Overlay transparency (0–255).

    Returns:
        Tuple of (west, south, east, north) bounds in WGS84 degrees.
    """
    try:
        with rasterio.open(mask_path) as src:
            mask = src.read(1)
            bounds = src.bounds
            west, south, east, north = bounds.left, bounds.bottom, bounds.right, bounds.top

        def hex_to_rgb(hex_color):
            hex_color = hex_color.lstrip("#")
            return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))

        rgb_colors = [hex_to_rgb(c) for c in class_colors]

        h, w = mask.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)

        for c_idx, color in enumerate(rgb_colors):
            class_pixels = mask == c_idx
            rgba[class_pixels, 0] = color[0]
            rgba[class_pixels, 1] = color[1]
            rgba[class_pixels, 2] = color[2]
            rgba[class_pixels, 3] = alpha

        img = Image.fromarray(rgba)
        img.save(output_png_path)

        return (west, south, east, north)
    except Exception as e:
        print(f"Error in segmentation_mask_to_rgba_png: {e}")
        raise


def build_folium_map(
    rgba_png_path: str,
    bounds: tuple[float, float, float, float],
    center: tuple[float, float] | None = None,
) -> folium.Map:
    """Create a Folium map with a segmentation overlay.

    Args:
        rgba_png_path: Path to the RGBA PNG overlay image.
        bounds: Tuple of (west, south, east, north) in WGS84 degrees.
        center: Optional (lat, lon) center for the map view.

    Returns:
        Configured folium.Map object with the overlay and layer control.
    """
    try:
        west, south, east, north = bounds
        if center is None:
            center = [(south + north) / 2, (west + east) / 2]

        m = folium.Map(location=center, zoom_start=13, tiles="OpenStreetMap")
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
