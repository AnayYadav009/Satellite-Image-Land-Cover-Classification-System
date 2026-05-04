"""
Raster utilities for GeoTIFF export and patch stitching.

Provides functions to save segmentation masks as GeoTIFFs with spatial
metadata and to reassemble predicted patches into a full scene mosaic.
"""

import numpy as np

try:
    import rasterio

    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    print("Warning: rasterio not found. Falling back to basic numpy loading.")


def save_segmentation_geotiff(mask, output_path, profile):
    """Save a predicted segmentation mask as a GeoTIFF.

    Args:
        mask: Integer mask array of shape (H, W).
        output_path: Destination file path for the GeoTIFF.
        profile: Rasterio profile dict with spatial metadata (CRS, transform, etc.).
    """
    if HAS_RASTERIO:
        profile.update(dtype=rasterio.uint8, count=1, compress="lzw")
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(mask.astype(np.uint8), 1)
    else:
        print("Warning: Could not save as GeoTIFF (rasterio missing). Saving as .npy instead.")
        np.save(output_path.replace(".tif", ".npy"), mask)


def stitch_patches(patches, original_shape, patch_size=256, overlap=0):
    """Reassemble predicted patches into a full scene mosaic.

    Places patches sequentially in row-major order into a canvas of the
    original scene dimensions. Patches at the borders are clipped to fit.

    Args:
        patches: List of 2D numpy arrays, each of shape (patch_size, patch_size).
        original_shape: Tuple of (H, W) for the full scene.
        patch_size: Spatial dimension of each square patch.
        overlap: Unused (reserved for future weighted overlap blending).

    Returns:
        Uint8 array of shape (H, W) containing the stitched segmentation mask.
    """
    H, W = original_shape
    full_mask = np.zeros((H, W), dtype=np.uint8)

    rows = (H + patch_size - 1) // patch_size
    cols = (W + patch_size - 1) // patch_size

    idx = 0
    for r in range(rows):
        for c in range(cols):
            if idx >= len(patches):
                break

            patch = patches[idx]
            y = r * patch_size
            x = c * patch_size

            h_p = min(patch_size, H - y)
            w_p = min(patch_size, W - x)

            full_mask[y : y + h_p, x : x + w_p] = patch[:h_p, :w_p]
            idx += 1

    return full_mask
