import numpy as np

try:
    import rasterio

    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    print("Warning: rasterio not found. Falling back to basic numpy loading.")


def save_segmentation_geotiff(mask, output_path, profile):
    """✅ ADDED: Saves predicted mask as a GeoTIFF with original spatial metadata."""
    if HAS_RASTERIO:
        # Update profile for single band output
        profile.update(dtype=rasterio.uint8, count=1, compress="lzw")
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(mask.astype(np.uint8), 1)
    else:
        print("Warning: Could not save as GeoTIFF (rasterio missing). Saving as .npy instead.")
        np.save(output_path.replace(".tif", ".npy"), mask)


def stitch_patches(patches, original_shape, patch_size=256, overlap=0):
    """✅ ADDED: Reassembles output tiles back into a full scene."""
    H, W = original_shape
    full_mask = np.zeros((H, W), dtype=np.uint8)

    # Simple non-overlapping stitching for demo
    # In production, we'd use weight maps for overlapping patches
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

            # Clip patch if it goes out of bounds
            h_p = min(patch_size, H - y)
            w_p = min(patch_size, W - x)

            full_mask[y : y + h_p, x : x + w_p] = patch[:h_p, :w_p]
            idx += 1

    return full_mask
