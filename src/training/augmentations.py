import albumentations as A
import cv2  # ✅ FIXED: Add import cv2 at the top


def get_train_transforms():
    """Returns spatial and color augmentations for training."""
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.0625,
                scale_limit=0.1,
                rotate_limit=45,
                p=0.5,
                border_mode=cv2.BORDER_REFLECT,
            ),  # ✅ FIXED: Add border_mode parameter
            # Note: We avoid standard RGB color jitter as we have multispectral data
        ]
    )


def get_val_transforms():
    """Returns no-op transforms for validation."""
    return A.Compose([])
