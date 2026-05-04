"""
Augmentation pipelines for multispectral land-cover segmentation training.

Provides spatial augmentations suitable for satellite imagery where standard
RGB color jitter is not applicable due to multispectral band structure.
"""

import albumentations as A
import cv2


def get_train_transforms():
    """Return an albumentations Compose pipeline with spatial augmentations for training.

    Includes horizontal/vertical flips, 90-degree rotations, and affine
    shift-scale-rotate with reflective border padding. Color jitter is
    intentionally omitted because the data is multispectral.

    Returns:
        A.Compose: Configured augmentation pipeline.
    """
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
            ),
        ]
    )


def get_val_transforms():
    """Return an identity (no-op) augmentation pipeline for validation.

    Returns:
        A.Compose: Empty augmentation pipeline.
    """
    return A.Compose([])
