"""
SegFormer semantic segmentation Lightning module.

Uses nvidia/mit-b0 backbone fine-tuned for land cover classification.
The pretrained 3-channel patch embedding is replaced with a new Conv2d
accepting NUM_BANDS channels, initialized with kaiming_normal.
"""

import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy, F1Score, JaccardIndex
from transformers import (
    SegformerConfig,
    SegformerForSemanticSegmentation,
)
import numpy as np

try:
    import torch.serialization

    torch.serialization.add_safe_globals(
        [
            np._core.multiarray._reconstruct,
            np.dtype,
            np.ndarray,
            np.dtypes.Float32DType,
            np.dtypes.Int64DType,
        ]
    )
except (ImportError, AttributeError):
    pass


class SegFormerModule(pl.LightningModule):
    """SegFormer semantic segmentation module using nvidia/mit-b0 backbone.

    Input:  (B, C, H, W) float32
    Output: (B, num_classes, H, W) logits upsampled to full resolution.
    """

    def __init__(self, num_classes, num_bands, lr=5e-4, class_weights=None):
        """Initialize SegFormer module.

        Args:
            num_classes: Number of output segmentation classes.
            num_bands: Number of input spectral bands.
            lr: Learning rate for AdamW optimizer.
            class_weights: Optional per-class weight array for loss weighting.
        """
        super().__init__()
        weights_list = (
            class_weights.tolist() if isinstance(class_weights, np.ndarray) else class_weights
        )
        self.save_hyperparameters(ignore=["class_weights"])
        self.hparams.class_weights = weights_list

        if class_weights is not None and isinstance(class_weights, np.ndarray):
            class_weights = torch.from_numpy(class_weights).float()

        self.lr = lr
        self.num_classes = num_classes

        try:
            config = SegformerConfig.from_pretrained(
                "nvidia/mit-b0",
                num_labels=num_classes,
                ignore_mismatched_sizes=True,
            )
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                "nvidia/mit-b0",
                config=config,
                ignore_mismatched_sizes=True,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load SegFormer pretrained model 'nvidia/mit-b0': {e}. "
                "Ensure 'transformers' is installed and you have internet access for "
                "first-time model download."
            ) from e

        # Dynamically find the first projection Conv2d layer in the model (compatible across all transformers versions)
        proj_name = None
        old_proj = None
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d) and module.in_channels == 3:
                proj_name = name
                old_proj = module
                break

        if old_proj is None:
            raise RuntimeError(
                "Could not find the input Conv2d projection layer in the SegFormer model. "
                "Ensure you are loading a standard SegformerForSemanticSegmentation model."
            )

        new_proj = torch.nn.Conv2d(
            num_bands,
            old_proj.out_channels,
            kernel_size=old_proj.kernel_size,
            stride=old_proj.stride,
            padding=old_proj.padding,
        )
        torch.nn.init.kaiming_normal_(new_proj.weight, mode="fan_out")
        if new_proj.bias is not None:
            torch.nn.init.zeros_(new_proj.bias)

        # Traverse and set the new projection layer on the parent module
        parts = proj_name.split(".")
        parent = self.model
        for part in parts[:-1]:
            if part.isdigit():
                parent = parent[int(part)]
            else:
                parent = getattr(parent, part)
        setattr(parent, parts[-1], new_proj)

        self.focal_loss = smp.losses.FocalLoss(
            mode="multiclass", alpha=0.25, gamma=2.0, ignore_index=9
        )
        valid_classes = (
            [i for i, w in enumerate(class_weights) if w > 0.0 and i != 9]
            if class_weights is not None
            else None
        )
        self.dice_loss = smp.losses.DiceLoss(
            mode="multiclass",
            from_logits=True,
            classes=valid_classes,
            ignore_index=(9 if valid_classes is None else None),
        )

        self.val_iou = JaccardIndex(task="multiclass", num_classes=num_classes, average="macro")
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")

    def forward(self, x):
        """Forward pass with bilinear upsampling to input resolution.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Logits tensor of shape (B, num_classes, H, W).
        """
        outputs = self.model(pixel_values=x)
        logits = outputs.logits
        logits = F.interpolate(
            logits,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return logits

    def _shared_step(self, batch):
        """Shared forward and loss computation for train/val steps."""
        x, y = batch
        logits = self(x)
        loss = 0.5 * self.focal_loss(logits, y) + 0.5 * self.dice_loss(logits, y)
        preds = logits.argmax(dim=1)
        return loss, preds, y

    def training_step(self, batch, batch_idx):
        """Compute training loss and log."""
        loss, _, _ = self._shared_step(batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Compute validation loss, update metrics, and log."""
        loss, preds, y = self._shared_step(batch)
        self.val_iou(preds, y)
        self.val_acc(preds, y)
        self.val_f1(preds, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_mIoU", self.val_iou, on_epoch=True, prog_bar=True)
        self.log("val_acc", self.val_acc, on_epoch=True, prog_bar=True)
        self.log("val_f1", self.val_f1, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """Configure AdamW optimizer with CosineAnnealingLR scheduler."""
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=1e-6)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
