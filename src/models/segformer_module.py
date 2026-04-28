"""
✅ FEATURE 5 - SEGFORMER: SegFormer semantic segmentation Lightning module.
Uses nvidia/mit-b0 backbone fine-tuned for land cover classification.
Input:  (B, C, H, W) float32 — same format as LandCoverModule.
Output: (B, num_classes, H, W) logits at full resolution.

Note on input channels: SegFormer's pretrained weights expect 3 channels
(RGB). Since we have NUM_BANDS channels, we replace the patch embedding
conv layer with a new Conv2d(NUM_BANDS, embed_dim, ...) and initialize it
with kaiming_normal. The rest of the pretrained weights are kept.
"""

import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy, F1Score, JaccardIndex
from transformers import (  # ✅ FEATURE 5 - SEGFORMER: HuggingFace SegFormer
    SegformerConfig,
    SegformerForSemanticSegmentation,
)


class SegFormerModule(pl.LightningModule):
    """
    SegFormer semantic segmentation module.
    Uses nvidia/mit-b0 backbone fine-tuned for land cover classification.
    Input:  (B, C, H, W) float32 — same format as LandCoverModule.
    Output: (B, num_classes, H, W) logits at full resolution.

    Note on input channels: SegFormer's pretrained weights expect 3 channels
    (RGB). Since we have NUM_BANDS channels, we replace the patch embedding
    conv layer with a new Conv2d(NUM_BANDS, embed_dim, ...) and initialize it
    with kaiming_normal. The rest of the pretrained weights are kept.
    """

    def __init__(self, num_classes, num_bands, lr=5e-4, class_weights=None):
        """
        Initialize SegFormer module with pretrained nvidia/mit-b0 backbone.

        Args:
            num_classes: Number of output segmentation classes.
            num_bands: Number of input spectral bands (replaces default 3-channel input).
            lr: Learning rate for AdamW optimizer.
            class_weights: Optional array of per-class weights for loss weighting.
        """
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.num_classes = num_classes

        # ✅ FEATURE 5 - SEGFORMER: Load pretrained config and model
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

        # ✅ FEATURE 5 - SEGFORMER: Replace first patch embedding conv to accept NUM_BANDS input channels
        # Original: PatchEmbedding.proj = Conv2d(3, 32, kernel_size=7, stride=4, padding=3)
        old_proj = self.model.segformer.encoder.patch_embeddings[0].proj
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
        self.model.segformer.encoder.patch_embeddings[0].proj = new_proj

        # ✅ FEATURE 5 - SEGFORMER: Combined Focal + Dice loss — same as LandCoverModule for fair comparison
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

        # ✅ FEATURE 5 - SEGFORMER: Validation metrics — same as LandCoverModule
        self.val_iou = JaccardIndex(task="multiclass", num_classes=num_classes, average="macro")
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_f1 = F1Score(task="multiclass", num_classes=num_classes, average="macro")

    def forward(self, x):
        """
        Forward pass through SegFormer.
        SegFormer returns logits at 1/4 resolution (H/4, W/4).
        We bilinearly upsample back to (H, W) to match UNet output format.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Logits tensor of shape (B, num_classes, H, W).
        """
        outputs = self.model(pixel_values=x)
        logits = outputs.logits  # (B, num_classes, H/4, W/4)
        logits = F.interpolate(
            logits,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return logits

    def _shared_step(self, batch):
        """Shared forward + loss computation for train and val steps."""
        x, y = batch
        logits = self(x)
        loss = 0.5 * self.focal_loss(logits, y) + 0.5 * self.dice_loss(logits, y)
        preds = logits.argmax(dim=1)
        return loss, preds, y

    def training_step(self, batch, batch_idx):
        """Training step: compute loss and log."""
        loss, _, _ = self._shared_step(batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step: compute loss, update metrics, and log."""
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
