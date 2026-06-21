"""
Cross-Attention Fusion Module for Sentinel-2 (optical) and Sentinel-1 (SAR) data.

Provides a PyTorch model that extracts features from optical and SAR streams separately,
aligns and weights them using a self/cross-attention block, and outputs refined features
for semantic segmentation.
"""

import numpy as np
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy, F1Score, JaccardIndex


class CrossAttentionBlock(nn.Module):
    """Cross-Attention block that uses Optical features as Query, and SAR features as Key/Value.

    Allows the model to dynamically look at SAR features to recover details in regions
    where optical features are degraded (e.g. due to clouds or noise).
    """

    def __init__(self, opt_channels, sar_channels, out_channels):
        super().__init__()
        self.query_conv = nn.Conv2d(opt_channels, out_channels // 2, kernel_size=1)
        self.key_conv = nn.Conv2d(sar_channels, out_channels // 2, kernel_size=1)
        self.value_conv = nn.Conv2d(sar_channels, out_channels, kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.out_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x_opt, x_sar):
        """Forward pass.

        Args:
            x_opt: Optical features of shape (B, C_opt, H, W).
            x_sar: SAR features of shape (B, C_sar, H, W).

        Returns:
            Fused features of shape (B, out_channels, H, W).
        """
        batch_size, _, height, width = x_opt.size()

        # Project to Q, K, V
        proj_query = (
            self.query_conv(x_opt).view(batch_size, -1, width * height).permute(0, 2, 1)
        )  # B x N x C'
        proj_key = self.key_conv(x_sar).view(batch_size, -1, width * height)  # B x C' x N
        proj_value = self.value_conv(x_sar).view(batch_size, -1, width * height)  # B x C_out x N

        # Calculate attention map
        energy = torch.bmm(proj_query, proj_key)  # B x N x N
        attention = F.softmax(energy, dim=-1)  # B x N x N

        # Apply attention to values
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # B x C_out x N
        out = out.view(batch_size, -1, height, width)

        # Residual connection
        out = self.gamma * out + x_opt
        out = self.out_conv(out)
        return out


class CrossAttentionUNet(pl.LightningModule):
    """Multi-modal Cross-Attention segmentation model.

    Separately processes optical bands and SAR bands, fuses them using a CrossAttentionBlock,
    and classifies them using a UNet decoder.
    """

    def __init__(self, num_classes, num_opt_bands=16, num_sar_bands=3, lr=5e-4, class_weights=None):
        super().__init__()
        weights_list = (
            class_weights.tolist() if isinstance(class_weights, np.ndarray) else class_weights
        )
        self.save_hyperparameters(ignore=["class_weights"])
        self.hparams.class_weights = weights_list

        self.lr = lr
        self.num_classes = num_classes

        # Optical Stream Encoder (projecting 16 bands to 64 feature channels)
        self.opt_conv = nn.Sequential(
            nn.Conv2d(num_opt_bands, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # SAR Stream Encoder (projecting 3 bands to 64 feature channels)
        self.sar_conv = nn.Sequential(
            nn.Conv2d(num_sar_bands, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Cross-Attention fusion layer
        self.attention_fusion = CrossAttentionBlock(
            opt_channels=64, sar_channels=64, out_channels=64
        )

        # Core segmentation model accepting the fused 64-channel feature map
        self.unet = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=64,
            classes=num_classes,
            decoder_use_batchnorm=True,
        )

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
        """Forward pass. Expects x to be the fused tensor (C_opt + C_sar, H, W)."""
        # Split into optical and SAR components
        # C_opt = 16, C_sar = 3
        x_opt = x[:, :16, :, :]
        x_sar = x[:, 16:, :, :]

        # Extract features
        feat_opt = self.opt_conv(x_opt)
        feat_sar = self.sar_conv(x_sar)

        # Fuse via cross-attention
        feat_fused = self.attention_fusion(feat_opt, feat_sar)

        # Run through UNet
        logits = self.unet(feat_fused)
        return logits

    def _shared_step(self, batch):
        x, y = batch
        logits = self(x)
        loss = 0.5 * self.focal_loss(logits, y) + 0.5 * self.dice_loss(logits, y)
        preds = logits.argmax(dim=1)
        return loss, preds, y

    def training_step(self, batch, batch_idx):
        loss, _, _ = self._shared_step(batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
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
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=1e-6)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
