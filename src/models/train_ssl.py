"""
SimCLR Self-Supervised pre-training script for Sentinel-2 encoders.

Pre-trains a ResNet34 encoder on unlabeled multispectral imagery patches
using contrastive learning (NT-Xent loss) with dual-view augmentations.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class ContrastiveDataset(Dataset):
    """Generates two stochastically augmented views of the same image patch
    for contrastive learning.
    """

    def __init__(self, image_dir, stats_path=None):
        self.image_paths = sorted(Path(image_dir).glob("*.npy"))
        self.means = None
        self.stds = None
        if stats_path and Path(stats_path).exists():
            stats = np.load(stats_path)
            self.means = stats[0].astype(np.float32).reshape(-1, 1, 1)
            self.stds = stats[1].astype(np.float32).reshape(-1, 1, 1)

    def __len__(self):
        return len(self.image_paths)

    def _apply_random_transforms(self, img):
        # Apply stochastic spatial and color transforms for multispectral inputs
        C, H, W = img.shape
        # Random horizontal/vertical flip
        if np.random.rand() > 0.5:
            img = np.flip(img, axis=2).copy()
        if np.random.rand() > 0.5:
            img = np.flip(img, axis=1).copy()

        # Add random gaussian noise
        if np.random.rand() > 0.5:
            noise = np.random.normal(0, 0.05, img.shape).astype(np.float32)
            img = np.clip(img + noise, 0, 1)

        # Scale intensity randomly (contrast jitter)
        if np.random.rand() > 0.5:
            factor = np.random.uniform(0.8, 1.2)
            img = np.clip(img * factor, 0, 1)

        return img

    def __getitem__(self, idx):
        img_raw = np.load(self.image_paths[idx]).astype(np.float32)

        # Resize to 256x256 if not already
        if img_raw.shape[1:] != (256, 256):
            import cv2

            img_hwc = img_raw.transpose(1, 2, 0)
            img_hwc = cv2.resize(img_hwc, (256, 256), interpolation=cv2.INTER_LINEAR)
            img_raw = img_hwc.transpose(2, 0, 1)

        # Generate two stochastically augmented views
        view_1 = self._apply_random_transforms(img_raw.copy())
        view_2 = self._apply_random_transforms(img_raw.copy())

        # Normalize if stats are present
        if self.means is not None:
            view_1 = (view_1 - self.means) / (self.stds + 1e-8)
            view_2 = (view_2 - self.means) / (self.stds + 1e-8)

        return torch.from_numpy(view_1), torch.from_numpy(view_2)


class NTXentLoss(nn.Module):
    """Normalized Temperature-scaled Cross Entropy Loss for SimCLR."""

    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        batch_size = z_i.shape[0]
        # Normalize representations
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        # Concatenate projections
        representations = torch.cat([z_i, z_j], dim=0)  # Shape: (2*B, out_dim)

        # Calculate cosine similarity matrix
        similarity_matrix = F.cosine_similarity(
            representations.unsqueeze(1), representations.unsqueeze(0), dim=2
        )

        # Positive pairs masking
        sim_ij = torch.diag(similarity_matrix, batch_size)
        sim_ji = torch.diag(similarity_matrix, -batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)

        # Negative pairs masking
        mask = (~torch.eye(batch_size * 2, device=z_i.device, dtype=torch.bool)).float()

        nominator = torch.exp(positives / self.temperature)
        denominator = mask * torch.exp(similarity_matrix / self.temperature)

        loss = -torch.log(nominator / (denominator.sum(dim=1) + 1e-8)).mean()
        return loss


class SimCLRModel(pl.LightningModule):
    """SimCLR PyTorch Lightning Module with ResNet34 backbone and MLP projection head."""

    def __init__(self, in_channels=16, projection_dim=128, lr=1e-3, temperature=0.5):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr

        # 1. Base Encoder: standard ResNet34 adapted for input channels
        import torchvision.models as models

        resnet = models.resnet34(weights=None)

        # Modify first layer for target spectral band count (e.g. 16 bands)
        resnet.conv1 = nn.Conv2d(
            in_channels,
            resnet.conv1.out_channels,
            kernel_size=resnet.conv1.kernel_size,
            stride=resnet.conv1.stride,
            padding=resnet.conv1.padding,
            bias=False,
        )

        self.encoder = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        # 2. Projection Head (MLP)
        self.projector = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, projection_dim)
        )

        self.loss_fn = NTXentLoss(temperature=temperature)

    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        return z

    def training_step(self, batch, batch_idx):
        x_i, x_j = batch
        z_i = self(x_i)
        z_j = self(x_j)
        loss = self.loss_fn(z_i, z_j)
        self.log("ssl_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-4)
        return optimizer


def main():
    parser = argparse.ArgumentParser(description="SimCLR Self-Supervised Pre-Training")
    parser.add_argument(
        "--data_dir", type=str, default="data/real/train/images", help="Unlabeled image directory"
    )
    parser.add_argument(
        "--stats_path", type=str, default="data/real/band_stats.npy", help="Band stats file path"
    )
    parser.add_argument("--epochs", type=int, default=15, help="Number of pre-training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--out_ckpt",
        type=str,
        default="outputs/checkpoints/ssl_encoder.ckpt",
        help="Output checkpoint path",
    )

    args = parser.parse_args()

    if not Path(args.data_dir).exists():
        print(f"❌ Unlabeled directory {args.data_dir} does not exist. Skipping pre-training.")
        return

    dataset = ContrastiveDataset(args.data_dir, stats_path=args.stats_path)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True
    )

    model = SimCLRModel(in_channels=16, lr=args.lr)

    # Train
    print("🚀 Starting SimCLR Self-Supervised Encoder Pre-Training...")
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="auto",
        devices=1,
        enable_checkpointing=False,
        logger=False,
    )
    trainer.fit(model, dataloader)

    # Save encoder state dict
    os.makedirs(os.path.dirname(args.out_ckpt), exist_ok=True)
    torch.save(model.encoder.state_dict(), args.out_ckpt)
    print(f"✅ Self-supervised encoder pre-training completed. Saved state to: {args.out_ckpt}")


if __name__ == "__main__":
    main()
