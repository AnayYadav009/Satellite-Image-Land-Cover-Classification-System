"""
SAR-to-Optical Image Translation Module (Pix2Pix cGAN).

Provides a conditional GAN model that translates 3-channel Sentinel-1 SAR backscatter
[VV, VH, VV/VH ratio] to 3-channel optical RGB (Red, Green, Blue) to reconstruct
surface imagery under heavy cloud cover.
"""

import os
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


# -------------------------------------------------------------
# 1. Generator: U-Net Encoder-Decoder Architecture
# -------------------------------------------------------------
class UNetBlock(nn.Module):
    def __init__(self, in_c, out_c, down=True, use_dropout=False):
        super().__init__()
        if down:
            self.block = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            self.block = nn.Sequential(
                nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            )
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        out = self.block(x)
        if self.use_dropout:
            out = self.dropout(out)
        return out


class SARToOpticalGenerator(nn.Module):
    """UNet Generator mapping (B, 3, 256, 256) SAR to (B, 3, 256, 256) Optical RGB."""

    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        # Encoder
        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.down2 = UNetBlock(64, 128, down=True)
        self.down3 = UNetBlock(128, 256, down=True)
        self.down4 = UNetBlock(256, 512, down=True)
        self.down5 = UNetBlock(512, 512, down=True)

        # Decoder (with skip connections)
        self.up1 = UNetBlock(512, 512, down=False, use_dropout=True)
        self.up2 = UNetBlock(1024, 256, down=False)
        self.up3 = UNetBlock(512, 128, down=False)
        self.up4 = UNetBlock(256, 64, down=False)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),  # Normalized output range [-1, 1]
        )

    def forward(self, x):
        # Encoder passes
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)

        # Decoder passes with skip connections
        u1 = self.up1(d5)
        u2 = self.up2(torch.cat([u1, d4], dim=1))
        u3 = self.up3(torch.cat([u2, d3], dim=1))
        u4 = self.up4(torch.cat([u3, d2], dim=1))

        out = self.final(torch.cat([u4, d1], dim=1))
        return out


# -------------------------------------------------------------
# 2. Discriminator: PatchGAN (70x70 receptive field)
# -------------------------------------------------------------
class PatchGANDiscriminator(nn.Module):
    """Evaluates if the generated image is a realistic pairing with the SAR backscatter."""

    def __init__(self, in_channels=6):  # Fused input: SAR (3) + Optical (3)
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),  # Output single-channel grid
        )

    def forward(self, sar, optical):
        x = torch.cat([sar, optical], dim=1)
        return self.model(x)


# -------------------------------------------------------------
# 3. PyTorch Lightning GAN Module
# -------------------------------------------------------------
class SARTranslatorGAN(pl.LightningModule):
    """Pix2Pix GAN Model for SAR to Optical RGB Translation."""

    def __init__(self, lr=2e-4, lambda_l1=100.0):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False  # Manual GAN optimization

        self.gen = SARToOpticalGenerator()
        self.disc = PatchGANDiscriminator()

        self.adversarial_loss = nn.BCEWithLogitsLoss()
        self.l1_loss = nn.L1Loss()

    def forward(self, sar):
        return self.gen(sar)

    def training_step(self, batch, batch_idx):
        sar, real_opt = batch  # Shapes: (B, 3, 256, 256)

        # Normalize real_opt to [-1, 1] range to match Tanh generator output
        real_opt_norm = real_opt * 2.0 - 1.0

        opt_g, opt_d = self.optimizers()

        # ---------------------
        # Train Discriminator
        # ---------------------
        self.toggle_optimizer(opt_d)

        # Real loss
        pred_real = self.disc(sar, real_opt_norm)
        loss_d_real = self.adversarial_loss(pred_real, torch.ones_like(pred_real))

        # Fake loss
        fake_opt = self.gen(sar)
        pred_fake = self.disc(sar, fake_opt.detach())
        loss_d_fake = self.adversarial_loss(pred_fake, torch.zeros_like(pred_fake))

        loss_d = (loss_d_real + loss_d_fake) * 0.5
        self.manual_backward(loss_d)
        opt_d.step()
        opt_d.zero_grad()
        self.untoggle_optimizer(opt_d)

        # ---------------------
        # Train Generator
        # ---------------------
        self.toggle_optimizer(opt_g)

        # Adversarial loss (Generator tries to fool Discriminator)
        pred_fake_g = self.disc(sar, fake_opt)
        loss_g_gan = self.adversarial_loss(pred_fake_g, torch.ones_like(pred_fake_g))

        # Reconstruction L1 loss
        loss_g_l1 = self.l1_loss(fake_opt, real_opt_norm)

        loss_g = loss_g_gan + self.hparams.lambda_l1 * loss_g_l1
        self.manual_backward(loss_g)
        opt_g.step()
        opt_g.zero_grad()
        self.untoggle_optimizer(opt_g)

        # Log metrics
        self.log_dict(
            {"loss_g": loss_g, "loss_g_gan": loss_g_gan, "loss_g_l1": loss_g_l1, "loss_d": loss_d},
            prog_bar=True,
            on_epoch=True,
        )

    def configure_optimizers(self):
        lr = self.hparams.lr
        opt_g = torch.optim.Adam(self.gen.parameters(), lr=lr, betas=(0.5, 0.999))
        opt_d = torch.optim.Adam(self.disc.parameters(), lr=lr, betas=(0.5, 0.999))
        return [opt_g, opt_d], []


# -------------------------------------------------------------
# 4. Dataset for translation (loads matching SAR & RGB bands)
# -------------------------------------------------------------
class TranslationDataset(Dataset):
    """Loads matching pairs of [SAR, Optical RGB] from data directories."""

    def __init__(self, data_dir, stats_path=None):
        self.image_paths = sorted(Path(data_dir).glob("*.npy"))
        self.means = None
        self.stds = None
        if stats_path and Path(stats_path).exists():
            stats = np.load(stats_path)
            self.means = stats[0].astype(np.float32).reshape(-1, 1, 1)
            self.stds = stats[1].astype(np.float32).reshape(-1, 1, 1)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load optical patch
        img_raw = np.load(self.image_paths[idx]).astype(np.float32)
        if img_raw.shape[1:] != (256, 256):
            import cv2

            img_hwc = img_raw.transpose(1, 2, 0)
            img_hwc = cv2.resize(img_hwc, (256, 256), interpolation=cv2.INTER_LINEAR)
            img_raw = img_hwc.transpose(2, 0, 1)

        # Extract RGB channels (B4, B3, B2 are bands 3, 2, 1)
        # Note: B4/B3/B2 are standard indices for Sentinel-2 in GEE
        # In our dataset, bands are: B1-B12 + NDVI, NDWI, NDBI, dummy (16 bands)
        # B4, B3, B2 are indices 3, 2, 1
        rgb = img_raw[[3, 2, 1], :, :]  # Shape: (3, 256, 256)

        # Create matching synthetic SAR from labels or load real SAR if available
        # To avoid circular imports, construct synthetic SAR directly
        # Labels are in sibling "labels" folder
        lbl_path = self.image_paths[idx].parent.parent / "labels" / self.image_paths[idx].name
        if lbl_path.exists():
            label = np.load(lbl_path).astype(np.int64)
            if label.shape != (256, 256):
                import cv2

                label = cv2.resize(
                    label.astype(np.uint8), (256, 256), interpolation=cv2.INTER_NEAREST
                ).astype(np.int64)
        else:
            label = np.zeros((256, 256), dtype=np.int64)

        sar_path = self.image_paths[idx].parent.parent / "sar" / self.image_paths[idx].name
        if sar_path.exists():
            sar = np.load(sar_path).astype(np.float32)
            if sar.shape[1:] != (256, 256):
                import cv2

                sar_hwc = sar.transpose(1, 2, 0)
                sar_hwc = cv2.resize(sar_hwc, (256, 256), interpolation=cv2.INTER_LINEAR)
                sar = sar_hwc.transpose(2, 0, 1)
        else:
            # Construct simple synthetic SAR proxy mapping
            # Urban (0), Forest (1), Cropland (2), Grassland (3), Water (6)
            h, w = label.shape
            vv = np.zeros((h, w), dtype=np.float32)
            vh = np.zeros((h, w), dtype=np.float32)
            # Add synthetic noise
            vv[label == 0] = 0.8  # Urban has high backscatter
            vh[label == 0] = 0.4
            vv[label == 1] = 0.3  # Forest has medium backscatter
            vh[label == 1] = 0.5  # High volume scattering
            vv[label == 6] = 0.1  # Water has very low backscatter
            vh[label == 6] = 0.05

            vv = np.clip(vv + np.random.normal(0, 0.05, (h, w)), 0, 1)
            vh = np.clip(vh + np.random.normal(0, 0.05, (h, w)), 0, 1)
            ratio = vv / (vh + 1e-8)
            sar = np.stack([vv, vh, np.clip(ratio, 0, 1)]).astype(np.float32)

        return torch.from_numpy(sar), torch.from_numpy(rgb)


def train_translator(
    data_dir="data/real/train/images",
    epochs=10,
    batch_size=4,
    out_ckpt="outputs/checkpoints/sar_translator.ckpt",
):
    if not Path(data_dir).exists():
        print(f"❌ Image directory {data_dir} does not exist. Skipping translator training.")
        return

    dataset = TranslationDataset(data_dir)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True
    )

    model = SARTranslatorGAN()
    trainer = pl.Trainer(
        max_epochs=epochs, accelerator="auto", devices=1, enable_checkpointing=False, logger=False
    )

    print("🚀 Starting Pix2Pix SAR-to-Optical Translator Training...")
    trainer.fit(model, dataloader)

    os.makedirs(os.path.dirname(out_ckpt), exist_ok=True)
    torch.save(model.gen.state_dict(), out_ckpt)
    print(f"✅ Translator training complete. Generator saved to: {out_ckpt}")


if __name__ == "__main__":
    train_translator()
