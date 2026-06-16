"""
Master pipeline for the Satellite Image Land-Cover Classification System.

This script coordinates data generation, model training (UNet and SegFormer),
evaluation, change detection, and time-series analysis. It provides a
command-line interface for running the full end-to-end workflow.

Usage:
    python run_pipeline.py --mode quickstart
    python run_pipeline.py --mode gee --model segformer --fusion
"""

import json
import os
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch
from torch.utils.data import DataLoader

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

_orig_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    """Wrapper for torch.load that defaults to weights_only=False."""
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)


torch.load = _patched_torch_load

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

if "PROJ_DATA" not in os.environ:
    proj_path = ROOT / ".venv" / "Lib" / "site-packages" / "rasterio" / "proj_data"
    if proj_path.exists():
        os.environ["PROJ_DATA"] = str(proj_path)

import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torchmetrics import Accuracy, F1Score, JaccardIndex

try:
    from src.data.dataset import LandCoverDataset
    from src.data.download_ee import download_bhopal_dataset
    from src.data.download_quickstart import CLASS_NAMES, NUM_CLASSES, create_quickstart_dataset
    from src.data.raster_utils import save_segmentation_geotiff, stitch_patches
    from src.training.augmentations import get_train_transforms, get_val_transforms
except (ImportError, ModuleNotFoundError):
    sys.path.append(str(ROOT / "src"))
    from data.dataset import LandCoverDataset
    from data.download_ee import download_bhopal_dataset
    from data.download_quickstart import CLASS_NAMES, NUM_CLASSES, create_quickstart_dataset
    from data.raster_utils import save_segmentation_geotiff, stitch_patches
    from training.augmentations import get_train_transforms, get_val_transforms

CLASS_COLORS = [
    "#FF0000",
    "#006400",
    "#FFD700",
    "#7CFC00",
    "#D2B48C",
    "#00CED1",
    "#0000FF",
    "#FFFFFF",
    "#8B4513",
    "#808080",
]
SEED = 42
PATCH_SIZE = 256
NUM_BANDS = 16
NUM_BANDS_FUSION = NUM_BANDS + 3
BATCH_SIZE = 16
LR = 5e-4
MAX_EPOCHS = 25
PATIENCE = 10
ENCODER = "resnet34"
DATA_DIR_QUICK = ROOT / "data" / "quickstart"
DATA_DIR_GEE = ROOT / "data" / "real"
OUT_DIR = ROOT / "outputs"
CKPT_DIR = OUT_DIR / "checkpoints"
MAP_DIR = OUT_DIR / "maps"
REPORT_DIR = OUT_DIR / "reports"

for d in [CKPT_DIR, MAP_DIR, REPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def seed_everything(seed: int = SEED):
    """Set all random seeds for reproducible experiments."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def visual_audit_dataset(data_dir, output_path):
    """Saves a diagnostic plot of raw training patches to verify data quality."""
    img_dir = data_dir / "train" / "images"
    lbl_dir = data_dir / "train" / "labels"

    img_files = sorted(img_dir.glob("*.npy"))[:3]
    lbl_files = sorted(lbl_dir.glob("*.npy"))[:3]

    if not img_files:
        print("  [Audit] No images found for visual audit. Skipping.")
        return

    from matplotlib.colors import ListedColormap

    cmap = ListedColormap(CLASS_COLORS)

    fig, axes = plt.subplots(len(img_files), 2, figsize=(10, 5 * len(img_files)))
    for i, (f_img, f_lbl) in enumerate(zip(img_files, lbl_files)):
        img = np.load(f_img)
        lbl = np.load(f_lbl)

        rgb = img[[3, 2, 1]].transpose(1, 2, 0)
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)

        axes[i, 0].imshow(rgb)
        axes[i, 0].set_title(f"Patch {i} RGB (B4,3,2)")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(lbl, cmap=cmap, vmin=0, vmax=9)
        axes[i, 1].set_title(f"Patch {i} Label Map")
        axes[i, 1].axis("off")

    plt.tight_layout()
    fig.savefig(output_path, dpi=100)
    plt.close(fig)
    print(f"  Diagnostic visual audit saved to {output_path}")


def phase_data(mode="quickstart", gee_project=None):
    """Phase 1: Generate or download the dataset and compute class distribution."""
    print("\n" + "=" * 70)
    if mode == "gee":
        print("  PHASE 1 -> Fetching REAL Sentinel-2 Data for Bhopal (GEE)")
        print("=" * 70)
        download_bhopal_dataset(
            output_dir=str(DATA_DIR_GEE), num_patches=150, project_id=gee_project
        )
        data_dir = DATA_DIR_GEE
    else:
        print("  PHASE 1 -> Generating Quick-Start Synthetic Dataset")
        print("=" * 70)
        create_quickstart_dataset(
            output_dir=str(DATA_DIR_QUICK),
            num_train=60,
            num_val=15,
            num_test=15,
            patch_size=PATCH_SIZE,
            num_bands=NUM_BANDS,
        )
        data_dir = DATA_DIR_QUICK

    lbl_dir = data_dir / "train" / "labels"
    counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    for f in sorted(lbl_dir.glob("*.npy")):
        lbl = np.load(f)
        for c in range(NUM_CLASSES):
            counts[c] += (lbl == c).sum()
    total = counts.sum()
    if total == 0:
        print("\n  ⚠️ Warning: No training data found. Phase 1 likely failed.")
    else:
        print("\n  Class distribution (training set):")
        for i, name in enumerate(CLASS_NAMES):
            pct = 100 * counts[i] / total
            print(f"    {i}: {name:<14s}  {counts[i]:>8d} px  ({pct:5.1f}%)")

    if (data_dir / "band_stats.npy").exists():
        visual_audit_dataset(data_dir, MAP_DIR / "dataset_audit.png")
    else:
        raise FileNotFoundError(
            f"Missing {data_dir / 'band_stats.npy'}. Data generation or Earth Engine "
            "download failed. If using Earth Engine, ensure you are authenticated."
        )

    return counts, data_dir


class LandCoverModule(pl.LightningModule):
    """UNet-based semantic segmentation module with Focal + Dice loss."""

    def __init__(self, class_weights=None, in_channels=NUM_BANDS):
        """Initialize UNet model with combined Focal + Dice loss.

        Args:
            class_weights: Optional per-class weight array.
            in_channels: Number of input bands.
        """
        super().__init__()
        weights_list = (
            class_weights.tolist() if isinstance(class_weights, np.ndarray) else class_weights
        )
        self.save_hyperparameters(ignore=["class_weights"])
        self.hparams.class_weights = weights_list
        self.model = smp.Unet(
            encoder_name=ENCODER,
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=NUM_CLASSES,
            activation=None,
        )
        self.ce_loss = smp.losses.FocalLoss(
            mode="multiclass", alpha=0.25, gamma=2.0, ignore_index=9
        )

        if class_weights is not None:
            valid_classes = [i for i, w in enumerate(class_weights) if w > 0.0 and i != 9]
            self.dice_loss = smp.losses.DiceLoss(
                mode="multiclass", from_logits=True, classes=valid_classes, ignore_index=9
            )
        else:
            self.dice_loss = smp.losses.DiceLoss(
                mode="multiclass", from_logits=True, ignore_index=9
            )

        self.val_iou = JaccardIndex(task="multiclass", num_classes=NUM_CLASSES, average="macro")
        self.val_acc = Accuracy(task="multiclass", num_classes=NUM_CLASSES)
        self.val_f1 = F1Score(task="multiclass", num_classes=NUM_CLASSES, average="macro")

    def forward(self, x):
        """Forward pass through the UNet encoder-decoder."""
        return self.model(x)

    def _shared_step(self, batch):
        """Shared forward and loss computation for train/val steps."""
        x, y = batch
        logits = self(x)

        ce_loss = self.ce_loss(logits, y)
        dice_loss = self.dice_loss(logits, y)
        loss = 0.5 * ce_loss + 0.5 * dice_loss

        preds = logits.argmax(dim=1)
        return loss, preds, y

    def training_step(self, batch, batch_idx):
        """Compute training loss and log."""
        loss, _, _ = self._shared_step(batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """Configure AdamW optimizer with CosineAnnealingLR scheduler."""
        optimizer = torch.optim.AdamW(self.parameters(), lr=LR, weight_decay=1e-4)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=MAX_EPOCHS,
            eta_min=1e-6,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }

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


def phase_train(class_counts, data_dir, fusion=False):
    """Phase 2: Train a UNet model with the given class distribution."""
    print("\n" + "=" * 70)
    print(f"  PHASE 2 -> Training Baseline UNet ({ENCODER.capitalize()} encoder)")
    print("=" * 70)
    seed_everything()
    stats_path = data_dir / "band_stats.npy"

    if fusion:
        try:
            from src.data.fusion_dataset import FusionDataset
        except (ImportError, ModuleNotFoundError):
            from data.fusion_dataset import FusionDataset
        DatasetClass = FusionDataset
        in_channels = NUM_BANDS_FUSION
    else:
        DatasetClass = LandCoverDataset
        in_channels = NUM_BANDS

    train_ds = DatasetClass(
        data_dir / "train" / "images",
        data_dir / "train" / "labels",
        transform=get_train_transforms(),
        stats_path=stats_path,
    )
    val_ds = DatasetClass(
        data_dir / "val" / "images",
        data_dir / "val" / "labels",
        transform=get_val_transforms(),
        stats_path=stats_path,
    )

    import os

    has_gpu = torch.cuda.is_available()
    num_workers = max(2, os.cpu_count() // 2) if has_gpu else 0
    pin_memory = has_gpu

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )

    total = class_counts.sum()
    freq = class_counts.astype(np.float64) / (total + 1e-8)
    weights = 1.0 / (freq + 0.05)
    weights[class_counts == 0] = 0.0
    weights = np.clip(weights / (weights[weights > 0].mean() + 1e-8), 0.2, 3.0).astype(np.float32)
    print(f"  Class weights: {np.round(weights, 2).tolist()}")

    model = LandCoverModule(class_weights=weights, in_channels=in_channels)

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(CKPT_DIR),
        filename="best-unet-{epoch:02d}-{val_acc:.3f}",
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        verbose=True,
    )
    early_stop_cb = EarlyStopping(monitor="val_loss", patience=PATIENCE, mode="min", verbose=True)
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="epoch")

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="auto",
        precision="16-mixed" if has_gpu else "32-true",
        callbacks=[checkpoint_cb, early_stop_cb, lr_monitor],
        default_root_dir=str(OUT_DIR),
        log_every_n_steps=5,
        enable_progress_bar=True,
    )

    t0 = time.time()
    try:
        trainer.fit(model, train_loader, val_loader)
    except Exception as e:
        print(f"\n  Warning: Trainer.fit encountered an error: {e}")
        import traceback

        traceback.print_exc()
        print("  Attempting to proceed with evaluation using existing checkpoints...")

    elapsed = time.time() - t0
    print(f"\n  Training phase took {elapsed / 60:.1f} minutes")

    best_path = checkpoint_cb.best_model_path
    if not best_path or not Path(best_path).exists():
        ckpts = list(CKPT_DIR.glob("*.ckpt"))
        if ckpts:
            best_path = str(ckpts[-1])
            print(f"  Using latest available checkpoint: {best_path}")
        else:
            raise FileNotFoundError("No checkpoints found. Training might have failed completely.")

    print(f"  Proceeding with: {best_path}")
    return best_path


def load_model_for_inference(ckpt_path, model_name="unet", in_channels=NUM_BANDS):
    """Load a trained model from checkpoint for inference with strict shape matching."""
    if model_name == "segformer":
        try:
            from src.models.segformer_module import SegFormerModule
        except (ImportError, ModuleNotFoundError):
            from models.segformer_module import SegFormerModule
        model = SegFormerModule.load_from_checkpoint(ckpt_path, num_bands=in_channels, strict=True)
    else:
        model = LandCoverModule.load_from_checkpoint(
            ckpt_path, in_channels=in_channels, strict=True
        )
    model.eval()
    model.freeze()
    return model


def phase_evaluate(best_ckpt, data_dir, fusion=False, model_name="unet"):
    """Phase 3: Evaluate the trained model and generate diagnostic plots."""
    print("\n" + "=" * 70)
    print("  PHASE 3 -> Evaluation & Error Analysis")
    print("=" * 70)
    seed_everything()
    stats_path = data_dir / "band_stats.npy"

    if fusion:
        try:
            from src.data.fusion_dataset import FusionDataset
        except (ImportError, ModuleNotFoundError):
            from data.fusion_dataset import FusionDataset
        DatasetClass = FusionDataset
    else:
        DatasetClass = LandCoverDataset

    test_ds = DatasetClass(
        data_dir / "test" / "images",
        data_dir / "test" / "labels",
        stats_path=stats_path,
    )

    import os

    has_gpu = torch.cuda.is_available()
    num_workers = max(2, os.cpu_count() // 2) if has_gpu else 0
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=has_gpu,
        persistent_workers=(num_workers > 0),
    )

    _in_ch = NUM_BANDS_FUSION if fusion else NUM_BANDS
    model = load_model_for_inference(best_ckpt, model_name=model_name, in_channels=_in_ch)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    iou_metric = JaccardIndex(task="multiclass", num_classes=NUM_CLASSES, average="none")
    acc_metric = Accuracy(task="multiclass", num_classes=NUM_CLASSES)
    f1_metric = F1Score(task="multiclass", num_classes=NUM_CLASSES, average="none")

    confusion = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)
    all_preds, all_labels = [], []

    try:
        from src.eval.uncertainty import compute_confidence_map, mc_dropout_uncertainty
    except (ImportError, ModuleNotFoundError):
        from eval.uncertainty import compute_confidence_map, mc_dropout_uncertainty
    all_conf_means = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1).cpu()
            y_cpu = y.cpu()
            iou_metric.update(preds, y_cpu)
            acc_metric.update(preds, y_cpu)
            f1_metric.update(preds, y_cpu)
            for p, t in zip(preds.view(-1), y_cpu.view(-1)):
                confusion[t, p] += 1
            all_preds.append(preds)
            all_labels.append(y_cpu)

            for i in range(logits.shape[0]):
                conf_map = compute_confidence_map(logits[i : i + 1].cpu())
                all_conf_means.append(conf_map.mean())

    per_class_iou = iou_metric.compute().numpy()
    overall_acc = acc_metric.compute().item()
    per_class_f1 = f1_metric.compute().numpy()
    mean_iou = per_class_iou.mean()

    mean_conf_across_test_set = np.mean(all_conf_means)

    print(f"\n  Overall Accuracy : {overall_acc * 100:.2f}%")
    print(f"  Mean IoU         : {mean_iou * 100:.2f}%")
    print(f"  Mean Confidence  : {mean_conf_across_test_set * 100:.2f}%")
    print(f"\n  {'Class':<14s}  {'IoU':>6s}  {'F1':>6s}")
    print(f"  {'-' * 30}")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name:<14s}  {per_class_iou[i] * 100:5.1f}%  {per_class_f1[i] * 100:5.1f}%")

    conf_norm = confusion.float() / (confusion.sum(dim=1, keepdim=True) + 1e-8)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        conf_norm.numpy(),
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Normalised Confusion Matrix")
    plt.tight_layout()
    fig.savefig(MAP_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    print(f"\n  Confusion matrix saved to {MAP_DIR / 'confusion_matrix.png'}")

    _plot_sample_predictions(model, test_ds)

    try:
        n_mc = 4
        print(f"\n  Running MC Dropout Uncertainty for first {n_mc} samples...")
        fig, axes = plt.subplots(n_mc, 3, figsize=(15, 4 * n_mc))
        mc_uncertainties = []

        for i in range(n_mc):
            img, label = test_ds[i]
            mean_pred, uncertainty = mc_dropout_uncertainty(model, img.unsqueeze(0), n_passes=20)
            mc_uncertainties.append(uncertainty.mean())

            rgb = img[[3, 2, 1]].numpy().transpose(1, 2, 0)
            rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
            axes[i, 0].imshow(rgb)
            axes[i, 0].set_title("RGB")
            axes[i, 0].axis("off")

            axes[i, 1].imshow(mean_pred, cmap=ListedColormap(CLASS_COLORS), vmin=0, vmax=9)
            axes[i, 1].set_title("Mean Prediction (MC)")
            axes[i, 1].axis("off")

            im_u = axes[i, 2].imshow(uncertainty, cmap="inferno", vmin=0, vmax=1)
            axes[i, 2].set_title("Entropy Uncertainty")
            axes[i, 2].axis("off")
            plt.colorbar(im_u, ax=axes[i, 2], fraction=0.046, pad=0.04)

        plt.tight_layout()
        fig.savefig(MAP_DIR / "uncertainty_maps.png", dpi=150)
        plt.close(fig)
        print(f"  Uncertainty maps saved to {MAP_DIR / 'uncertainty_maps.png'}")
        print(f"  Mean Uncertainty (first 4): {np.mean(mc_uncertainties):.4f}")

    except Exception as e:
        print(f"  Warning: MC Dropout evaluation failed: {e}")

    metrics = {
        "overall_accuracy": round(float(overall_acc), 4),
        "mean_iou": round(float(mean_iou), 4),
        "mean_confidence": round(float(mean_conf_across_test_set), 4),
        "per_class_iou": {
            CLASS_NAMES[i]: round(float(per_class_iou[i]), 4) for i in range(NUM_CLASSES)
        },
        "per_class_f1": {
            CLASS_NAMES[i]: round(float(per_class_f1[i]), 4) for i in range(NUM_CLASSES)
        },
    }
    with open(REPORT_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved to {REPORT_DIR / 'metrics.json'}")

    return metrics


def _plot_sample_predictions(model, dataset, n_samples=4):
    """Plot a grid of RGB | Ground Truth | Prediction | Confidence."""
    try:
        from src.eval.uncertainty import compute_confidence_map
    except (ImportError, ModuleNotFoundError):
        from eval.uncertainty import compute_confidence_map

    cmap = ListedColormap(CLASS_COLORS)
    fig, axes = plt.subplots(n_samples, 4, figsize=(20, 4 * n_samples))
    if n_samples == 1:
        axes = axes[np.newaxis, :]

    dev = next(model.parameters()).device
    for i in range(min(n_samples, len(dataset))):
        img, label = dataset[i]
        with torch.no_grad():
            img_dev = img.unsqueeze(0).to(dev)
            logits = model(img_dev).cpu()
            pred = logits.argmax(dim=1).squeeze().numpy()
            conf_map = compute_confidence_map(logits)

        label = label.numpy()

        rgb = img[[3, 2, 1]].numpy().transpose(1, 2, 0)
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)

        axes[i, 0].imshow(rgb)
        axes[i, 0].set_title("Sentinel-2 (pseudo-RGB)")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(label, cmap=cmap, vmin=0, vmax=9)
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(pred, cmap=cmap, vmin=0, vmax=9)
        axes[i, 2].set_title("Prediction")
        axes[i, 2].axis("off")

        im_conf = axes[i, 3].imshow(conf_map, cmap="RdYlGn", vmin=0, vmax=1)
        axes[i, 3].set_title("Confidence")
        axes[i, 3].axis("off")
        plt.colorbar(im_conf, ax=axes[i, 3], fraction=0.046, pad=0.04)

    patches = [
        mpatches.Patch(color=CLASS_COLORS[i], label=CLASS_NAMES[i]) for i in range(NUM_CLASSES)
    ]
    fig.legend(handles=patches, loc="lower center", ncol=5, fontsize=9, frameon=True)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(MAP_DIR / "sample_predictions.png", dpi=150)
    plt.close(fig)
    print(f"  Sample predictions saved to {MAP_DIR / 'sample_predictions.png'}")


def phase_postprocess(best_ckpt, data_dir, fusion=False, model_name="unet"):
    """Phase 4: Apply morphological cleanup and measure mIoU improvement."""
    print("\n" + "=" * 70)
    print("  PHASE 4 -> Post-Processing (Morphological Cleanup)")
    print("=" * 70)
    from skimage.morphology import closing, disk, opening, remove_small_objects

    stats_path = data_dir / "band_stats.npy"
    if fusion:
        try:
            from src.data.fusion_dataset import FusionDataset
        except (ImportError, ModuleNotFoundError):
            from data.fusion_dataset import FusionDataset
        DatasetClass = FusionDataset
    else:
        DatasetClass = LandCoverDataset

    test_ds = DatasetClass(
        data_dir / "test" / "images",
        data_dir / "test" / "labels",
        stats_path=stats_path,
    )

    _in_ch = NUM_BANDS_FUSION if fusion else NUM_BANDS
    model = load_model_for_inference(best_ckpt, model_name=model_name, in_channels=_in_ch)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    iou_raw = JaccardIndex(task="multiclass", num_classes=NUM_CLASSES, average="macro")
    iou_post = JaccardIndex(task="multiclass", num_classes=NUM_CLASSES, average="macro")

    selem = disk(2)
    for img, label in test_ds:
        with torch.no_grad():
            img_dev = img.unsqueeze(0).to(device)
            pred_raw = model(img_dev).argmax(dim=1).squeeze().cpu()

        iou_raw.update(pred_raw.unsqueeze(0), label.unsqueeze(0))

        pred_np = pred_raw.numpy().astype(np.int64)
        cleaned = np.zeros_like(pred_np)
        for c in range(NUM_CLASSES):
            mask = (pred_np == c).astype(np.uint8)
            mask = opening(mask, selem)
            mask = closing(mask, selem)
            mask_bool = mask.astype(bool)
            mask_bool = remove_small_objects(mask_bool, min_size=50)
            cleaned[mask_bool] = c

        unset = cleaned == 0
        if unset.any():
            cleaned[unset] = pred_np[unset]

        pred_post = torch.from_numpy(cleaned)
        iou_post.update(pred_post.unsqueeze(0), label.unsqueeze(0))

    raw_val = iou_raw.compute().item()
    post_val = iou_post.compute().item()
    delta = post_val - raw_val

    print(f"  Raw  mIoU : {raw_val * 100:.2f}%")
    print(f"  Post mIoU : {post_val * 100:.2f}%")
    print(f"  Delta     : {delta * 100:+.2f}%")
    return {"raw_miou": raw_val, "post_miou": post_val, "delta": delta}


def phase_stitch(best_ckpt, data_dir, fusion=False, model_name="unet"):
    """Phase 6: Reassemble predicted test patches into a full scene."""
    print("\n" + "=" * 70)
    print("  PHASE 6 -> Patch Stitching & Scene Reassembly")
    print("=" * 70)

    _in_ch = NUM_BANDS_FUSION if fusion else NUM_BANDS
    model = load_model_for_inference(best_ckpt, model_name=model_name, in_channels=_in_ch)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if fusion:
        try:
            from src.data.fusion_dataset import FusionDataset
        except (ImportError, ModuleNotFoundError):
            from data.fusion_dataset import FusionDataset
        DatasetClass = FusionDataset
    else:
        DatasetClass = LandCoverDataset

    test_ds = DatasetClass(
        data_dir / "test" / "images",
        data_dir / "test" / "labels",
        stats_path=data_dir / "band_stats.npy",
    )

    preds = []
    for i in range(len(test_ds)):
        img, _ = test_ds[i]
        with torch.no_grad():
            img_dev = img.unsqueeze(0).to(device)
            out = model(img_dev).argmax(dim=1).squeeze().cpu().numpy()
            preds.append(out)

    shape_file = data_dir / "test" / "scene_shape.npy"
    if shape_file.exists():
        H, W = np.load(shape_file)
        full_mask = stitch_patches(preds, (H, W), patch_size=PATCH_SIZE)

        plt.imsave(
            MAP_DIR / "full_stitched_scene.png",
            full_mask,
            cmap=ListedColormap(CLASS_COLORS),
            vmin=0,
            vmax=9,
        )
        print(f"  Stitched scene saved to {MAP_DIR / 'full_stitched_scene.png'}")

        try:
            from rasterio.transform import from_origin

            mock_profile = {
                "driver": "GTiff",
                "height": H,
                "width": W,
                "count": 1,
                "dtype": "uint8",
                "crs": "EPSG:4326",
                "transform": from_origin(west=77.0, north=23.0, xsize=0.0001, ysize=0.0001),
            }
            save_segmentation_geotiff(
                full_mask, str(REPORT_DIR / "final_segmentation.tif"), mock_profile
            )

            try:
                from src.vis.map_export import reproject_to_wgs84, segmentation_mask_to_rgba_png
            except (ImportError, ModuleNotFoundError):
                from vis.map_export import reproject_to_wgs84, segmentation_mask_to_rgba_png

            orig_tif = str(REPORT_DIR / "final_segmentation.tif")
            wgs84_tif = str(REPORT_DIR / "final_segmentation_wgs84.tif")
            overlay_png = str(MAP_DIR / "segmentation_overlay.png")
            bounds_json = str(REPORT_DIR / "map_bounds.json")

            reproject_to_wgs84(orig_tif, wgs84_tif)

            bounds = segmentation_mask_to_rgba_png(wgs84_tif, overlay_png, CLASS_COLORS, alpha=180)

            with open(bounds_json, "w") as f:
                json.dump(bounds, f)

            print(f"  Map overlay saved to {overlay_png}")
            print(f"  Map bounds saved to {bounds_json}")

        except Exception as e:
            print(f"  Warning: Map overlay generation failed. {e}")

    else:
        print(f"  Skipping stitching: {shape_file.name} not found.")
        print(
            "  (Full scene stitching only works in Quickstart mode "
            "where a contiguous grid is generated.)"
        )


def phase_change_detection(best_ckpt, data_dir, mode="quickstart", fusion=False, model_name="unet"):
    """Phase 5: Simulate bi-temporal change detection between two patches."""
    print("\n" + "=" * 70)
    print("  PHASE 5 -> Change Detection (T1 vs T2 Simulation)")
    print("=" * 70)

    if mode == "gee":
        try:
            from src.data.download_ee import get_real_data_patch as generate_patch
        except (ImportError, ModuleNotFoundError):
            from data.download_ee import get_real_data_patch as generate_patch
    else:
        try:
            from src.data.download_quickstart import generate_synthetic_patch as generate_patch
        except (ImportError, ModuleNotFoundError):
            from data.download_quickstart import generate_synthetic_patch as generate_patch

    _in_ch = NUM_BANDS_FUSION if fusion else NUM_BANDS
    model = load_model_for_inference(best_ckpt, model_name=model_name, in_channels=_in_ch)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    stats = np.load(data_dir / "band_stats.npy")
    means = stats[0].astype(np.float32).reshape(-1, 1, 1)
    stds = stats[1].astype(np.float32).reshape(-1, 1, 1)

    img_t1, label_t1 = (
        generate_patch(23.18, 77.41) if mode == "gee" else generate_patch(size=256, seed=1000)
    )
    img_t2, label_t2 = (
        generate_patch(23.25, 77.48) if mode == "gee" else generate_patch(size=256, seed=2000)
    )

    if fusion:
        try:
            from src.data.download_sar import generate_sar_for_patch
        except (ImportError, ModuleNotFoundError):
            from data.download_sar import generate_sar_for_patch
        sar_t1 = torch.from_numpy(generate_sar_for_patch(label_t1, seed=1000))
        sar_t2 = torch.from_numpy(generate_sar_for_patch(label_t2, seed=2000))
        img_t1 = torch.cat([torch.from_numpy(img_t1), sar_t1], dim=0).numpy()
        img_t2 = torch.cat([torch.from_numpy(img_t2), sar_t2], dim=0).numpy()

    def predict(img_np):
        optical_norm = (img_np[:NUM_BANDS] - means) / (stds + 1e-8)
        if fusion:
            img_norm = np.concatenate([optical_norm, img_np[NUM_BANDS:]], axis=0)
        else:
            img_norm = optical_norm

        t = torch.from_numpy(img_norm).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(t).cpu()
            return logits.argmax(dim=1).squeeze().numpy()

    pred_t1 = predict(img_t1)
    pred_t2 = predict(img_t2)

    change_map = (pred_t1 != pred_t2).astype(np.uint8)
    total_pixels = change_map.size
    changed_pixels = change_map.sum()
    print(
        f"  Changed pixels: {changed_pixels:,} / {total_pixels:,} "
        f"({100 * changed_pixels / total_pixels:.1f}%)"
    )

    transition = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for c1 in range(NUM_CLASSES):
        for c2 in range(NUM_CLASSES):
            transition[c1, c2] = ((pred_t1 == c1) & (pred_t2 == c2)).sum()

    area_ha = transition * 0.01

    changes = []
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            if i != j and transition[i, j] > 0:
                changes.append((CLASS_NAMES[i], CLASS_NAMES[j], transition[i, j], area_ha[i, j]))
    changes.sort(key=lambda x: x[2], reverse=True)

    print("\n  Top 10 Land-Cover Transitions:")
    print(f"  {'From':<14s} -> {'To':<14s}  {'Pixels':>8s}  {'Area (ha)':>10s}")
    print(f"  {'-' * 52}")
    for frm, to, px, ha in changes[:10]:
        print(f"  {frm:<14s} -> {to:<14s}  {px:>8,}  {ha:>10.1f}")

    print("\n  Key Transitions:")
    key_pairs = [
        (1, 2, "Deforestation (Forest -> Cropland)"),
        (2, 0, "Urbanisation (Cropland -> Urban)"),
        (6, 4, "Water Loss (Water -> Bare Soil)"),
        (5, 4, "Wetland Loss (Wetlands -> Bare Soil)"),
    ]
    for c1, c2, desc in key_pairs:
        px = transition[c1, c2]
        ha = area_ha[c1, c2]
        print(f"    {desc}: {px:,} px ({ha:.1f} ha)")

    cmap = ListedColormap(CLASS_COLORS)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].imshow(pred_t1, cmap=cmap, vmin=0, vmax=9)
    axes[0].set_title("T1 (Pre-monsoon)")
    axes[0].axis("off")
    axes[1].imshow(pred_t2, cmap=cmap, vmin=0, vmax=9)
    axes[1].set_title("T2 (Post-monsoon)")
    axes[1].axis("off")
    axes[2].imshow(change_map, cmap="RdYlGn_r", vmin=0, vmax=1)
    axes[2].set_title(f"Change Map ({100 * changed_pixels / total_pixels:.1f}% changed)")
    axes[2].axis("off")
    patches = [
        mpatches.Patch(color=CLASS_COLORS[i], label=CLASS_NAMES[i]) for i in range(NUM_CLASSES)
    ]
    fig.legend(handles=patches, loc="lower center", ncol=5, fontsize=8)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(MAP_DIR / "change_detection_maps.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 8))
    df = pd.DataFrame(transition, index=CLASS_NAMES, columns=CLASS_NAMES)
    sns.heatmap(df, annot=True, fmt="d", cmap="YlOrRd", ax=ax)
    ax.set_xlabel("To (T2)")
    ax.set_ylabel("From (T1)")
    ax.set_title("Class Transition Matrix (pixel counts)")
    plt.tight_layout()
    fig.savefig(MAP_DIR / "transition_matrix.png", dpi=150)
    plt.close(fig)

    df.to_csv(REPORT_DIR / "transition_matrix.csv")
    area_df = pd.DataFrame(area_ha, index=CLASS_NAMES, columns=CLASS_NAMES)
    area_df.to_csv(REPORT_DIR / "transition_area_ha.csv")


def phase_timeseries(data_dir: Path):
    """Runs the NDVI time-series analysis pipeline."""
    print("\n" + "=" * 70)
    print("  PHASE 7 -> NDVI Time-Series & Anomaly Detection")
    print("=" * 70)

    try:
        try:
            from src.analysis.ndvi_anomaly import plot_ndvi_curve
            from src.data.timeseries import (
                compute_ndvi_stats,
                detect_ndvi_anomalies,
                generate_monthly_ndvi_series,
                save_ndvi_animation_frames,
            )
        except (ImportError, ModuleNotFoundError):
            from analysis.ndvi_anomaly import plot_ndvi_curve
            from data.timeseries import (
                compute_ndvi_stats,
                detect_ndvi_anomalies,
                generate_monthly_ndvi_series,
                save_ndvi_animation_frames,
            )

        MONTH_NAMES = [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]
        TS_DIR = MAP_DIR / "timeseries"
        TS_DIR.mkdir(exist_ok=True)

        ndvi_series = generate_monthly_ndvi_series(region_seed=42)
        np.save(REPORT_DIR / "ndvi_series.npy", ndvi_series)

        anomaly_map = detect_ndvi_anomalies(ndvi_series)
        plt.imsave(MAP_DIR / "ndvi_anomaly_map.png", anomaly_map, cmap="hot_r")

        save_ndvi_animation_frames(ndvi_series, str(TS_DIR), MONTH_NAMES)

        stats = compute_ndvi_stats(ndvi_series)
        with open(REPORT_DIR / "ndvi_stats.json", "w") as f:
            json.dump(stats, f, indent=2)

        plot_ndvi_curve(stats, MONTH_NAMES, str(MAP_DIR / "ndvi_curve.png"))

        print(f"  Peak NDVI Month    : {MONTH_NAMES[stats['peak_month']]}")
        print(f"  Trough NDVI Month  : {MONTH_NAMES[stats['trough_month']]}")
        print(f"  Anomalous Pixels   : {stats['pct_anomalous']:.2f}%")

    except Exception as e:
        print(f"  Warning: NDVI Time-Series phase failed: {e}")


def phase_fusion_comparison(class_counts, data_dir, model_name="unet", main_result=None):
    """
    Trains and evaluates optical-only and fusion (optical+SAR) models,
    then saves comparison plot and results JSON.

    Args:
        class_counts: Per-class pixel counts from training data.
        data_dir: Path to the dataset directory.
        model_name: Architecture to use for the benchmark.

    Returns:
        Tuple of (best_ckpt_optical, best_ckpt_fusion, fusion_results_dict).
    """
    print("\n" + "=" * 70)
    print("  PHASE F1 -> SAR + Optical Fusion Comparison")
    print("=" * 70)

    try:
        try:
            from src.models.benchmark import run_benchmark
        except (ImportError, ModuleNotFoundError):
            from models.benchmark import run_benchmark

        results = []
        if main_result:
            results.append(main_result)

        best_ckpts = {}

        print("\n  [Fusion] Training OPTICAL-ONLY model...")
        result_optical = run_benchmark(
            model_name=model_name,
            class_counts=class_counts,
            data_dir=data_dir,
            out_dir=OUT_DIR,
            ckpt_dir=CKPT_DIR / "fusion_optical",
            num_classes=NUM_CLASSES,
            num_bands=NUM_BANDS,
            class_names=CLASS_NAMES,
            batch_size=BATCH_SIZE,
            max_epochs=MAX_EPOCHS,
            patience=PATIENCE,
            encoder=ENCODER,
            lr=LR,
            fusion=False,
        )
        result_optical["model"] = "Optical-Only"
        results.append(result_optical)
        best_ckpts["optical"] = result_optical["best_ckpt"]

        print("\n  [Fusion] Training FUSION (Optical + SAR) model...")
        result_fusion = run_benchmark(
            model_name=model_name,
            class_counts=class_counts,
            data_dir=data_dir,
            out_dir=OUT_DIR,
            ckpt_dir=CKPT_DIR / "fusion_fused",
            num_classes=NUM_CLASSES,
            num_bands=NUM_BANDS,
            class_names=CLASS_NAMES,
            batch_size=BATCH_SIZE,
            max_epochs=MAX_EPOCHS,
            patience=PATIENCE,
            encoder=ENCODER,
            lr=LR,
            fusion=True,
            num_bands_fusion=NUM_BANDS_FUSION,
        )
        result_fusion["model"] = "Optical+SAR Fusion"
        results.append(result_fusion)
        best_ckpts["fusion"] = result_fusion["best_ckpt"]

        _save_fusion_comparison_plot(results, CLASS_NAMES)

        try:
            import json as _json

            with open(REPORT_DIR / "fusion_results.json", "w") as f:
                _json.dump(results, f, indent=2)
            print(f"  [Fusion] Results saved to {REPORT_DIR / 'fusion_results.json'}")
        except Exception as e:
            print(f"  Warning: Failed to save fusion results JSON: {e}")

        return best_ckpts.get("optical", ""), best_ckpts.get("fusion", ""), results

    except Exception as e:
        print(f"  ERROR: Fusion comparison failed: {e}")
        return "", "", []


def _save_fusion_comparison_plot(results, class_names):
    """
    Saves a grouped bar chart comparing optical-only vs fusion model per-class IoU.

    Args:
        results: List of result dicts from run_benchmark.
        class_names: List of class name strings.
    """
    try:
        fig, ax = plt.subplots(figsize=(14, 7))
        n_classes = len(class_names)
        n_models = len(results)
        x = np.arange(n_classes)
        bar_width = 0.8 / n_models
        colors = ["#1976D2", "#43A047", "#E64A19", "#7B1FA2", "#FBC02D"]

        for idx, result in enumerate(results):
            per_class = result.get("per_class_iou", [])
            if isinstance(per_class, dict):
                iou_values = [per_class.get(name, 0.0) * 100 for name in class_names]
            elif isinstance(per_class, (list, np.ndarray)):
                iou_values = [
                    per_class[i] * 100 if i < len(per_class) else 0.0 for i in range(n_classes)
                ]
            else:
                iou_values = [0.0] * n_classes

            label = f"{result['model']} (mIoU: {result['mean_iou'] * 100:.1f}%)"
            ax.bar(
                x + (idx - n_models / 2 + 0.5) * bar_width,
                iou_values,
                bar_width,
                label=label,
                color=colors[idx % len(colors)],
                alpha=0.85,
            )

        ax.set_ylabel("Intersection over Union (IoU) %")
        ax.set_title("Model Performance Comparison (Per Class)", fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_ylim(0, 100)
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        plt.tight_layout()

        fig_path = MAP_DIR / "fusion_vs_optical.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  [Fusion] Comparison plot saved to {fig_path}")
    except Exception as e:
        print(f"  Warning: Failed to save fusion comparison plot: {e}")


def phase_cloud_simulation(best_ckpt_optical, best_ckpt_fusion, data_dir, model_name="unet"):
    """
    Demonstrates SAR recovery under simulated cloud cover.

    For 4 test patches:
        1. Load optical image
        2. Create cloud mask: randomly zero out a 100x100 region (simulated cloud)
        3. Run optical-only model on clouded image -> pred_clouded
        4. Run fusion model on clouded image + SAR -> pred_fusion
        5. Run optical-only model on clean image -> pred_clean (reference)

    Saves visualization as MAP_DIR / "cloud_recovery.png".

    Args:
        best_ckpt_optical: Path to best optical-only model checkpoint.
        best_ckpt_fusion: Path to best fusion model checkpoint.
        data_dir: Path to the dataset directory.
        model_name: Model architecture name ("unet" or "segformer").
    """
    print("\n" + "=" * 70)
    print("  PHASE F2 -> Cloud Recovery Simulation (SAR vs Optical)")
    print("=" * 70)

    try:
        try:
            from src.data.dataset import LandCoverDataset
            from src.data.download_sar import generate_sar_for_patch
        except (ImportError, ModuleNotFoundError):
            from data.dataset import LandCoverDataset
            from data.download_sar import generate_sar_for_patch

        stats_path = data_dir / "band_stats.npy"

        if not best_ckpt_optical or not Path(best_ckpt_optical).exists():
            print("  Warning: Optical checkpoint not found. Skipping cloud simulation.")
            return
        if not best_ckpt_fusion or not Path(best_ckpt_fusion).exists():
            print("  Warning: Fusion checkpoint not found. Skipping cloud simulation.")
            return

        # Load optical and fusion models using standard shape-safe wrapper
        model_optical = load_model_for_inference(
            best_ckpt_optical, model_name=model_name, in_channels=NUM_BANDS
        )
        model_fusion = load_model_for_inference(
            best_ckpt_fusion, model_name=model_name, in_channels=NUM_BANDS_FUSION
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_optical = model_optical.to(device)
        model_fusion = model_fusion.to(device)

        test_ds = LandCoverDataset(
            data_dir / "test" / "images",
            data_dir / "test" / "labels",
            stats_path=stats_path,
        )

        n_samples = min(4, len(test_ds))
        cmap = ListedColormap(CLASS_COLORS)
        iou_metric_fn = JaccardIndex(task="multiclass", num_classes=NUM_CLASSES, average="macro")

        fig, axes = plt.subplots(n_samples, 4, figsize=(20, 5 * n_samples))
        if n_samples == 1:
            axes = axes[np.newaxis, :]

        rng = np.random.default_rng(42)

        for i in range(n_samples):
            img, label = test_ds[i]
            label_np = label.numpy()

            with torch.no_grad():
                pred_clean = (
                    model_optical(img.unsqueeze(0).to(device)).argmax(dim=1).squeeze().cpu().numpy()
                )

            clouded_img = img.clone()
            h, w = img.shape[1], img.shape[2]
            cloud_size = min(100, h, w)
            cy = rng.integers(0, max(1, h - cloud_size))
            cx = rng.integers(0, max(1, w - cloud_size))
            clouded_img[:, cy : cy + cloud_size, cx : cx + cloud_size] = 0.0

            with torch.no_grad():
                pred_clouded = (
                    model_optical(clouded_img.unsqueeze(0).to(device))
                    .argmax(dim=1)
                    .squeeze()
                    .cpu()
                    .numpy()
                )

            try:
                sar = generate_sar_for_patch(label_np, seed=i + 10000)
                sar_tensor = torch.from_numpy(sar).float()
                fused_input = torch.cat([clouded_img, sar_tensor], dim=0)
                with torch.no_grad():
                    pred_fusion = (
                        model_fusion(fused_input.unsqueeze(0).to(device))
                        .argmax(dim=1)
                        .squeeze()
                        .cpu()
                        .numpy()
                    )
            except Exception as e:
                print(f"  Warning: Fusion prediction failed for patch {i}: {e}")
                pred_fusion = pred_clouded

            iou_clouded = iou_metric_fn(
                torch.from_numpy(pred_clouded).unsqueeze(0),
                torch.from_numpy(pred_clean).unsqueeze(0),
            ).item()
            iou_metric_fn.reset()

            iou_fusion = iou_metric_fn(
                torch.from_numpy(pred_fusion).unsqueeze(0),
                torch.from_numpy(pred_clean).unsqueeze(0),
            ).item()
            iou_metric_fn.reset()

            rgb = img[[3, 2, 1]].numpy().transpose(1, 2, 0)
            rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
            axes[i, 0].imshow(rgb)
            axes[i, 0].set_title("Clean RGB")
            axes[i, 0].axis("off")

            rgb_cloud = clouded_img[[3, 2, 1]].numpy().transpose(1, 2, 0)
            rgb_cloud = (rgb_cloud - rgb_cloud.min()) / (rgb_cloud.max() - rgb_cloud.min() + 1e-8)
            axes[i, 1].imshow(rgb_cloud)
            axes[i, 1].set_title("Clouded RGB")
            axes[i, 1].axis("off")

            axes[i, 2].imshow(pred_clouded, cmap=cmap, vmin=0, vmax=9)
            axes[i, 2].set_title(f"Optical Pred\n(IoU: {iou_clouded * 100:.1f}%)")
            axes[i, 2].axis("off")

            axes[i, 3].imshow(pred_fusion, cmap=cmap, vmin=0, vmax=9)
            axes[i, 3].set_title(f"Fusion Pred\n(IoU: {iou_fusion * 100:.1f}%)")
            axes[i, 3].axis("off")

        plt.suptitle(
            "Cloud Recovery: Optical vs SAR+Optical Fusion", fontsize=14, fontweight="bold"
        )
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        fig_path = MAP_DIR / "cloud_recovery.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  [Cloud] Cloud recovery visualization saved to {fig_path}")

    except Exception as e:
        print(f"  ERROR: Cloud simulation failed: {e}")


def main(mode="quickstart", model="unet", fusion=False, gee_project=None):
    """Run the full segmentation pipeline end-to-end."""
    print("+" + "=" * 68 + "+")
    print("|  Sentinel-2 Land Cover Segmentation -- Full Pipeline              |")
    model_label = model.upper() if model != "unet" else f"UNet-{ENCODER.capitalize()}"
    print(f"|  Mode: {mode.upper():<10s} |  Model: {model_label:<16s} |  10 Classes      |")
    print("+" + "=" * 68 + "+")

    seed_everything()
    t_start = time.time()

    stale_map = MAP_DIR / "full_stitched_scene.png"
    if stale_map.exists():
        stale_map.unlink()

    class_counts, data_dir = phase_data(mode=mode, gee_project=gee_project)

    if model == "both":
        try:
            try:
                from src.models.benchmark import run_benchmark, save_benchmark_report
            except (ImportError, ModuleNotFoundError):
                from models.benchmark import run_benchmark, save_benchmark_report

            benchmark_results = []
            for model_name in ["unet", "segformer"]:
                print(f"\n  Benchmarking: {model_name.upper()}")
                result = run_benchmark(
                    model_name=model_name,
                    class_counts=class_counts,
                    data_dir=data_dir,
                    out_dir=OUT_DIR,
                    ckpt_dir=CKPT_DIR,
                    num_classes=NUM_CLASSES,
                    num_bands=NUM_BANDS,
                    class_names=CLASS_NAMES,
                    batch_size=BATCH_SIZE,
                    max_epochs=MAX_EPOCHS,
                    patience=PATIENCE,
                    encoder=ENCODER,
                    lr=LR,
                    fusion=fusion,
                    num_bands_fusion=NUM_BANDS_FUSION if fusion else 0,
                )
                benchmark_results.append(result)

            save_benchmark_report(benchmark_results, CLASS_NAMES, REPORT_DIR, MAP_DIR)

            # Select the best model (highest mean IoU) between UNet and SegFormer
            best_model_idx = 0
            if len(benchmark_results) > 1:
                if benchmark_results[1]["mean_iou"] > benchmark_results[0]["mean_iou"]:
                    best_model_idx = 1

            best_ckpt = benchmark_results[best_model_idx].get("best_ckpt", "")
            if not best_ckpt:
                raise RuntimeError("Benchmark did not produce a checkpoint.")

            metrics = {
                "overall_accuracy": benchmark_results[best_model_idx]["overall_accuracy"],
                "mean_iou": benchmark_results[best_model_idx]["mean_iou"],
                "per_class_iou": benchmark_results[best_model_idx]["per_class_iou"],
                "per_class_f1": benchmark_results[best_model_idx]["per_class_f1"],
            }
            downstream_model_name = benchmark_results[best_model_idx]["model"]
        except Exception as e:
            print(f"\n  ERROR: Benchmark mode failed: {e}")
            import traceback

            traceback.print_exc()
            print("  Falling back to UNet-only training...")
            best_ckpt = phase_train(class_counts, data_dir, fusion=fusion)
            metrics = phase_evaluate(best_ckpt, data_dir, fusion=fusion, model_name="unet")
            downstream_model_name = "unet"

    elif model == "segformer":
        try:
            try:
                from src.models.benchmark import run_benchmark
            except (ImportError, ModuleNotFoundError):
                from models.benchmark import run_benchmark

            result = run_benchmark(
                model_name="segformer",
                class_counts=class_counts,
                data_dir=data_dir,
                out_dir=OUT_DIR,
                ckpt_dir=CKPT_DIR,
                num_classes=NUM_CLASSES,
                num_bands=NUM_BANDS,
                class_names=CLASS_NAMES,
                batch_size=BATCH_SIZE,
                max_epochs=MAX_EPOCHS,
                patience=PATIENCE,
                encoder=ENCODER,
                lr=LR,
                fusion=fusion,
                num_bands_fusion=NUM_BANDS_FUSION if fusion else 0,
            )
            best_ckpt = result.get("best_ckpt", "")
            if not best_ckpt:
                raise RuntimeError("SegFormer benchmark did not produce a checkpoint.")

            metrics = phase_evaluate(best_ckpt, data_dir, fusion=fusion, model_name="segformer")
            downstream_model_name = "segformer"
        except Exception as e:
            print(f"\n  ERROR: SegFormer training failed: {e}")
            import traceback

            traceback.print_exc()
            print("  Falling back to UNet-only training...")
            best_ckpt = phase_train(class_counts, data_dir, fusion=fusion)
            metrics = phase_evaluate(best_ckpt, data_dir, fusion=fusion, model_name="unet")
            downstream_model_name = "unet"

    else:
        best_ckpt = phase_train(class_counts, data_dir, fusion=fusion)
        metrics = phase_evaluate(best_ckpt, data_dir, fusion=fusion, model_name="unet")
        downstream_model_name = "unet"

    post_metrics = phase_postprocess(
        best_ckpt, data_dir, fusion=fusion, model_name=downstream_model_name
    )

    phase_change_detection(
        best_ckpt, data_dir, mode=mode, fusion=fusion, model_name=downstream_model_name
    )

    phase_stitch(best_ckpt, data_dir, fusion=fusion, model_name=downstream_model_name)

    phase_timeseries(data_dir)

    if fusion:
        try:
            ckpt_optical, ckpt_fusion, fusion_results = phase_fusion_comparison(
                class_counts,
                data_dir,
                model_name=downstream_model_name,
                main_result=(metrics if downstream_model_name == "segformer" else None),
            )
            if ckpt_optical and ckpt_fusion:
                phase_cloud_simulation(
                    ckpt_optical, ckpt_fusion, data_dir, model_name=downstream_model_name
                )
            else:
                print("  Skipping cloud simulation — fusion training did not produce checkpoints.")
        except Exception as e:
            print(f"  Warning: SAR Fusion phases failed: {e}")

    elapsed = time.time() - t_start
    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE                                              ")
    print("=" * 70)
    print(f"  Total time       : {elapsed / 60:6.1f} min{' ' * 37}")
    print(f"  Overall Accuracy  : {metrics['overall_accuracy'] * 100:5.1f}%{' ' * 39}")
    print(f"  Mean IoU          : {metrics['mean_iou'] * 100:5.1f}%{' ' * 39}")
    print(f"  Post-proc Delta mIoU  : {post_metrics['delta'] * 100:+5.2f}%{' ' * 38}")
    print("=" * 70)
    print("  Outputs:                                                       ")
    print(f"    {str(MAP_DIR / 'sample_predictions.png'):<65s}")
    print(f"    {str(MAP_DIR / 'confusion_matrix.png'):<65s}")
    print(f"    {str(MAP_DIR / 'change_detection_maps.png'):<65s}")
    print(f"    {str(MAP_DIR / 'transition_matrix.png'):<65s}")
    print(f"    {str(REPORT_DIR / 'metrics.json'):<65s}")
    print(f"    {str(REPORT_DIR / 'transition_matrix.csv'):<65s}")
    print("=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="quickstart", choices=["quickstart", "gee"])
    parser.add_argument(
        "--model",
        type=str,
        default="unet",
        choices=["unet", "segformer", "both"],
        help="Model architecture to train/evaluate",
    )
    parser.add_argument(
        "--fusion",
        action="store_true",
        default=False,
        help="Enable SAR + Optical fusion (Sentinel-1 + Sentinel-2)",
    )
    parser.add_argument(
        "--gee_project", type=str, default=None, help="Google Cloud Project ID for Earth Engine"
    )
    args = parser.parse_args()
    main(mode=args.mode, model=args.model, fusion=args.fusion, gee_project=args.gee_project)
