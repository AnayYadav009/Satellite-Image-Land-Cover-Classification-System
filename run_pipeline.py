"""
Master pipeline: generates data, trains UNet, evaluates, runs change detection.
Fully self-contained — run with: python run_pipeline.py
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

# ── PyTorch 2.6 compatibility ─────────────────────────────────────────
# PyTorch 2.6 changed torch.load to default weights_only=True, which
# blocks numpy types (ndarray, dtype, etc.) during checkpoint loading.
# Since all checkpoints are generated locally by our own code, we
# safely default to weights_only=False to avoid deserialization errors.
_orig_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)


torch.load = _patched_torch_load

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

# ── ensure project root is on sys.path ────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# ✅ FEATURE 4 - MAP: Fix PROJ_DATA to avoid conflicts with other GIS software
if "PROJ_DATA" not in os.environ:
    proj_path = ROOT / ".venv" / "Lib" / "site-packages" / "rasterio" / "proj_data"
    if proj_path.exists():
        os.environ["PROJ_DATA"] = str(proj_path)

# ── constants ─────────────────────────────────────────────────────────
import pytorch_lightning as pl  # noqa: E402
import segmentation_models_pytorch as smp  # noqa: E402
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint  # noqa: E402
from torchmetrics import Accuracy, F1Score, JaccardIndex  # noqa: E402

try:
    # --- Standard package-style imports (for CLI execution) ---
    from src.data.dataset import LandCoverDataset
    from src.data.download_ee import download_bhopal_dataset
    from src.data.download_quickstart import CLASS_NAMES, NUM_CLASSES, create_quickstart_dataset
    from src.data.raster_utils import save_segmentation_geotiff, stitch_patches
    from src.training.augmentations import get_train_transforms, get_val_transforms
except (ImportError, ModuleNotFoundError):
    # --- Fallback for IDEs that treat 'src' as the root directory ---
    sys.path.append(str(ROOT / "src"))
    from data.dataset import LandCoverDataset
    from data.download_ee import download_bhopal_dataset
    from data.download_quickstart import CLASS_NAMES, NUM_CLASSES, create_quickstart_dataset
    from data.raster_utils import save_segmentation_geotiff, stitch_patches
    from training.augmentations import get_train_transforms, get_val_transforms

# ── constants ─────────────────────────────────────────────────────────
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
PATCH_SIZE = 256  # ✅ UPDATED: Increased to 256 for better context
NUM_BANDS = 16
NUM_BANDS_FUSION = NUM_BANDS + 3  # ✅ FEATURE 3 - SAR: optical + VV + VH + VV/VH ratio
BATCH_SIZE = 16  # Increased for better speed
LR = 5e-4  # Peak LR for OneCycle
MAX_EPOCHS = 25  # Accelerated convergence
PATIENCE = 10
ENCODER = "resnet34"  # Balanced speed/capacity
DATA_DIR_QUICK = ROOT / "data" / "quickstart"
DATA_DIR_GEE = ROOT / "data" / "real"
OUT_DIR = ROOT / "outputs"
CKPT_DIR = OUT_DIR / "checkpoints"
MAP_DIR = OUT_DIR / "maps"
REPORT_DIR = OUT_DIR / "reports"

for d in [CKPT_DIR, MAP_DIR, REPORT_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ── reproducibility ──────────────────────────────────────────────────
def seed_everything(seed: int = SEED):
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

        # Stretch RGB (B4, B3, B2) for visualization
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


# ══════════════════════════════════════════════════════════════════════
#  PHASE 1 — DATA GENERATION
# ══════════════════════════════════════════════════════════════════════
def phase_data(mode="quickstart", gee_project=None):
    print("\n" + "=" * 70)
    if mode == "gee":
        print("  PHASE 1 -> Fetching REAL Sentinel-2 Data for Bhopal (GEE)")
        print("=" * 70)
        # Optimized for speed/accuracy balance
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

    # Print class distribution
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

    # Run visual audit
    if (data_dir / "band_stats.npy").exists():
        visual_audit_dataset(data_dir, MAP_DIR / "dataset_audit.png")
    else:
        raise FileNotFoundError(
            f"Missing {data_dir / 'band_stats.npy'}. Data generation or Earth Engine "
            "download failed. If using Earth Engine, ensure you are authenticated."
        )

    return counts, data_dir


# ══════════════════════════════════════════════════════════════════════
#  PHASE 2 — MODEL + LIGHTNING MODULE
# ══════════════════════════════════════════════════════════════════════
class LandCoverModule(pl.LightningModule):
    def __init__(
        self, class_weights=None, in_channels=NUM_BANDS
    ):  # ✅ FEATURE 5 - SEGFORMER: accept dynamic in_channels
        super().__init__()
        # ✅ FIXED BUG 3 (part 1): save in_channels so load_from_checkpoint
        #                          can restore it automatically
        self.save_hyperparameters(ignore=["class_weights"])
        self.model = smp.Unet(
            encoder_name=ENCODER,
            encoder_weights="imagenet",
            in_channels=in_channels,  # ✅ FEATURE 5: dynamic channel count
            classes=NUM_CLASSES,
            activation=None,
        )
        # Combined CE + Dice loss
        # Only compute Dice loss for classes that exist in the training data
        # Focal Loss helps the model focus on "hard" pixels, speeding up accuracy gains
        self.ce_loss = smp.losses.FocalLoss(
            mode="multiclass", alpha=0.25, gamma=2.0, ignore_index=9
        )
        if class_weights is not None:
            # Exclude index 9 (Clouds/Background) from Dice calculation
            valid_classes = [i for i, w in enumerate(class_weights) if w > 0.0 and i != 9]
            self.dice_loss = smp.losses.DiceLoss(
                mode="multiclass", from_logits=True, classes=valid_classes
            )
        else:
            self.dice_loss = smp.losses.DiceLoss(
                mode="multiclass", from_logits=True, ignore_index=9
            )

        self.val_iou = JaccardIndex(task="multiclass", num_classes=NUM_CLASSES, average="macro")
        self.val_acc = Accuracy(task="multiclass", num_classes=NUM_CLASSES)
        self.val_f1 = F1Score(task="multiclass", num_classes=NUM_CLASSES, average="macro")

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch):
        x, y = batch
        logits = self(x)

        # Combined Loss: 0.5 * FocalLoss + 0.5 * DiceLoss  # ✅ FIXED
        ce_loss = self.ce_loss(logits, y)
        dice_loss = self.dice_loss(logits, y)
        loss = 0.5 * ce_loss + 0.5 * dice_loss

        preds = logits.argmax(dim=1)
        return loss, preds, y

    def training_step(self, batch, batch_idx):
        loss, _, _ = self._shared_step(batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # CosineAnnealingLR for stable convergence without depending on loader length
        optimizer = torch.optim.AdamW(self.parameters(), lr=LR, weight_decay=1e-4)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=MAX_EPOCHS,
            eta_min=1e-6,  # ✅ FIXED: Replace OneCycleLR with epoch-based CosineAnnealingLR
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },  # ✅ FIXED: Set interval to epoch
        }

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


# ══════════════════════════════════════════════════════════════════════
#  PHASE 3 — TRAINING
# ══════════════════════════════════════════════════════════════════════
def phase_train(class_counts, data_dir, fusion=False):
    print("\n" + "=" * 70)
    print(
        f"  PHASE 2 -> Training Baseline UNet ({ENCODER.capitalize()} encoder)"
    )  # ✅ FIXED: Use ENCODER constant instead of hardcoded string
    print("=" * 70)
    seed_everything()
    stats_path = data_dir / "band_stats.npy"

    # ✅ FIXED: Use FusionDataset when fusion=True
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

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False
    )

    # Class weights (smoothed inverse frequency)
    total = class_counts.sum()
    freq = class_counts.astype(np.float64) / (total + 1e-8)
    weights = 1.0 / (freq + 0.05)
    weights[class_counts == 0] = 0.0  # Zero weight for empty classes
    weights = np.clip(weights / (weights[weights > 0].mean() + 1e-8), 0.2, 3.0).astype(np.float32)
    print(f"  Class weights: {np.round(weights, 2).tolist()}")

    # ✅ FIXED: Pass in_channels to LandCoverModule
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
    # ✅ ADDED: LR Monitor to track scheduler performance
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="epoch")

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="auto",
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
        print("  Attempting to proceed with evaluation using existing checkpoints...")

    elapsed = time.time() - t0
    print(f"\n  Training phase took {elapsed / 60:.1f} minutes")

    # Robustly find best model path
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


# ✅ FIXED WARNING 1: accept in_channels so fusion checkpoints load
#                     with the correct first conv layer shape
def load_model_for_inference(ckpt_path, model_name="unet", in_channels=NUM_BANDS):
    if model_name == "segformer":
        try:
            from src.models.segformer_module import SegFormerModule
        except (ImportError, ModuleNotFoundError):
            from models.segformer_module import SegFormerModule
        model = SegFormerModule.load_from_checkpoint(ckpt_path, strict=False)
    else:
        model = LandCoverModule.load_from_checkpoint(
            ckpt_path, in_channels=in_channels, strict=False
        )
    model.eval()
    model.freeze()
    return model


# ══════════════════════════════════════════════════════════════════════
#  PHASE 4 — EVALUATION
# ══════════════════════════════════════════════════════════════════════
def phase_evaluate(best_ckpt, data_dir, fusion=False, model_name="unet"):
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
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ✅ FIXED WARNING 1: pass correct channel count for fusion mode
    _in_ch = NUM_BANDS_FUSION if fusion else NUM_BANDS
    model = load_model_for_inference(best_ckpt, model_name=model_name, in_channels=_in_ch)

    # Compute metrics
    iou_metric = JaccardIndex(task="multiclass", num_classes=NUM_CLASSES, average="none")
    acc_metric = Accuracy(task="multiclass", num_classes=NUM_CLASSES)
    f1_metric = F1Score(task="multiclass", num_classes=NUM_CLASSES, average="none")

    confusion = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)
    all_preds, all_labels = [], []

    # ✅ FEATURE 2 - CONFIDENCE: Imports and initialization
    try:
        from src.eval.uncertainty import compute_confidence_map, mc_dropout_uncertainty
    except (ImportError, ModuleNotFoundError):
        from eval.uncertainty import compute_confidence_map, mc_dropout_uncertainty
    all_conf_means = []

    with torch.no_grad():
        for x, y in test_loader:
            logits = model(x)
            preds = logits.argmax(dim=1)
            iou_metric.update(preds, y)
            acc_metric.update(preds, y)
            f1_metric.update(preds, y)
            # Confusion matrix
            for p, t in zip(preds.view(-1), y.view(-1)):
                confusion[t, p] += 1
            all_preds.append(preds)
            all_labels.append(y)

            # ✅ FEATURE 2 - CONFIDENCE: Accumulate confidence for whole test set
            for i in range(logits.shape[0]):
                conf_map = compute_confidence_map(logits[i : i + 1])
                all_conf_means.append(conf_map.mean())

    per_class_iou = iou_metric.compute().numpy()
    overall_acc = acc_metric.compute().item()
    per_class_f1 = f1_metric.compute().numpy()
    mean_iou = per_class_iou.mean()

    # ✅ FEATURE 2 - CONFIDENCE: Mean confidence
    mean_conf_across_test_set = np.mean(all_conf_means)

    print(f"\n  Overall Accuracy : {overall_acc * 100:.2f}%")
    print(f"  Mean IoU         : {mean_iou * 100:.2f}%")
    print(f"  Mean Confidence  : {mean_conf_across_test_set * 100:.2f}%")
    print(f"\n  {'Class':<14s}  {'IoU':>6s}  {'F1':>6s}")
    print(f"  {'-' * 30}")
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name:<14s}  {per_class_iou[i] * 100:5.1f}%  {per_class_f1[i] * 100:5.1f}%")

    # ── Confusion matrix plot ─────────────────────────────────────────
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

    # ── Sample segmentation map ───────────────────────────────────────
    _plot_sample_predictions(model, test_ds)

    # ✅ FEATURE 2 - CONFIDENCE: MC Dropout Uncertainty Analysis
    try:
        n_mc = 4
        print(f"\n  Running MC Dropout Uncertainty for first {n_mc} samples...")
        fig, axes = plt.subplots(n_mc, 3, figsize=(15, 4 * n_mc))
        mc_uncertainties = []

        for i in range(n_mc):
            img, label = test_ds[i]
            mean_pred, uncertainty = mc_dropout_uncertainty(model, img.unsqueeze(0), n_passes=20)
            mc_uncertainties.append(uncertainty.mean())

            # RGB
            rgb = img[[3, 2, 1]].numpy().transpose(1, 2, 0)
            rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
            axes[i, 0].imshow(rgb)
            axes[i, 0].set_title("RGB")
            axes[i, 0].axis("off")

            # Prediction
            axes[i, 1].imshow(mean_pred, cmap=ListedColormap(CLASS_COLORS), vmin=0, vmax=9)
            axes[i, 1].set_title("Mean Prediction (MC)")
            axes[i, 1].axis("off")

            # Uncertainty
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

    # ── Save metrics JSON ─────────────────────────────────────────────
    metrics = {
        "overall_accuracy": round(float(overall_acc), 4),
        "mean_iou": round(float(mean_iou), 4),
        "mean_confidence": round(float(mean_conf_across_test_set), 4),  # ✅ FEATURE 2
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
    # ✅ FEATURE 2 - CONFIDENCE: Import utility
    try:
        from src.eval.uncertainty import compute_confidence_map
    except (ImportError, ModuleNotFoundError):
        from eval.uncertainty import compute_confidence_map

    cmap = ListedColormap(CLASS_COLORS)
    fig, axes = plt.subplots(n_samples, 4, figsize=(20, 4 * n_samples))
    if n_samples == 1:
        axes = axes[np.newaxis, :]

    for i in range(min(n_samples, len(dataset))):
        img, label = dataset[i]
        with torch.no_grad():
            logits = model(img.unsqueeze(0))
            pred = logits.argmax(dim=1).squeeze().numpy()
            conf_map = compute_confidence_map(logits)

        label = label.numpy()

        # Create pseudo-RGB from bands 3, 2, 1 (R, G, B)
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

        # ✅ FEATURE 2 - CONFIDENCE: 4th column for confidence heatmap
        im_conf = axes[i, 3].imshow(conf_map, cmap="RdYlGn", vmin=0, vmax=1)
        axes[i, 3].set_title("Confidence")
        axes[i, 3].axis("off")
        plt.colorbar(im_conf, ax=axes[i, 3], fraction=0.046, pad=0.04)

    # Legend
    patches = [
        mpatches.Patch(color=CLASS_COLORS[i], label=CLASS_NAMES[i]) for i in range(NUM_CLASSES)
    ]
    fig.legend(handles=patches, loc="lower center", ncol=5, fontsize=9, frameon=True)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(MAP_DIR / "sample_predictions.png", dpi=150)
    plt.close(fig)
    print(f"  Sample predictions saved to {MAP_DIR / 'sample_predictions.png'}")


# ══════════════════════════════════════════════════════════════════════
#  PHASE 5 — POST-PROCESSING
# ══════════════════════════════════════════════════════════════════════
def phase_postprocess(best_ckpt, data_dir, fusion=False, model_name="unet"):
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

    # ✅ FIXED WARNING 1: pass correct channel count for fusion mode
    _in_ch = NUM_BANDS_FUSION if fusion else NUM_BANDS
    model = load_model_for_inference(best_ckpt, model_name=model_name, in_channels=_in_ch)

    iou_raw = JaccardIndex(task="multiclass", num_classes=NUM_CLASSES, average="macro")
    iou_post = JaccardIndex(task="multiclass", num_classes=NUM_CLASSES, average="macro")

    selem = disk(2)
    for img, label in test_ds:
        with torch.no_grad():
            pred_raw = model(img.unsqueeze(0)).argmax(dim=1).squeeze()

        iou_raw.update(pred_raw.unsqueeze(0), label.unsqueeze(0))

        # Post-process each class mask
        pred_np = pred_raw.numpy().astype(np.int64)
        cleaned = np.zeros_like(pred_np)
        for c in range(NUM_CLASSES):
            mask = (pred_np == c).astype(np.uint8)
            mask = opening(mask, selem)
            mask = closing(mask, selem)
            # Remove tiny regions
            mask_bool = mask.astype(bool)
            mask_bool = remove_small_objects(mask_bool, min_size=50)
            cleaned[mask_bool] = c

        # Fill any zero-gaps with raw prediction
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


# ══════════════════════════════════════════════════════════════════════
#  PHASE 6 — PATCH STITCHING & GEOTIFF
# ══════════════════════════════════════════════════════════════════════
def phase_stitch(best_ckpt, data_dir, fusion=False, model_name="unet"):
    """✅ ADDED: Phase to reassemble patches into a full scene and save as GeoTIFF."""
    print("\n" + "=" * 70)
    print("  PHASE 6 -> Patch Stitching & Scene Reassembly")
    print("=" * 70)

    # Load model
    # ✅ FIXED WARNING 1: pass correct channel count for fusion mode
    _in_ch = NUM_BANDS_FUSION if fusion else NUM_BANDS
    model = load_model_for_inference(best_ckpt, model_name=model_name, in_channels=_in_ch)

    # Load test set
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

    # Collect predictions
    preds = []
    for i in range(len(test_ds)):
        img, _ = test_ds[i]
        with torch.no_grad():
            out = model(img.unsqueeze(0)).argmax(dim=1).squeeze().numpy()
            preds.append(out)

    # Load original scene shape
    shape_file = data_dir / "test" / "scene_shape.npy"
    if shape_file.exists():
        H, W = np.load(shape_file)
        full_mask = stitch_patches(preds, (H, W), patch_size=PATCH_SIZE)

        # Save as PNG for dashboard
        plt.imsave(
            MAP_DIR / "full_stitched_scene.png",
            full_mask,
            cmap=ListedColormap(CLASS_COLORS),
            vmin=0,
            vmax=9,
        )
        print(f"  Stitched scene saved to {MAP_DIR / 'full_stitched_scene.png'}")

        # ✅ FEATURE 4 - MAP: Reproject, create RGBA overlay, save bounds
        try:
            # Attempt to save a sample GeoTIFF (mock profile if real one doesn't exist)
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

            # 1. Reproject to WGS84
            reproject_to_wgs84(orig_tif, wgs84_tif)

            # 2. Convert to RGBA PNG
            bounds = segmentation_mask_to_rgba_png(wgs84_tif, overlay_png, CLASS_COLORS, alpha=180)

            # 3. Save bounds
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

    model = load_model_for_inference(best_ckpt, model_name=model_name)

    stats = np.load(data_dir / "band_stats.npy")
    means = stats[0].astype(np.float32).reshape(-1, 1, 1)
    stds = stats[1].astype(np.float32).reshape(-1, 1, 1)

    # Generate two "timestamps" for the same AOI
    img_t1, label_t1 = (
        generate_patch(23.18, 77.41) if mode == "gee" else generate_patch(size=256, seed=1000)
    )
    img_t2, label_t2 = (
        generate_patch(23.25, 77.48) if mode == "gee" else generate_patch(size=256, seed=2000)
    )

    # ✅ FIXED: Apply fusion dataset transformation to raw patches if active
    if fusion:
        try:
            from src.data.download_sar import generate_sar_for_patch
        except (ImportError, ModuleNotFoundError):
            from data.download_sar import generate_sar_for_patch
        sar_t1 = torch.from_numpy(generate_sar_for_patch(label_t1, seed=1000))
        sar_t2 = torch.from_numpy(generate_sar_for_patch(label_t2, seed=2000))
        img_t1 = torch.cat([torch.from_numpy(img_t1), sar_t1], dim=0).numpy()
        img_t2 = torch.cat([torch.from_numpy(img_t2), sar_t2], dim=0).numpy()

    # Normalise and predict
    def predict(img_np):
        # ✅ FIXED: Only normalize optical bands (first NUM_BANDS channels)
        optical_norm = (img_np[:NUM_BANDS] - means) / (stds + 1e-8)
        if fusion:
            img_norm = np.concatenate([optical_norm, img_np[NUM_BANDS:]], axis=0)
        else:
            img_norm = optical_norm

        t = torch.from_numpy(img_norm).unsqueeze(0)
        with torch.no_grad():
            logits = model(t)
            return logits.argmax(dim=1).squeeze().numpy()

    pred_t1 = predict(img_t1)
    pred_t2 = predict(img_t2)

    # Binary change map
    change_map = (pred_t1 != pred_t2).astype(np.uint8)
    total_pixels = change_map.size
    changed_pixels = change_map.sum()
    print(
        f"  Changed pixels: {changed_pixels:,} / {total_pixels:,} "
        f"({100 * changed_pixels / total_pixels:.1f}%)"
    )

    # Transition matrix
    transition = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for c1 in range(NUM_CLASSES):
        for c2 in range(NUM_CLASSES):
            transition[c1, c2] = ((pred_t1 == c1) & (pred_t2 == c2)).sum()

    # Area in hectares (10m resolution → 100 m²/pixel → 0.01 ha/pixel)
    area_ha = transition * 0.01

    # Top transitions (off-diagonal)
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

    # Key transitions of policy interest
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

    # ── Plots ─────────────────────────────────────────────────────────
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

    # Transition heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    df = pd.DataFrame(transition, index=CLASS_NAMES, columns=CLASS_NAMES)
    sns.heatmap(df, annot=True, fmt="d", cmap="YlOrRd", ax=ax)
    ax.set_xlabel("To (T2)")
    ax.set_ylabel("From (T1)")
    ax.set_title("Class Transition Matrix (pixel counts)")
    plt.tight_layout()
    fig.savefig(MAP_DIR / "transition_matrix.png", dpi=150)
    plt.close(fig)

    # Save CSV
    df.to_csv(REPORT_DIR / "transition_matrix.csv")
    area_df = pd.DataFrame(area_ha, index=CLASS_NAMES, columns=CLASS_NAMES)
    area_df.to_csv(REPORT_DIR / "transition_area_ha.csv")


# ✅ FEATURE 1 - TIMESERIES: New phase for NDVI monitoring
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

        # 1. Call generate_monthly_ndvi_series() — save result as REPORT_DIR / "ndvi_series.npy"
        ndvi_series = generate_monthly_ndvi_series(region_seed=42)
        np.save(REPORT_DIR / "ndvi_series.npy", ndvi_series)

        # 2. Call detect_ndvi_anomalies() — save result as MAP_DIR / "ndvi_anomaly_map.png"
        anomaly_map = detect_ndvi_anomalies(ndvi_series)
        plt.imsave(MAP_DIR / "ndvi_anomaly_map.png", anomaly_map, cmap="hot_r")

        # 3. Call save_ndvi_animation_frames() into TS_DIR
        save_ndvi_animation_frames(ndvi_series, str(TS_DIR), MONTH_NAMES)

        # 4. Call compute_ndvi_stats() — save as REPORT_DIR / "ndvi_stats.json"
        stats = compute_ndvi_stats(ndvi_series)
        with open(REPORT_DIR / "ndvi_stats.json", "w") as f:
            json.dump(stats, f, indent=2)

        # 5. Call plot_ndvi_curve() — save as MAP_DIR / "ndvi_curve.png"
        plot_ndvi_curve(stats, MONTH_NAMES, str(MAP_DIR / "ndvi_curve.png"))

        # 6. Print: peak month name, trough month name, % anomalous pixels
        print(f"  Peak NDVI Month    : {MONTH_NAMES[stats['peak_month']]}")
        print(f"  Trough NDVI Month  : {MONTH_NAMES[stats['trough_month']]}")
        print(f"  Anomalous Pixels   : {stats['pct_anomalous']:.2f}%")

    except Exception as e:
        print(f"  Warning: NDVI Time-Series phase failed: {e}")


# ✅ FEATURE 3 - SAR: Fusion comparison phase
def phase_fusion_comparison(class_counts, data_dir):
    """
    Trains and evaluates optical-only and fusion (optical+SAR) models,
    then saves comparison plot and results JSON.

    Args:
        class_counts: Per-class pixel counts from training data.
        data_dir: Path to the dataset directory.

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
        best_ckpts = {}

        # ✅ FEATURE 3 - SAR: Train optical-only model
        print("\n  [Fusion] Training OPTICAL-ONLY model...")
        result_optical = run_benchmark(
            model_name="unet",
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

        # ✅ FEATURE 3 - SAR: Train fusion model (optical + SAR)
        print("\n  [Fusion] Training FUSION (Optical + SAR) model...")
        result_fusion = run_benchmark(
            model_name="unet",
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

        # ✅ FEATURE 3 - SAR: Save comparison plot (same format as benchmark)
        _save_fusion_comparison_plot(results, CLASS_NAMES)

        # ✅ FEATURE 3 - SAR: Save results JSON
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
        fig, ax = plt.subplots(figsize=(12, 6))

        n_classes = len(class_names)
        x = np.arange(n_classes)
        bar_width = 0.35
        colors = ["#1976D2", "#43A047"]  # ✅ FEATURE 3 - SAR: blue for optical, green for fusion

        for idx, result in enumerate(results):
            iou_values = [result["per_class_iou"].get(name, 0.0) * 100 for name in class_names]
            label = f"{result['model']} (mIoU: {result['mean_iou'] * 100:.1f}%)"
            ax.bar(
                x + idx * bar_width,
                iou_values,
                bar_width,
                label=label,
                color=colors[idx % 2],
                alpha=0.85,
            )
            ax.axhline(
                y=result["mean_iou"] * 100,
                color=colors[idx % 2],
                linestyle="--",
                alpha=0.5,
                linewidth=1,
            )

        short_names = [name[:6] for name in class_names]
        ax.set_xticks(x + bar_width / 2)
        ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=9)
        ax.set_ylabel("IoU (%)")
        ax.set_ylim(0, 100)
        ax.set_title(
            "Optical-Only vs SAR+Optical Fusion: Per-Class IoU", fontsize=13, fontweight="bold"
        )
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        fig_path = MAP_DIR / "fusion_vs_optical.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  [Fusion] Comparison plot saved to {fig_path}")
    except Exception as e:
        print(f"  Warning: Failed to save fusion comparison plot: {e}")


# ✅ FEATURE 3 - SAR: Cloud recovery simulation phase
def phase_cloud_simulation(best_ckpt_optical, best_ckpt_fusion, data_dir):
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

        # ✅ FEATURE 3 - SAR: Load both models
        if not best_ckpt_optical or not Path(best_ckpt_optical).exists():
            print("  Warning: Optical checkpoint not found. Skipping cloud simulation.")
            return
        if not best_ckpt_fusion or not Path(best_ckpt_fusion).exists():
            print("  Warning: Fusion checkpoint not found. Skipping cloud simulation.")
            return

        model_optical = LandCoverModule.load_from_checkpoint(best_ckpt_optical, strict=False)
        model_optical.eval()
        model_optical.freeze()

        # ✅ FIXED BUG 3 (part 2): pass explicit in_channels so fusion
        #                          checkpoint loads with correct first conv
        model_fusion = LandCoverModule.load_from_checkpoint(
            best_ckpt_fusion,
            in_channels=NUM_BANDS_FUSION,
            strict=False,
        )
        model_fusion.eval()
        model_fusion.freeze()

        # ✅ FEATURE 3 - SAR: Load test dataset (optical only)
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

            # ✅ FEATURE 3 - SAR: 1. Clean prediction (reference)
            with torch.no_grad():
                pred_clean = model_optical(img.unsqueeze(0)).argmax(dim=1).squeeze().numpy()

            # ✅ FEATURE 3 - SAR: 2. Create cloud mask — zero out 100x100 region
            clouded_img = img.clone()
            h, w = img.shape[1], img.shape[2]
            cloud_size = min(100, h, w)
            cy = rng.integers(0, max(1, h - cloud_size))
            cx = rng.integers(0, max(1, w - cloud_size))
            clouded_img[:, cy : cy + cloud_size, cx : cx + cloud_size] = 0.0

            # ✅ FEATURE 3 - SAR: 3. Optical prediction on clouded image
            with torch.no_grad():
                pred_clouded = (
                    model_optical(clouded_img.unsqueeze(0)).argmax(dim=1).squeeze().numpy()
                )

            # ✅ FEATURE 3 - SAR: 4. Fusion prediction on clouded optical + SAR
            try:
                sar = generate_sar_for_patch(label_np, seed=i + 10000)
                sar_tensor = torch.from_numpy(sar).float()
                fused_input = torch.cat([clouded_img, sar_tensor], dim=0)
                with torch.no_grad():
                    pred_fusion = (
                        model_fusion(fused_input.unsqueeze(0)).argmax(dim=1).squeeze().numpy()
                    )
            except Exception as e:
                print(f"  Warning: Fusion prediction failed for patch {i}: {e}")
                pred_fusion = pred_clouded  # fallback

            # ✅ FEATURE 3 - SAR: Compute IoU vs clean reference
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

            # ✅ FEATURE 3 - SAR: 5. Plot 4 columns
            # Col 1: Clean RGB
            rgb = img[[3, 2, 1]].numpy().transpose(1, 2, 0)
            rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
            axes[i, 0].imshow(rgb)
            axes[i, 0].set_title("Clean RGB")
            axes[i, 0].axis("off")

            # Col 2: Clouded RGB
            rgb_cloud = clouded_img[[3, 2, 1]].numpy().transpose(1, 2, 0)
            rgb_cloud = (rgb_cloud - rgb_cloud.min()) / (rgb_cloud.max() - rgb_cloud.min() + 1e-8)
            axes[i, 1].imshow(rgb_cloud)
            axes[i, 1].set_title("Clouded RGB")
            axes[i, 1].axis("off")

            # Col 3: Optical pred (clouded)
            axes[i, 2].imshow(pred_clouded, cmap=cmap, vmin=0, vmax=9)
            axes[i, 2].set_title(f"Optical Pred\n(IoU: {iou_clouded * 100:.1f}%)")
            axes[i, 2].axis("off")

            # Col 4: Fusion pred (clouded + SAR)
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


# ----------------------------------------------------------------------
#  MAIN
# ----------------------------------------------------------------------
def main(
    mode="quickstart", model="unet", fusion=False, gee_project=None
):  # ✅ FEATURE 5 - SEGFORMER: accept model arg  # ✅ FEATURE 3 - SAR: accept fusion arg
    print("+" + "=" * 68 + "+")
    print("|  Sentinel-2 Land Cover Segmentation -- Full Pipeline              |")
    model_label = (
        model.upper() if model != "unet" else f"UNet-{ENCODER.capitalize()}"
    )  # ✅ FEATURE 5 - SEGFORMER: dynamic label
    print(f"|  Mode: {mode.upper():<10s} |  Model: {model_label:<16s} |  10 Classes      |")
    print("+" + "=" * 68 + "+")

    seed_everything()
    t_start = time.time()

    # Clear stale results to prevent dashboard confusion
    stale_map = MAP_DIR / "full_stitched_scene.png"
    if stale_map.exists():
        stale_map.unlink()

    # Phase 1: Data
    class_counts, data_dir = phase_data(mode=mode, gee_project=gee_project)

    # ✅ FEATURE 5 - SEGFORMER: Model selection branching
    if model == "both":
        # ✅ FEATURE 5 - SEGFORMER: Run full benchmark — train and eval both models
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
                    fusion=fusion,  # ✅ FEATURE 3 - SAR: pass fusion flag
                    num_bands_fusion=NUM_BANDS_FUSION
                    if fusion
                    else 0,  # ✅ FEATURE 3 - SAR: pass fusion channel count
                )
                benchmark_results.append(result)

            save_benchmark_report(benchmark_results, CLASS_NAMES, REPORT_DIR, MAP_DIR)
            best_ckpt = benchmark_results[0].get("best_ckpt", "")  # UNet ckpt for downstream phases
            if not best_ckpt:
                raise RuntimeError("Benchmark did not produce a checkpoint.")

            metrics = {
                "overall_accuracy": benchmark_results[0]["overall_accuracy"],
                "mean_iou": benchmark_results[0]["mean_iou"],
                "per_class_iou": benchmark_results[0]["per_class_iou"],
                "per_class_f1": benchmark_results[0]["per_class_f1"],
            }
            downstream_model_name = "unet"
        except Exception as e:
            print(f"\n  ERROR: Benchmark mode failed: {e}")
            print("  Falling back to UNet-only training...")
            best_ckpt = phase_train(class_counts, data_dir, fusion=fusion)
            metrics = phase_evaluate(best_ckpt, data_dir, fusion=fusion, model_name="unet")
            downstream_model_name = "unet"

    elif model == "segformer":
        # ✅ FEATURE 5 - SEGFORMER: Train SegFormer only — mirror existing UNet flow
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
                fusion=fusion,  # ✅ FEATURE 3 - SAR: pass fusion flag
                num_bands_fusion=NUM_BANDS_FUSION
                if fusion
                else 0,  # ✅ FEATURE 3 - SAR: pass fusion channel count
            )
            best_ckpt = result.get("best_ckpt", "")
            if not best_ckpt:
                raise RuntimeError("SegFormer benchmark did not produce a checkpoint.")

            metrics = {
                k: result[k]
                for k in ["overall_accuracy", "mean_iou", "per_class_iou", "per_class_f1"]
            }
            downstream_model_name = "segformer"
        except Exception as e:
            print(f"\n  ERROR: SegFormer training failed: {e}")
            print("  Falling back to UNet-only training...")
            best_ckpt = phase_train(class_counts, data_dir, fusion=fusion)
            metrics = phase_evaluate(best_ckpt, data_dir, fusion=fusion, model_name="unet")
            downstream_model_name = "unet"

    else:
        # ✅ FEATURE 5 - SEGFORMER: existing UNet-only flow — unchanged
        # Phase 2: Training
        best_ckpt = phase_train(class_counts, data_dir, fusion=fusion)
        # Phase 3: Evaluation
        metrics = phase_evaluate(best_ckpt, data_dir, fusion=fusion, model_name="unet")
        downstream_model_name = "unet"

    # Phase 4: Post-processing
    post_metrics = phase_postprocess(
        best_ckpt, data_dir, fusion=fusion, model_name=downstream_model_name
    )

    # Phase 5: Change Detection
    phase_change_detection(
        best_ckpt, data_dir, mode=mode, fusion=fusion, model_name=downstream_model_name
    )

    # ✅ ADDED: Phase 6: Stitching
    phase_stitch(best_ckpt, data_dir, fusion=fusion, model_name=downstream_model_name)

    # ✅ FEATURE 1 - TIMESERIES: Phase 7
    phase_timeseries(data_dir)

    # ✅ FEATURE 3 - SAR: Run fusion comparison and cloud simulation when --fusion is active
    if fusion:
        try:
            ckpt_optical, ckpt_fusion, fusion_results = phase_fusion_comparison(
                class_counts, data_dir
            )
            if ckpt_optical and ckpt_fusion:
                phase_cloud_simulation(ckpt_optical, ckpt_fusion, data_dir)
            else:
                print("  Skipping cloud simulation — fusion training did not produce checkpoints.")
        except Exception as e:
            print(f"  Warning: SAR Fusion phases failed: {e}")

    # ── Summary ───────────────────────────────────────────────────────
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
    )  # ✅ FEATURE 5 - SEGFORMER: model selection CLI arg
    parser.add_argument(
        "--fusion",
        action="store_true",
        default=False,
        help="Enable SAR + Optical fusion (Sentinel-1 + Sentinel-2)",
    )  # ✅ FEATURE 3 - SAR: fusion CLI flag
    parser.add_argument(
        "--gee_project", type=str, default=None, help="Google Cloud Project ID for Earth Engine"
    )
    args = parser.parse_args()
    main(
        mode=args.mode, model=args.model, fusion=args.fusion, gee_project=args.gee_project
    )  # ✅ FEATURE 5 - SEGFORMER: pass model arg  # ✅ FEATURE 3 - SAR: pass fusion arg
