"""
Benchmark module for comparing UNet vs SegFormer.

Trains and evaluates both models on the same data with identical
configuration, then produces a side-by-side comparison report with
per-class IoU charts and summary metrics tables.
"""

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, F1Score, JaccardIndex

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
    """Patch torch.load to default to weights_only=False for local checkpoints."""
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _orig_torch_load(*args, **kwargs)


torch.load = _patched_torch_load


def run_benchmark(
    model_name: str,
    class_counts: np.ndarray,
    data_dir: Path,
    out_dir: Path,
    ckpt_dir: Path,
    num_classes: int,
    num_bands: int,
    class_names: list,
    batch_size: int = 16,
    max_epochs: int = 25,
    patience: int = 10,
    encoder: str = "resnet34",
    lr: float = 5e-4,
    fusion: bool = False,
    num_bands_fusion: int = 0,
) -> dict:
    """Train and evaluate a single model end-to-end.

    Args:
        model_name: Model architecture name (``"unet"`` or ``"segformer"``).
        class_counts: Per-class pixel counts from training data.
        data_dir: Path to the dataset directory.
        out_dir: Root output directory for logs and artifacts.
        ckpt_dir: Directory for saving model checkpoints.
        num_classes: Number of segmentation classes.
        num_bands: Number of optical input bands.
        class_names: List of class name strings.
        batch_size: Training batch size.
        max_epochs: Maximum training epochs.
        patience: Early stopping patience.
        encoder: UNet encoder backbone name.
        lr: Learning rate.
        fusion: Whether to use the SAR+optical fusion dataset.
        num_bands_fusion: Total channel count when fusion is active.

    Returns:
        Dict with keys: model, best_ckpt, overall_accuracy, mean_iou,
        per_class_iou, per_class_f1, train_time_sec.
    """
    try:
        try:
            from ..data.dataset import LandCoverDataset
            from ..training.augmentations import get_train_transforms, get_val_transforms
        except (ImportError, ValueError):
            from src.data.dataset import LandCoverDataset
            from src.training.augmentations import get_train_transforms, get_val_transforms

        if fusion:
            try:
                try:
                    from ..data.fusion_dataset import FusionDataset
                except (ImportError, ValueError):
                    from src.data.fusion_dataset import FusionDataset
                DatasetClass = FusionDataset
                in_channels = num_bands_fusion
            except ImportError:
                print("  Warning: FusionDataset not available, falling back to optical-only.")
                DatasetClass = LandCoverDataset
                in_channels = num_bands
        else:
            DatasetClass = LandCoverDataset
            in_channels = num_bands

        print(
            f"\n  [Benchmark] Training {model_name.upper()} "
            f"({'fusion' if fusion else 'optical'}, {in_channels} channels)"
        )

        stats_path = data_dir / "band_stats.npy"

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
        test_ds = DatasetClass(
            data_dir / "test" / "images",
            data_dir / "test" / "labels",
            stats_path=stats_path,
        )

        import os

        # Use multiple background workers and pinned memory for GPU-accelerated training
        has_gpu = torch.cuda.is_available()
        num_workers = max(2, os.cpu_count() // 2) if has_gpu else 0
        pin_memory = has_gpu

        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(num_workers > 0),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(num_workers > 0),
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(num_workers > 0),
        )

        total = class_counts.sum()
        freq = class_counts.astype(np.float64) / (total + 1e-8)
        weights = 1.0 / (freq + 0.05)
        weights[class_counts == 0] = 0.0
        weights = np.clip(weights / (weights[weights > 0].mean() + 1e-8), 0.2, 3.0).astype(
            np.float32
        )

        if model_name == "segformer":
            try:
                from .segformer_module import SegFormerModule
            except (ImportError, ValueError):
                from src.models.segformer_module import SegFormerModule
            model = SegFormerModule(
                num_classes=num_classes,
                num_bands=in_channels,
                lr=lr,
                class_weights=weights,
            )
        else:
            import importlib

            pipeline = importlib.import_module("run_pipeline")
            model = pipeline.LandCoverModule(
                class_weights=weights,
                in_channels=in_channels,
            )

        model_ckpt_dir = ckpt_dir / model_name
        model_ckpt_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_cb = ModelCheckpoint(
            dirpath=str(model_ckpt_dir),
            filename=f"best-{model_name}" + "-{epoch:02d}-{val_acc:.3f}",
            monitor="val_acc",
            mode="max",
            save_top_k=1,
            verbose=True,
        )
        early_stop_cb = EarlyStopping(
            monitor="val_loss", patience=patience, mode="min", verbose=True
        )
        lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="epoch")

        trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator="auto",
            precision="16-mixed" if has_gpu else "32-true",
            callbacks=[checkpoint_cb, early_stop_cb, lr_monitor],
            default_root_dir=str(out_dir),
            log_every_n_steps=5,
            enable_progress_bar=True,
        )

        t0 = time.time()
        try:
            trainer.fit(model, train_loader, val_loader)
        except Exception as e:
            print(f"\n  Warning: {model_name} training encountered an error: {e}")
            import traceback

            traceback.print_exc()
            print("  Attempting to proceed with evaluation using existing checkpoints...")
        train_time = time.time() - t0

        best_path = checkpoint_cb.best_model_path
        if not best_path or not Path(best_path).exists():
            ckpts = list(model_ckpt_dir.glob("*.ckpt"))
            if ckpts:
                best_path = str(ckpts[-1])
            else:
                raise FileNotFoundError(f"No checkpoints found for {model_name}. Training failed.")

        if model_name == "segformer":
            try:
                from .segformer_module import SegFormerModule
            except (ImportError, ValueError):
                from src.models.segformer_module import SegFormerModule
            eval_model = SegFormerModule.load_from_checkpoint(
                best_path, num_bands=in_channels, strict=False
            )
        else:
            import importlib

            pipeline = importlib.import_module("run_pipeline")
            eval_model = pipeline.LandCoverModule.load_from_checkpoint(best_path, strict=False)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        eval_model = eval_model.to(device)
        eval_model.eval()
        eval_model.freeze()

        iou_metric = JaccardIndex(task="multiclass", num_classes=num_classes, average="none")
        acc_metric = Accuracy(task="multiclass", num_classes=num_classes)
        f1_metric = F1Score(task="multiclass", num_classes=num_classes, average="none")

        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                logits = eval_model(x)
                preds = logits.argmax(dim=1).cpu()
                iou_metric.update(preds, y)
                acc_metric.update(preds, y)
                f1_metric.update(preds, y)

        per_class_iou = iou_metric.compute().numpy()
        overall_acc = acc_metric.compute().item()
        per_class_f1 = f1_metric.compute().numpy()
        mean_iou = per_class_iou.mean()

        result = {
            "model": model_name,
            "best_ckpt": str(best_path),
            "overall_accuracy": round(float(overall_acc), 4),
            "mean_iou": round(float(mean_iou), 4),
            "per_class_iou": {
                class_names[i]: round(float(per_class_iou[i]), 4) for i in range(num_classes)
            },
            "per_class_f1": {
                class_names[i]: round(float(per_class_f1[i]), 4) for i in range(num_classes)
            },
            "train_time_sec": round(train_time, 2),
        }

        print(f"\n  [Benchmark] {model_name.upper()} Results:")
        print(f"    Overall Accuracy : {overall_acc * 100:.2f}%")
        print(f"    Mean IoU         : {mean_iou * 100:.2f}%")
        print(f"    Training Time    : {train_time / 60:.1f} min")

        return result

    except Exception as e:
        print(f"\n  ERROR: Benchmark for {model_name} failed: {e}")
        import traceback

        traceback.print_exc()
        return {
            "model": model_name,
            "best_ckpt": "",
            "overall_accuracy": 0.0,
            "mean_iou": 0.0,
            "per_class_iou": {name: 0.0 for name in class_names},
            "per_class_f1": {name: 0.0 for name in class_names},
            "train_time_sec": 0.0,
        }


def save_benchmark_report(
    results: list,
    class_names: list,
    report_dir: Path,
    map_dir: Path,
) -> None:
    """Save benchmark results as JSON and a comparison figure.

    Produces:
        1. ``report_dir/benchmark_results.json`` — full results for all models.
        2. ``map_dir/benchmark_comparison.png`` — per-class IoU bar chart and summary table.

    Args:
        results: List of result dicts from ``run_benchmark``.
        class_names: List of class name strings.
        report_dir: Directory for JSON report output.
        map_dir: Directory for PNG figure output.
    """
    try:
        report_dir.mkdir(parents=True, exist_ok=True)
        map_dir.mkdir(parents=True, exist_ok=True)

        report_path = report_dir / "benchmark_results.json"
        with open(report_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  [Benchmark] JSON report saved to {report_path}")

        fig, (ax_bar, ax_table) = plt.subplots(
            1, 2, figsize=(18, 7), gridspec_kw={"width_ratios": [2, 1]}
        )

        n_classes = len(class_names)
        x = np.arange(n_classes)
        bar_width = 0.35
        colors = ["#4285F4", "#FF7043"]

        for idx, result in enumerate(results):
            iou_values = [result["per_class_iou"].get(name, 0.0) * 100 for name in class_names]
            label = f"{result['model'].upper()} (mIoU: {result['mean_iou'] * 100:.1f}%)"
            ax_bar.bar(
                x + idx * bar_width,
                iou_values,
                bar_width,
                label=label,
                color=colors[idx % 2],
                alpha=0.85,
            )
            ax_bar.axhline(
                y=result["mean_iou"] * 100,
                color=colors[idx % 2],
                linestyle="--",
                alpha=0.5,
                linewidth=1,
            )

        short_names = [name[:6] for name in class_names]
        ax_bar.set_xticks(x + bar_width / 2)
        ax_bar.set_xticklabels(short_names, rotation=45, ha="right", fontsize=9)
        ax_bar.set_ylabel("IoU (%)")
        ax_bar.set_ylim(0, 100)
        ax_bar.set_title("Per-Class IoU Comparison", fontsize=13, fontweight="bold")
        ax_bar.legend(loc="upper right", fontsize=9)
        ax_bar.grid(axis="y", alpha=0.3)

        ax_table.axis("off")

        if len(results) >= 2:
            r0, r1 = results[0], results[1]
            row_labels = ["Overall Accuracy", "Mean IoU", "Training Time"]
            col_labels = [r0["model"].upper(), r1["model"].upper()]
            cell_text = [
                [f"{r0['overall_accuracy'] * 100:.1f}%", f"{r1['overall_accuracy'] * 100:.1f}%"],
                [f"{r0['mean_iou'] * 100:.1f}%", f"{r1['mean_iou'] * 100:.1f}%"],
                [f"{r0['train_time_sec'] / 60:.1f} min", f"{r1['train_time_sec'] / 60:.1f} min"],
            ]

            cell_colors = []
            comparisons = [
                (r0["overall_accuracy"], r1["overall_accuracy"]),
                (r0["mean_iou"], r1["mean_iou"]),
                (r1["train_time_sec"], r0["train_time_sec"]),
            ]
            for v0, v1 in comparisons:
                if v0 > v1:
                    cell_colors.append(["#C8E6C9", "#FFCDD2"])
                elif v1 > v0:
                    cell_colors.append(["#FFCDD2", "#C8E6C9"])
                else:
                    cell_colors.append(["#E0E0E0", "#E0E0E0"])

            table = ax_table.table(
                cellText=cell_text,
                rowLabels=row_labels,
                colLabels=col_labels,
                cellColours=cell_colors,
                loc="center",
                cellLoc="center",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(11)
            table.scale(1.2, 1.8)
            ax_table.set_title("Summary Metrics", fontsize=13, fontweight="bold", pad=20)
        else:
            r = results[0]
            ax_table.text(
                0.5,
                0.5,
                f"{r['model'].upper()}\n"
                f"Acc: {r['overall_accuracy'] * 100:.1f}%\n"
                f"mIoU: {r['mean_iou'] * 100:.1f}%\n"
                f"Time: {r['train_time_sec'] / 60:.1f} min",
                ha="center",
                va="center",
                fontsize=14,
                transform=ax_table.transAxes,
            )

        plt.tight_layout()
        fig_path = map_dir / "benchmark_comparison.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  [Benchmark] Comparison figure saved to {fig_path}")

    except Exception as e:
        print(f"\n  ERROR: Failed to save benchmark report: {e}")
