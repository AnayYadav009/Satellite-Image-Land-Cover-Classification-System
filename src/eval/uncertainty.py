"""
Uncertainty estimation utilities for semantic segmentation models.

Provides softmax confidence mapping, Monte Carlo Dropout uncertainty
estimation, and confidence map visualization.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


def compute_confidence_map(logits: torch.Tensor) -> np.ndarray:
    """Compute per-pixel maximum softmax probability from raw logits.

    Args:
        logits: Raw model output of shape (1, C, H, W).

    Returns:
        Float32 numpy array of shape (H, W) with values in [0, 1],
        where 0.0 is completely uncertain and 1.0 is fully confident.
    """
    try:
        probs = F.softmax(logits, dim=1)
        conf, _ = torch.max(probs, dim=1)
        return conf.squeeze().cpu().numpy().astype(np.float32)
    except Exception as e:
        print(f"Error in compute_confidence_map: {e}")
        raise


def mc_dropout_uncertainty(
    model: torch.nn.Module,
    image: torch.Tensor,
    n_passes: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate prediction uncertainty via Monte Carlo Dropout.

    Performs multiple stochastic forward passes with dropout enabled,
    then computes the mean prediction and per-pixel entropy of the
    averaged softmax distribution.

    Args:
        model: Segmentation model (must contain dropout layers).
        image: Input image tensor of shape (1, C, H, W).
        n_passes: Number of stochastic forward passes.

    Returns:
        Tuple of (mean_pred, uncertainty):
            mean_pred: (H, W) int array — argmax of mean softmax across passes.
            uncertainty: (H, W) float32 array — pixel-wise entropy normalized to [0, 1].
    """
    try:
        model.train()
        all_probs = []
        num_classes = None
        dev = next(model.parameters()).device
        image_dev = image.to(dev)
        with torch.no_grad():
            for _ in range(n_passes):
                logits = model(image_dev)
                if num_classes is None:
                    num_classes = logits.shape[1]
                probs = F.softmax(logits, dim=1)
                all_probs.append(probs)

        mean_probs = torch.stack(all_probs).mean(dim=0)
        mean_pred = mean_probs.argmax(dim=1).squeeze().cpu().numpy()

        entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-8), dim=1).squeeze()
        uncertainty = (entropy / np.log(num_classes)).cpu().numpy().astype(np.float32)

        model.eval()
        return mean_pred, uncertainty
    except Exception as e:
        print(f"Error in mc_dropout_uncertainty: {e}")
        raise


def save_confidence_overlay(
    confidence_map: np.ndarray,
    output_path: str,
    colormap: str = "RdYlGn",
) -> None:
    """Save a confidence map as a colored PNG using a matplotlib colormap.

    Args:
        confidence_map: (H, W) float32 array with values in [0, 1].
        output_path: Destination file path for the PNG image.
        colormap: Matplotlib colormap name (default: RdYlGn, red=low, green=high).
    """
    try:
        plt.imsave(output_path, confidence_map, cmap=colormap, vmin=0, vmax=1)
    except Exception as e:
        print(f"Error in save_confidence_overlay: {e}")
        raise
