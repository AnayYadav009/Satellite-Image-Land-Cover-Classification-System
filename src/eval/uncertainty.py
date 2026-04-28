import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


# ✅ FEATURE 2 - CONFIDENCE: Compute max softmax probability
def compute_confidence_map(logits: torch.Tensor) -> np.ndarray:
    """Takes raw logits (1, C, H, W), applies softmax, returns the max
    probability per pixel as a float32 numpy array of shape (H, W).
    Values range 0.0 (completely uncertain) to 1.0 (fully confident)."""
    try:
        probs = F.softmax(logits, dim=1)
        conf, _ = torch.max(probs, dim=1)
        return conf.squeeze().cpu().numpy().astype(np.float32)
    except Exception as e:
        print(f"Error in compute_confidence_map: {e}")
        raise


# ✅ FEATURE 2 - CONFIDENCE: MC Dropout for uncertainty estimation
def mc_dropout_uncertainty(
    model: torch.nn.Module,
    image: torch.Tensor,  # (1, C, H, W)
    n_passes: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Runs n_passes forward passes with model.train() active (enabling dropout).
    Returns:
      mean_pred  — (H, W) int array, argmax of mean softmax across passes
      uncertainty — (H, W) float32 array, pixel-wise entropy of mean softmax
                    entropy = -sum(p * log(p+1e-8)) across classes, normalized
                    to [0, 1] by dividing by log(num_classes)"""
    try:
        model.train()  # Enable dropout
        all_probs = []
        num_classes = None
        with torch.no_grad():
            for _ in range(n_passes):
                logits = model(image)
                if num_classes is None:
                    num_classes = logits.shape[1]
                probs = F.softmax(logits, dim=1)
                all_probs.append(probs)

        mean_probs = torch.stack(all_probs).mean(dim=0)  # (1, C, H, W)
        mean_pred = mean_probs.argmax(dim=1).squeeze().cpu().numpy()

        # Entropy calculation
        entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-8), dim=1).squeeze()
        uncertainty = (entropy / np.log(num_classes)).cpu().numpy().astype(np.float32)

        # ✅ FIXED BUG 1: restore eval mode after MC passes so downstream
        #                 inference is not affected by active dropout
        model.eval()
        return mean_pred, uncertainty
    except Exception as e:
        print(f"Error in mc_dropout_uncertainty: {e}")
        raise


# ✅ FEATURE 2 - CONFIDENCE: Save confidence map as colored PNG
def save_confidence_overlay(
    confidence_map: np.ndarray,  # (H, W) float32, values in [0, 1]
    output_path: str,
    colormap: str = "RdYlGn",  # low confidence=red, high=green
) -> None:
    """Saves confidence map as a colored PNG using matplotlib colormap."""
    try:
        plt.imsave(output_path, confidence_map, cmap=colormap, vmin=0, vmax=1)
    except Exception as e:
        print(f"Error in save_confidence_overlay: {e}")
        raise
