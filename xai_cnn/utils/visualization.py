"""Visualisation helpers for attribution maps.

Everything renders through matplotlib and returns / saves figures so results
are reproducible from scripts. The key entry points are:

* :func:`overlay_heatmap`  -- blend a single attribution map onto the image.
* :func:`comparison_grid`  -- lay out many methods side by side for one image.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import torch

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None


def tensor_to_image(x: torch.Tensor, inverse_transform=None) -> np.ndarray:
    """Convert a ``(1,C,H,W)`` or ``(C,H,W)`` tensor to a HxWx3 uint8-free array."""
    if x.dim() == 4:
        x = x[0]
    img = x.detach().cpu()
    if inverse_transform is not None:
        img = inverse_transform(img)
    img = img.permute(1, 2, 0).numpy()
    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    return np.clip(img, 0, 1)


def _to_numpy(cam: torch.Tensor) -> np.ndarray:
    return cam.detach().cpu().numpy()


def overlay_heatmap(image: np.ndarray, cam: torch.Tensor, alpha: float = 0.5,
                    cmap: str = "jet") -> np.ndarray:
    """Blend a normalised attribution map over an image.

    Parameters
    ----------
    image:
        HxWx3 float array in ``[0, 1]``.
    cam:
        HxW attribution map in ``[0, 1]``.
    alpha:
        Heat-map opacity.
    """
    if plt is None:
        raise ImportError("matplotlib is required for visualisation.")
    heat = plt.get_cmap(cmap)(_to_numpy(cam))[..., :3]
    return np.clip((1 - alpha) * image + alpha * heat, 0, 1)


def comparison_grid(image: np.ndarray, attributions: Dict[str, torch.Tensor],
                    title: Optional[str] = None, alpha: float = 0.5,
                    save_path: Optional[str] = None, cols: int = 4):
    """Render the original image plus one overlaid panel per method.

    Parameters
    ----------
    attributions:
        Mapping ``method_name -> (H,W) map``.
    """
    if plt is None:
        raise ImportError("matplotlib is required for visualisation.")

    panels = [("Original", None)] + list(attributions.items())
    n = len(panels)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.0, rows * 3.2))
    axes = np.atleast_1d(axes).ravel()

    for ax, (label, cam) in zip(axes, panels):
        if cam is None:
            ax.imshow(image)
        else:
            ax.imshow(overlay_heatmap(image, cam, alpha=alpha))
        ax.set_title(label, fontsize=10)
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=13)
    # Explicit spacing rather than tight_layout to avoid panel overlap.
    fig.subplots_adjust(left=0.02, right=0.98, top=0.90 if title else 0.96,
                        bottom=0.02, wspace=0.10, hspace=0.18)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def save_single(image: np.ndarray, cam: torch.Tensor, save_path: str,
                title: str = "", alpha: float = 0.5) -> None:
    """Save one overlay next to the original for a quick look."""
    if plt is None:
        raise ImportError("matplotlib is required for visualisation.")
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(6.4, 3.4))
    a0.imshow(image); a0.set_title("Input"); a0.axis("off")
    a1.imshow(overlay_heatmap(image, cam, alpha=alpha))
    a1.set_title(title); a1.axis("off")
    fig.subplots_adjust(wspace=0.08, left=0.02, right=0.98, top=0.90, bottom=0.02)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
