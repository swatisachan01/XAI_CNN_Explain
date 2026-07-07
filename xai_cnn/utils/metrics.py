"""Quantitative evaluation of attribution maps.

Interpretability claims are only meaningful if attributions are *faithful* to
the model. This module implements the standard faithfulness and localisation
metrics so different explainers can be compared objectively rather than by eye.

Metrics
-------
* :func:`deletion_auc`  -- progressively remove the most salient pixels; a
  faithful map causes a fast confidence drop, so **lower AUC is better**.
* :func:`insertion_auc` -- progressively insert the most salient pixels onto a
  blurred baseline; **higher AUC is better**.
* :func:`pointing_game` -- does the attribution's peak fall inside a ground-truth
  bounding box? (localisation; needs boxes).
* :func:`sparsity`      -- Gini coefficient of the map (concentrated vs diffuse).

Deletion/Insertion follow Petsiuk et al. (2018).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def _gaussian_blur(x: torch.Tensor, ksize: int = 11, sigma: float = 5.0) -> torch.Tensor:
    """Depthwise Gaussian blur used as the insertion baseline."""
    channels = x.shape[1]
    coords = torch.arange(ksize, dtype=torch.float32, device=x.device) - ksize // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum())
    kernel = torch.outer(g, g).expand(channels, 1, ksize, ksize)
    return F.conv2d(x, kernel, padding=ksize // 2, groups=channels)


def _causal_curve(model, x: torch.Tensor, cam: torch.Tensor, target: int,
                  mode: str, step: int, device: str) -> np.ndarray:
    """Shared deletion/insertion sweep. Returns the probability curve."""
    _, c, h, w = x.shape
    n = h * w
    order = torch.argsort(cam.reshape(-1), descending=True)  # most salient first

    if mode == "deletion":
        start = x.clone()
        finish = torch.zeros_like(x)
    else:  # insertion
        start = _gaussian_blur(x)
        finish = x.clone()

    probs = []
    current = start.clone()
    with torch.no_grad():
        for k in range(0, n + 1, step):
            if k > 0:
                idx = order[k - step:k]
                ys, xs = idx // w, idx % w
                current[:, :, ys, xs] = finish[:, :, ys, xs]
            p = F.softmax(model(current), dim=1)[0, target].item()
            probs.append(p)
    return np.asarray(probs)


def deletion_auc(model, x: torch.Tensor, cam: torch.Tensor,
                 target: Optional[int] = None, step: int = 224,
                 device: str = "cpu") -> Tuple[float, np.ndarray]:
    """Deletion metric. Lower is better. Returns ``(auc, curve)``."""
    x = x.to(device)
    if target is None:
        with torch.no_grad():
            target = int(model(x).argmax(dim=1)[0].item())
    curve = _causal_curve(model, x, cam.to(device), target, "deletion", step, device)
    return float(np.trapz(curve) / (len(curve) - 1)), curve


def insertion_auc(model, x: torch.Tensor, cam: torch.Tensor,
                  target: Optional[int] = None, step: int = 224,
                  device: str = "cpu") -> Tuple[float, np.ndarray]:
    """Insertion metric. Higher is better. Returns ``(auc, curve)``."""
    x = x.to(device)
    if target is None:
        with torch.no_grad():
            target = int(model(x).argmax(dim=1)[0].item())
    curve = _causal_curve(model, x, cam.to(device), target, "insertion", step, device)
    return float(np.trapz(curve) / (len(curve) - 1)), curve


def pointing_game(cam: torch.Tensor, bbox: Tuple[int, int, int, int]) -> bool:
    """Return True if the map's argmax lies inside ``bbox = (x0, y0, x1, y1)``."""
    h, w = cam.shape
    flat = int(cam.reshape(-1).argmax().item())
    py, px = flat // w, flat % w
    x0, y0, x1, y1 = bbox
    return (x0 <= px <= x1) and (y0 <= py <= y1)


def sparsity(cam: torch.Tensor) -> float:
    """Gini coefficient of the attribution map; higher = more concentrated."""
    values = cam.detach().cpu().reshape(-1).abs().numpy()
    if values.sum() == 0:
        return 0.0
    values = np.sort(values)
    n = len(values)
    index = np.arange(1, n + 1)
    return float((np.sum((2 * index - n - 1) * values)) / (n * values.sum()))
