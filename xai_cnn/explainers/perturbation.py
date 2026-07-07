"""Perturbation-based attribution methods.

These treat the model as a black box and infer importance from how the output
changes when parts of the input are masked or replaced.

* :class:`Occlusion` -- slide an occluding patch, record the confidence drop.
* :class:`RISE`      -- random binary masks, weight each by the resulting score
  (Petsiuk et al., 2018).
* :class:`LIMEImage` -- superpixel surrogate model (Ribeiro et al., 2016),
  wrapping the ``lime`` library when available with a NumPy fallback.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Explainer, normalise_map


class Occlusion(Explainer):
    """Occlusion sensitivity (Zeiler & Fergus, 2014)."""

    name = "Occlusion"
    requires_grad = False

    def __init__(self, model: nn.Module, device: str = "cpu",
                 patch: int = 8, stride: int = 4, baseline: float = 0.0) -> None:
        super().__init__(model, device)
        self.patch = patch
        self.stride = stride
        self.baseline = baseline

    def attribute(self, x: torch.Tensor, target: Optional[int] = None) -> torch.Tensor:
        x = x.to(self.device)
        _, _, h, w = x.shape
        with torch.no_grad():
            logits = self.model(x)
            if target is None:
                target = int(logits.argmax(dim=1)[0].item())
            base = F.softmax(logits, dim=1)[0, target].item()

        heat = torch.zeros(h, w, device=self.device)
        counts = torch.zeros(h, w, device=self.device)
        with torch.no_grad():
            for top in range(0, h - self.patch + 1, self.stride):
                for left in range(0, w - self.patch + 1, self.stride):
                    occluded = x.clone()
                    occluded[:, :, top:top + self.patch, left:left + self.patch] = self.baseline
                    score = F.softmax(self.model(occluded), dim=1)[0, target].item()
                    drop = base - score
                    heat[top:top + self.patch, left:left + self.patch] += drop
                    counts[top:top + self.patch, left:left + self.patch] += 1
        heat = heat / counts.clamp(min=1)
        return normalise_map(heat.clamp(min=0))


class RISE(Explainer):
    """RISE (Petsiuk, Das & Saenko, 2018): randomised input sampling."""

    name = "RISE"
    requires_grad = False

    def __init__(self, model: nn.Module, device: str = "cpu",
                 n_masks: int = 2000, mask_res: int = 7, prob: float = 0.5,
                 batch_size: int = 100, seed: int = 0) -> None:
        super().__init__(model, device)
        self.n_masks = n_masks
        self.mask_res = mask_res
        self.prob = prob
        self.batch_size = batch_size
        self.seed = seed

    def _generate_masks(self, h: int, w: int) -> torch.Tensor:
        g = torch.Generator().manual_seed(self.seed)
        cell_h = int(np.ceil(h / self.mask_res))
        cell_w = int(np.ceil(w / self.mask_res))
        up_h = (self.mask_res + 1) * cell_h
        up_w = (self.mask_res + 1) * cell_w
        grid = (torch.rand(self.n_masks, 1, self.mask_res, self.mask_res, generator=g)
                < self.prob).float()
        masks = F.interpolate(grid, size=(up_h, up_w), mode="bilinear", align_corners=False)
        out = torch.empty(self.n_masks, 1, h, w)
        for i in range(self.n_masks):
            # Random crop for continuous mask shifting.
            oy = torch.randint(0, cell_h, (1,), generator=g).item()
            ox = torch.randint(0, cell_w, (1,), generator=g).item()
            out[i] = masks[i, :, oy:oy + h, ox:ox + w]
        return out

    def attribute(self, x: torch.Tensor, target: Optional[int] = None) -> torch.Tensor:
        x = x.to(self.device)
        _, _, h, w = x.shape
        masks = self._generate_masks(h, w).to(self.device)

        if target is None:
            with torch.no_grad():
                target = int(self.model(x).argmax(dim=1)[0].item())

        weighted = torch.zeros(h, w, device=self.device)
        with torch.no_grad():
            for start in range(0, self.n_masks, self.batch_size):
                end = min(start + self.batch_size, self.n_masks)
                batch = masks[start:end]                 # (b,1,H,W)
                masked = x * batch                       # broadcast over channels
                scores = F.softmax(self.model(masked), dim=1)[:, target]  # (b,)
                weighted += (scores.view(-1, 1, 1) * batch.squeeze(1)).sum(dim=0)
        weighted /= (self.n_masks * self.prob)
        return normalise_map(weighted)


class LIMEImage(Explainer):
    """LIME for images (Ribeiro, Singh & Guestrin, 2016).

    Uses the ``lime`` package if installed. Otherwise falls back to a compact
    built-in implementation: SLIC superpixels, random on/off perturbations, and
    a ridge-regression surrogate whose coefficients weight each superpixel.
    """

    name = "LIME"
    requires_grad = False

    def __init__(self, model: nn.Module, device: str = "cpu",
                 inverse_transform=None, n_samples: int = 1000,
                 n_segments: int = 50, seed: int = 0) -> None:
        super().__init__(model, device)
        self.inverse_transform = inverse_transform
        self.n_samples = n_samples
        self.n_segments = n_segments
        self.seed = seed

    def attribute(self, x: torch.Tensor, target: Optional[int] = None) -> torch.Tensor:
        try:
            from skimage.segmentation import slic
            from sklearn.linear_model import Ridge
        except ImportError as exc:  # pragma: no cover
            raise ImportError("LIMEImage requires scikit-image and scikit-learn.") from exc

        x = x.to(self.device)
        _, _, h, w = x.shape
        if target is None:
            with torch.no_grad():
                target = int(self.model(x).argmax(dim=1)[0].item())

        # Work in pixel space for segmentation.
        img = x[0].detach().cpu()
        if self.inverse_transform is not None:
            img = self.inverse_transform(img)
        img_np = img.permute(1, 2, 0).numpy()
        if img_np.shape[2] == 1:
            img_np = np.repeat(img_np, 3, axis=2)

        segments = slic(img_np, n_segments=self.n_segments, compactness=10, start_label=0)
        n_seg = int(segments.max()) + 1

        rng = np.random.default_rng(self.seed)
        on_off = rng.integers(0, 2, size=(self.n_samples, n_seg))
        on_off[0] = 1  # keep the original image as one sample

        seg_t = torch.from_numpy(segments).to(self.device)
        scores = np.zeros(self.n_samples, dtype=np.float32)
        with torch.no_grad():
            for i in range(self.n_samples):
                active = torch.from_numpy(on_off[i]).to(self.device)
                mask = active[seg_t].float()                     # (H,W)
                perturbed = x * mask.unsqueeze(0).unsqueeze(0)
                scores[i] = F.softmax(self.model(perturbed), dim=1)[0, target].item()

        # Weight samples by proximity to the original (cosine on the mask vector).
        distances = np.sqrt(((on_off - on_off[0]) ** 2).sum(axis=1))
        weights = np.exp(-(distances ** 2) / (0.25 * n_seg))
        surrogate = Ridge(alpha=1.0)
        surrogate.fit(on_off, scores, sample_weight=weights)

        importance = surrogate.coef_
        heat = torch.zeros(h, w)
        for s in range(n_seg):
            heat[segments == s] = float(importance[s])
        return normalise_map(heat.clamp(min=0).to(self.device))
