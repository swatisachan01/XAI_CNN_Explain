"""Class Activation Mapping (CAM) family.

All CAM methods produce a coarse heat-map from the activations of a chosen
convolutional layer (see :func:`xai_cnn.models.cnn.resolve_target_layer`) and
then upsample it to input resolution. They differ in how the per-channel
weights are computed:

============= ================================================================
Method        Weighting scheme
============= ================================================================
Grad-CAM      global-average-pooled gradients (Selvaraju et al., 2017)
Grad-CAM++    higher-order gradient weighting for better multi-instance maps
Layer-CAM     element-wise positive gradients (fine-grained, any layer)
Score-CAM     gradient-free; channel weight = masked-forward confidence gain
Ablation-CAM  gradient-free; weight = confidence drop when channel is ablated
Eigen-CAM     first principal component of the activation maps (class-agnostic)
============= ================================================================
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ActivationsAndGradients, Explainer, normalise_map


class _GradCAMBase(Explainer):
    """Shared machinery: run a forward/backward pass capturing the target layer."""

    def __init__(self, model: nn.Module, target_layer: nn.Module, device: str = "cpu") -> None:
        super().__init__(model, device)
        self.target_layer = target_layer

    def _forward_backward(self, x: torch.Tensor, target: Optional[int]):
        aag = ActivationsAndGradients(self.model, self.target_layer)
        x = x.to(self.device)
        logits = aag(x)
        if target is None:
            target = int(logits.argmax(dim=1)[0].item())
        score = logits[:, target].sum()
        self.model.zero_grad(set_to_none=True)
        score.backward()
        acts, grads = aag.activations, aag.gradients
        aag.release()
        return acts, grads, target, logits

    def _finalise(self, cam: torch.Tensor, size) -> torch.Tensor:
        cam = F.relu(cam)
        cam = self._upsample(cam, size)
        return normalise_map(cam)


class GradCAM(_GradCAMBase):
    name = "Grad-CAM"

    def attribute(self, x: torch.Tensor, target: Optional[int] = None) -> torch.Tensor:
        acts, grads, _, _ = self._forward_backward(x, target)
        weights = grads.mean(dim=(2, 3), keepdim=True)          # (1,C,1,1)
        cam = (weights * acts).sum(dim=1).squeeze(0)            # (h,w)
        return self._finalise(cam, x.shape[-2:])


class GradCAMPlusPlus(_GradCAMBase):
    name = "Grad-CAM++"

    def attribute(self, x: torch.Tensor, target: Optional[int] = None) -> torch.Tensor:
        acts, grads, _, _ = self._forward_backward(x, target)
        grads2 = grads.pow(2)
        grads3 = grads.pow(3)
        sum_acts = acts.sum(dim=(2, 3), keepdim=True)
        denom = 2 * grads2 + sum_acts * grads3
        denom = torch.where(denom != 0, denom, torch.ones_like(denom))
        alpha = grads2 / denom
        weights = (alpha * F.relu(grads)).sum(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1).squeeze(0)
        return self._finalise(cam, x.shape[-2:])


class LayerCAM(_GradCAMBase):
    name = "Layer-CAM"

    def attribute(self, x: torch.Tensor, target: Optional[int] = None) -> torch.Tensor:
        acts, grads, _, _ = self._forward_backward(x, target)
        weighted = F.relu(grads) * acts        # element-wise, keeps spatial detail
        cam = weighted.sum(dim=1).squeeze(0)
        return self._finalise(cam, x.shape[-2:])


class EigenCAM(_GradCAMBase):
    """Class-agnostic CAM: first principal component of the activation maps."""

    name = "Eigen-CAM"
    requires_grad = False

    def attribute(self, x: torch.Tensor, target: Optional[int] = None) -> torch.Tensor:
        aag = ActivationsAndGradients(self.model, self.target_layer)
        with torch.no_grad():
            aag(x.to(self.device))
        acts = aag.activations.squeeze(0)      # (C,h,w)
        aag.release()
        c, h, w = acts.shape
        flat = acts.reshape(c, h * w)          # (C, hw)
        flat = flat - flat.mean(dim=1, keepdim=True)
        # Leading right singular vector -> projection onto principal component.
        _, _, vh = torch.linalg.svd(flat, full_matrices=False)
        proj = vh[0].reshape(h, w)
        return self._finalise(proj.abs(), x.shape[-2:])


class ScoreCAM(_GradCAMBase):
    """Score-CAM (Wang et al., 2020): gradient-free channel weighting.

    Each activation channel is upsampled, min-max normalised into a soft mask,
    multiplied into the input, and forwarded; the resulting increase in the
    target-class confidence becomes that channel's weight. Costs one forward
    pass per channel, so ``batch_size`` controls memory/compute trade-off.
    """

    name = "Score-CAM"
    requires_grad = False

    def __init__(self, model: nn.Module, target_layer: nn.Module,
                 device: str = "cpu", batch_size: int = 32) -> None:
        super().__init__(model, target_layer, device)
        self.batch_size = batch_size

    def attribute(self, x: torch.Tensor, target: Optional[int] = None) -> torch.Tensor:
        x = x.to(self.device)
        aag = ActivationsAndGradients(self.model, self.target_layer)
        with torch.no_grad():
            logits = aag(x)
            if target is None:
                target = int(logits.argmax(dim=1)[0].item())
            acts = aag.activations             # (1,C,h,w)
        aag.release()

        c = acts.shape[1]
        masks = F.interpolate(acts, size=x.shape[-2:], mode="bilinear",
                              align_corners=False).squeeze(0)   # (C,H,W)
        # Per-channel min-max normalisation.
        flat = masks.view(c, -1)
        lo = flat.min(dim=1, keepdim=True).values
        hi = flat.max(dim=1, keepdim=True).values
        norm = ((flat - lo) / (hi - lo + 1e-8)).view(c, *x.shape[-2:])

        scores = torch.zeros(c, device=self.device)
        with torch.no_grad():
            for start in range(0, c, self.batch_size):
                end = min(start + self.batch_size, c)
                batch_masks = norm[start:end].unsqueeze(1)      # (b,1,H,W)
                masked_in = x * batch_masks                     # broadcast over channel
                out = self.model(masked_in)
                scores[start:end] = F.softmax(out, dim=1)[:, target]

        weights = F.softmax(scores, dim=0).view(c, 1, 1)
        cam = (weights * masks).sum(dim=0)
        return self._finalise(cam, x.shape[-2:])


class AblationCAM(_GradCAMBase):
    """Ablation-CAM (Desai & Ramaswamy, 2020): weight = relative confidence drop.

    For each channel, the activation is zeroed via a forward hook and the drop
    in the target logit measures that channel's importance.
    """

    name = "Ablation-CAM"
    requires_grad = False

    def __init__(self, model: nn.Module, target_layer: nn.Module,
                 device: str = "cpu", batch_size: int = 32) -> None:
        super().__init__(model, target_layer, device)
        self.batch_size = batch_size

    def attribute(self, x: torch.Tensor, target: Optional[int] = None) -> torch.Tensor:
        x = x.to(self.device)
        aag = ActivationsAndGradients(self.model, self.target_layer)
        with torch.no_grad():
            base_logits = aag(x)
            if target is None:
                target = int(base_logits.argmax(dim=1)[0].item())
            base_score = base_logits[0, target]
            acts = aag.activations             # (1,C,h,w)
        aag.release()

        c = acts.shape[1]
        weights = torch.zeros(c, device=self.device)
        # Ablate one channel at a time via a temporary hook.
        for ch in range(c):
            def hook(module, inp, out, ch=ch):
                out = out.clone()
                out[:, ch] = 0.0
                return out

            handle = self.target_layer.register_forward_hook(hook)
            with torch.no_grad():
                score = self.model(x)[0, target]
            handle.remove()
            weights[ch] = (base_score - score) / (base_score + 1e-8)

        cam = (weights.view(c, 1, 1) * acts.squeeze(0)).sum(dim=0)
        return self._finalise(cam, x.shape[-2:])
