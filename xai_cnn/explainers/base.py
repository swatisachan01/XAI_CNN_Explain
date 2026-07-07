"""Base classes shared by every explainer.

Design
------
Every explainer subclasses :class:`Explainer` and implements
:meth:`Explainer.attribute`, returning a single-channel saliency/attribution
map with the same spatial size as the input. Keeping the contract narrow means
metrics and visualisation code can treat all methods uniformly.

:class:`ActivationsAndGradients` centralises the forward/backward hook plumbing
that the CAM family relies on, with proper handle cleanup so repeated calls do
not leak hooks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def normalise_map(cam: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Min-max normalise a map (per-sample) into ``[0, 1]``."""
    if cam.dim() == 2:
        cam = cam.unsqueeze(0)
    flat = cam.view(cam.size(0), -1)
    lo = flat.min(dim=1, keepdim=True).values
    hi = flat.max(dim=1, keepdim=True).values
    flat = (flat - lo) / (hi - lo + eps)
    return flat.view_as(cam)


class ActivationsAndGradients:
    """Capture activations and gradients of a target layer via hooks."""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._handles = [
            target_layer.register_forward_hook(self._save_activation),
            # full backward hook receives grad w.r.t. the module output.
            target_layer.register_full_backward_hook(self._save_gradient),
        ]

    def _save_activation(self, module, inp, out) -> None:
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out) -> None:
        self.gradients = grad_out[0].detach()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def release(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []


class Explainer(ABC):
    """Abstract attribution method.

    Parameters
    ----------
    model:
        A CNN in eval mode. The explainer does not change the model's mode; the
        caller is responsible for ``model.eval()``.
    device:
        Torch device string.
    """

    #: Human-readable label used in plots and reports.
    name: str = "explainer"
    #: Whether the method requires gradients (affects ``torch.no_grad`` usage).
    requires_grad: bool = True

    def __init__(self, model: nn.Module, device: str = "cpu") -> None:
        self.model = model
        self.device = device

    @staticmethod
    def _select_target(logits: torch.Tensor, target: Optional[int]) -> torch.Tensor:
        """Return the scalar score to attribute for each sample in the batch."""
        if target is None:
            target = int(logits.argmax(dim=1)[0].item())
        return logits[:, target].sum(), target

    @abstractmethod
    def attribute(self, x: torch.Tensor, target: Optional[int] = None) -> torch.Tensor:
        """Return an attribution map of shape ``(H, W)`` in ``[0, 1]``.

        Parameters
        ----------
        x:
            Input tensor of shape ``(1, C, H, W)`` (single image).
        target:
            Class index to explain; defaults to the predicted class.
        """
        raise NotImplementedError

    # Convenience --------------------------------------------------------- #
    def __call__(self, x: torch.Tensor, target: Optional[int] = None) -> torch.Tensor:
        return self.attribute(x, target)

    def _upsample(self, cam: torch.Tensor, size) -> torch.Tensor:
        """Bilinearly resize a ``(1,1,h,w)`` or ``(h,w)`` map to ``size``."""
        if cam.dim() == 2:
            cam = cam.unsqueeze(0).unsqueeze(0)
        elif cam.dim() == 3:
            cam = cam.unsqueeze(0)
        cam = F.interpolate(cam, size=size, mode="bilinear", align_corners=False)
        return cam.squeeze(0).squeeze(0)
