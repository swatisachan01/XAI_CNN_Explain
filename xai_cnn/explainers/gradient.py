"""Gradient-based attribution methods.

These methods attribute a prediction to the *input pixels* by differentiating
the target logit with respect to the input (and, in the case of Guided
Backpropagation, by modifying how gradients flow through ReLUs).

Implemented
-----------
* :class:`Saliency`            -- vanilla gradient magnitude (Simonyan et al., 2013).
* :class:`InputXGradient`      -- element-wise input * gradient.
* :class:`IntegratedGradients` -- path integral from a baseline (Sundararajan, 2017).
* :class:`SmoothGrad`          -- gradients averaged over Gaussian-noised inputs.
* :class:`GuidedBackprop`      -- ReLU-masked backprop (Springenberg et al., 2014).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .base import Explainer, normalise_map


def _channel_reduce(grad: torch.Tensor) -> torch.Tensor:
    """Collapse a ``(1,C,H,W)`` gradient into an ``(H,W)`` saliency map.

    We take the maximum absolute value across channels, which is the standard
    choice for RGB saliency and preserves sharp edges better than the mean.
    """
    return grad.abs().amax(dim=1).squeeze(0)


class Saliency(Explainer):
    name = "Saliency"

    def attribute(self, x: torch.Tensor, target: Optional[int] = None) -> torch.Tensor:
        x = x.clone().to(self.device).requires_grad_(True)
        logits = self.model(x)
        score, _ = self._select_target(logits, target)
        self.model.zero_grad(set_to_none=True)
        score.backward()
        return normalise_map(_channel_reduce(x.grad.detach()))


class InputXGradient(Explainer):
    name = "Input x Gradient"

    def attribute(self, x: torch.Tensor, target: Optional[int] = None) -> torch.Tensor:
        x = x.clone().to(self.device).requires_grad_(True)
        logits = self.model(x)
        score, _ = self._select_target(logits, target)
        self.model.zero_grad(set_to_none=True)
        score.backward()
        attr = (x * x.grad).detach()
        return normalise_map(attr.abs().amax(dim=1).squeeze(0))


class IntegratedGradients(Explainer):
    """Integrated Gradients (Sundararajan, Taly & Yan, 2017).

    Approximates the path integral of gradients from a baseline (default: black
    image) to the input using a Riemann sum over ``steps`` interpolations.
    """

    name = "Integrated Gradients"

    def __init__(self, model: nn.Module, device: str = "cpu",
                 steps: int = 50, baseline: Optional[torch.Tensor] = None) -> None:
        super().__init__(model, device)
        self.steps = steps
        self.baseline = baseline

    def attribute(self, x: torch.Tensor, target: Optional[int] = None) -> torch.Tensor:
        x = x.to(self.device)
        baseline = self.baseline if self.baseline is not None else torch.zeros_like(x)
        baseline = baseline.to(self.device)

        if target is None:
            with torch.no_grad():
                target = int(self.model(x).argmax(dim=1)[0].item())

        alphas = torch.linspace(0, 1, self.steps, device=self.device)
        grad_sum = torch.zeros_like(x)
        for a in alphas:
            interp = (baseline + a * (x - baseline)).requires_grad_(True)
            logits = self.model(interp)
            score = logits[:, target].sum()
            self.model.zero_grad(set_to_none=True)
            score.backward()
            grad_sum += interp.grad.detach()

        avg_grad = grad_sum / self.steps
        attr = (x - baseline) * avg_grad
        return normalise_map(attr.abs().amax(dim=1).squeeze(0))


class SmoothGrad(Explainer):
    """SmoothGrad (Smilkov et al., 2017): average saliency over noisy samples."""

    name = "SmoothGrad"

    def __init__(self, model: nn.Module, device: str = "cpu",
                 n_samples: int = 25, noise_level: float = 0.15) -> None:
        super().__init__(model, device)
        self.n_samples = n_samples
        self.noise_level = noise_level

    def attribute(self, x: torch.Tensor, target: Optional[int] = None) -> torch.Tensor:
        x = x.to(self.device)
        if target is None:
            with torch.no_grad():
                target = int(self.model(x).argmax(dim=1)[0].item())

        sigma = self.noise_level * (x.max() - x.min())
        accum = torch.zeros(x.shape[-2:], device=self.device)
        for _ in range(self.n_samples):
            noisy = (x + torch.randn_like(x) * sigma).requires_grad_(True)
            logits = self.model(noisy)
            score = logits[:, target].sum()
            self.model.zero_grad(set_to_none=True)
            score.backward()
            accum += _channel_reduce(noisy.grad.detach())
        return normalise_map(accum / self.n_samples)


class GuidedBackprop(Explainer):
    """Guided Backpropagation (Springenberg et al., 2014).

    Overrides the backward pass of every ReLU so that only positive gradients
    for positive activations propagate. Hooks are installed on ``__init__`` and
    removed by :meth:`release`.
    """

    name = "Guided Backprop"

    def __init__(self, model: nn.Module, device: str = "cpu") -> None:
        super().__init__(model, device)
        self._handles = []
        self._register()

    def _register(self) -> None:
        def relu_backward_hook(module, grad_in, grad_out):
            # Only pass gradients that are positive (guided signal).
            return (torch.clamp(grad_in[0], min=0.0),)

        for module in self.model.modules():
            if isinstance(module, nn.ReLU):
                self._handles.append(module.register_full_backward_hook(relu_backward_hook))

    def attribute(self, x: torch.Tensor, target: Optional[int] = None) -> torch.Tensor:
        x = x.clone().to(self.device).requires_grad_(True)
        logits = self.model(x)
        score, _ = self._select_target(logits, target)
        self.model.zero_grad(set_to_none=True)
        score.backward()
        return normalise_map(_channel_reduce(x.grad.detach()))

    def release(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []
