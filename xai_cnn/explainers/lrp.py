"""Layer-wise Relevance Propagation (LRP).

Implements LRP (Bach et al., 2015) for sequential CNNs built from Conv2d,
Linear, ReLU, BatchNorm2d and pooling layers. Relevance from the target logit
is redistributed backwards to the input pixels using the epsilon-rule for most
layers and the z^+ (alpha=1, beta=0) rule option for convolutional layers,
which tends to give cleaner, more human-interpretable maps.

This is a self-contained implementation (no external LRP dependency). It walks
the flattened layer list of the model, caches each layer's input during the
forward pass, and applies the relevance-conservation redistribution on the way
back.

Note
----
LRP assumes a feed-forward stack. It supports :class:`SmallCNN` and torchvision
``vgg*`` out of the box. Networks with residual/skip connections (ResNet,
DenseNet) are *not* strictly supported by this simple walker; use Grad-CAM or
Integrated Gradients for those instead.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Explainer, normalise_map


def _flatten_layers(model: nn.Module) -> List[nn.Module]:
    """Return the ordered list of primitive layers of a sequential CNN."""
    layers: List[nn.Module] = []
    for module in model.children():
        if isinstance(module, nn.Sequential):
            layers.extend(_flatten_layers(module))
        elif list(module.children()):
            layers.extend(_flatten_layers(module))
        else:
            layers.append(module)
    return layers


class LRP(Explainer):
    name = "LRP"
    requires_grad = False

    def __init__(self, model: nn.Module, device: str = "cpu",
                 epsilon: float = 1e-6, use_zplus_for_conv: bool = True) -> None:
        super().__init__(model, device)
        self.epsilon = epsilon
        self.use_zplus = use_zplus_for_conv
        self.layers = _flatten_layers(model)

    # --- relevance rules ------------------------------------------------- #
    def _linear_forward(self, layer: nn.Module, a: torch.Tensor) -> torch.Tensor:
        """Functional forward for Conv2d/Linear with an optional z+ weight rule.

        Implemented functionally (rather than by cloning the module) so the
        original layer's parameters are never mutated or aliased.
        """
        zplus = self.use_zplus
        w = layer.weight.clamp(min=0) if zplus else layer.weight
        b = layer.bias
        if b is not None and zplus:
            b = b.clamp(min=0)
        if isinstance(layer, nn.Conv2d):
            return F.conv2d(a, w, b, stride=layer.stride, padding=layer.padding,
                            dilation=layer.dilation, groups=layer.groups)
        return F.linear(a, w, b)

    def _relprop_layer(self, layer: nn.Module, a: torch.Tensor,
                       R: torch.Tensor) -> torch.Tensor:
        """Redistribute relevance ``R`` at a layer's output back to its input ``a``."""
        a = a.clone().requires_grad_(True)

        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            z = self._linear_forward(layer, a) + self.epsilon
            s = R / z
            (z * s.detach()).sum().backward()
            return (a * a.grad).detach()

        if isinstance(layer, (nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d)):
            z = layer(a) + self.epsilon
            s = R / z
            (z * s.detach()).sum().backward()
            return (a * a.grad).detach()

        if isinstance(layer, nn.BatchNorm2d):
            z = layer(a) + self.epsilon
            s = R / z
            (z * s.detach()).sum().backward()
            return (a * a.grad).detach()

        # ReLU / Dropout / Flatten: relevance passes through unchanged
        # (shape may change for Flatten, handled by caller).
        return R

    def attribute(self, x: torch.Tensor, target: Optional[int] = None) -> torch.Tensor:
        x = x.to(self.device)

        # Forward pass, caching each layer's input activation.
        activations = [x]
        a = x
        for layer in self.layers:
            if isinstance(layer, nn.Flatten) or (
                isinstance(layer, nn.Linear) and a.dim() > 2
            ):
                a = a.flatten(1)
            a = layer(a)
            activations.append(a)
        logits = a
        if target is None:
            target = int(logits.argmax(dim=1)[0].item())

        # Initialise relevance at the target logit.
        R = torch.zeros_like(logits)
        R[0, target] = logits[0, target]

        # Backward relevance redistribution.
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            a_in = activations[i]
            if isinstance(layer, nn.Flatten):
                R = R.view_as(a_in)
                continue
            if isinstance(layer, nn.Linear) and a_in.dim() > 2:
                a_in_flat = a_in.flatten(1)
                R = self._relprop_layer(layer, a_in_flat, R)
                R = R.view_as(a_in)
                continue
            R = self._relprop_layer(layer, a_in, R)

        relevance = R.sum(dim=1).squeeze(0)      # collapse channels -> (H,W)
        return normalise_map(relevance.clamp(min=0))
