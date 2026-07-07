"""Convolutional neural network definitions and helpers.

Provides two routes:

1. ``SmallCNN`` -- a compact, from-scratch network suitable for CIFAR-10 /
   Fashion-MNIST that trains quickly on a laptop GPU (or even CPU). It is
   deliberately simple so that its convolutional stack is easy to inspect and
   attribute against.

2. ``build_pretrained`` -- thin wrappers around ``torchvision`` classifiers
   (ResNet, VGG, DenseNet, MobileNet) with ImageNet weights, for explaining
   real-world images.

The module also exposes :func:`resolve_target_layer`, which returns a sensible
final convolutional layer for CAM-family explainers given only the model.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# From-scratch network
# --------------------------------------------------------------------------- #
class SmallCNN(nn.Module):
    """A compact VGG-style CNN.

    Parameters
    ----------
    num_classes:
        Number of output logits.
    in_channels:
        Number of input channels (3 for RGB, 1 for greyscale).
    """

    def __init__(self, num_classes: int = 10, in_channels: int = 3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32 -> 16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16 -> 8
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # Named so that CAM explainers can find it robustly.
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)

    @property
    def target_layer(self) -> nn.Module:
        """The last convolutional layer, used as the default CAM target."""
        convs = [m for m in self.features if isinstance(m, nn.Conv2d)]
        return convs[-1]


# --------------------------------------------------------------------------- #
# Pretrained backbones
# --------------------------------------------------------------------------- #
_PRETRAINED = {
    "resnet18": ("resnet18", "ResNet18_Weights"),
    "resnet50": ("resnet50", "ResNet50_Weights"),
    "vgg16": ("vgg16", "VGG16_Weights"),
    "densenet121": ("densenet121", "DenseNet121_Weights"),
    "mobilenet_v3_large": ("mobilenet_v3_large", "MobileNet_V3_Large_Weights"),
}


def build_pretrained(name: str = "resnet50", weights: str = "DEFAULT") -> nn.Module:
    """Instantiate a torchvision classifier with pretrained weights.

    Parameters
    ----------
    name:
        One of ``_PRETRAINED``.
    weights:
        ``"DEFAULT"`` for the best available ImageNet weights, or ``None`` for
        random initialisation.
    """
    import torchvision.models as tvm  # local import keeps torchvision optional

    if name not in _PRETRAINED:
        raise ValueError(f"Unknown model '{name}'. Choose from {sorted(_PRETRAINED)}.")
    ctor_name, weights_enum_name = _PRETRAINED[name]
    ctor = getattr(tvm, ctor_name)
    if weights is None:
        return ctor(weights=None)
    weights_enum = getattr(tvm, weights_enum_name)
    chosen = weights_enum.DEFAULT if weights == "DEFAULT" else getattr(weights_enum, weights)
    return ctor(weights=chosen)


def build_model(
    name: str = "small_cnn",
    num_classes: int = 10,
    in_channels: int = 3,
    pretrained: bool = False,
) -> nn.Module:
    """Factory dispatching to :class:`SmallCNN` or a pretrained backbone."""
    if name == "small_cnn":
        return SmallCNN(num_classes=num_classes, in_channels=in_channels)
    return build_pretrained(name, weights="DEFAULT" if pretrained else None)


# --------------------------------------------------------------------------- #
# Target-layer resolution for CAM methods
# --------------------------------------------------------------------------- #
def resolve_target_layer(model: nn.Module, override: Optional[str] = None) -> nn.Module:
    """Return the final convolutional layer of ``model``.

    Grad-CAM and its relatives attribute at the last convolutional feature map,
    where spatial structure is retained but semantics are high-level. This
    helper walks the module tree and returns the last ``nn.Conv2d`` it finds,
    unless ``override`` names a specific sub-module (dotted path).
    """
    if override is not None:
        module = model
        for part in override.split("."):
            module = getattr(module, part)
        return module

    if isinstance(model, SmallCNN):
        return model.target_layer

    last_conv: Optional[nn.Module] = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    if last_conv is None:
        raise ValueError("No Conv2d layer found; specify `override` explicitly.")
    return last_conv
