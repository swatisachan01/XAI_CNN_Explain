"""Explainer registry and factory.

Use :func:`build_explainer` to instantiate any method by name, or iterate
:data:`EXPLAINER_REGISTRY` to run a full comparison. Methods are grouped as:

* gradient      : saliency, input_x_gradient, integrated_gradients, smoothgrad, guided_backprop
* cam           : gradcam, gradcam++, layercam, eigencam, scorecam, ablationcam
* perturbation  : occlusion, rise, lime
* reference     : shap, lrp
"""

from __future__ import annotations

from typing import Callable, Dict

import torch.nn as nn

from .base import Explainer
from .cam import (AblationCAM, EigenCAM, GradCAM, GradCAMPlusPlus, LayerCAM,
                  ScoreCAM)
from .gradient import (GuidedBackprop, InputXGradient, IntegratedGradients,
                       Saliency, SmoothGrad)
from .lrp import LRP
from .perturbation import RISE, LIMEImage, Occlusion

#: Methods that need a target convolutional layer at construction time.
_CAM_METHODS = {
    "gradcam": GradCAM,
    "gradcam++": GradCAMPlusPlus,
    "layercam": LayerCAM,
    "eigencam": EigenCAM,
    "scorecam": ScoreCAM,
    "ablationcam": AblationCAM,
}

#: Methods constructed with just ``(model, device)``.
_SIMPLE_METHODS = {
    "saliency": Saliency,
    "input_x_gradient": InputXGradient,
    "integrated_gradients": IntegratedGradients,
    "smoothgrad": SmoothGrad,
    "guided_backprop": GuidedBackprop,
    "occlusion": Occlusion,
    "rise": RISE,
    "lrp": LRP,
}

EXPLAINER_REGISTRY = {**_SIMPLE_METHODS, **_CAM_METHODS, "lime": LIMEImage, "shap": None}


def build_explainer(name: str, model: nn.Module, device: str = "cpu",
                    target_layer: nn.Module = None, **kwargs) -> Explainer:
    """Instantiate an explainer by name.

    Extra keyword arguments are forwarded to the explainer constructor, so you
    can pass e.g. ``steps=100`` for Integrated Gradients or ``n_masks=4000``
    for RISE.
    """
    name = name.lower()
    if name in _CAM_METHODS:
        if target_layer is None:
            raise ValueError(f"'{name}' needs a `target_layer`.")
        return _CAM_METHODS[name](model, target_layer, device=device, **kwargs)
    if name in _SIMPLE_METHODS:
        return _SIMPLE_METHODS[name](model, device=device, **kwargs)
    if name == "lime":
        return LIMEImage(model, device=device, **kwargs)
    if name == "shap":
        from .shap_explainer import ShapGradient
        return ShapGradient(model, device=device, **kwargs)
    raise ValueError(f"Unknown explainer '{name}'. Options: {sorted(EXPLAINER_REGISTRY)}.")


__all__ = [
    "Explainer", "build_explainer", "EXPLAINER_REGISTRY",
    "Saliency", "InputXGradient", "IntegratedGradients", "SmoothGrad", "GuidedBackprop",
    "GradCAM", "GradCAMPlusPlus", "LayerCAM", "EigenCAM", "ScoreCAM", "AblationCAM",
    "Occlusion", "RISE", "LIMEImage", "LRP",
]
