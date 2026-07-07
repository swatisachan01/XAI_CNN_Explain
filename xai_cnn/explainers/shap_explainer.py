"""SHAP attribution for CNNs.

Wraps the ``shap`` library's gradient-based explainers, which approximate
Shapley values by integrating gradients against a background distribution.
``GradientExplainer`` is used because it is robust across arbitrary torch
models (unlike ``DeepExplainer``, which needs supported layer types).

The background set should be a small, representative batch of inputs
(e.g. 20-50 training images). If ``shap`` is not installed, a clear error is
raised pointing the user to ``pip install shap``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from .base import Explainer, normalise_map


class ShapGradient(Explainer):
    name = "SHAP (Gradient)"
    requires_grad = True

    def __init__(self, model: nn.Module, background: torch.Tensor,
                 device: str = "cpu", n_samples: int = 50) -> None:
        super().__init__(model, device)
        self.background = background.to(device)
        self.n_samples = n_samples
        try:
            import shap  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise ImportError("ShapGradient requires the `shap` package (pip install shap).") from exc

    def attribute(self, x: torch.Tensor, target: Optional[int] = None) -> torch.Tensor:
        import shap

        x = x.to(self.device)
        if target is None:
            with torch.no_grad():
                target = int(self.model(x).argmax(dim=1)[0].item())

        explainer = shap.GradientExplainer(self.model, self.background)
        # ranked_outputs=None -> explain all classes; we pick `target` after.
        shap_values = explainer.shap_values(x, nsamples=self.n_samples)

        # shap_values: list over classes, each (B,C,H,W); or ndarray with class axis.
        if isinstance(shap_values, list):
            attr = shap_values[target][0]              # (C,H,W)
        else:
            attr = shap_values[0, ..., target]         # (C,H,W)
        attr = np.abs(attr).max(axis=0)                # collapse channels -> (H,W)
        return normalise_map(torch.from_numpy(attr).float().to(self.device))
