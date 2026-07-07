"""xai_cnn: a research-grade toolkit for explaining CNN image classifiers.

Public API
----------
>>> from xai_cnn import build_model, load_dataset, build_explainer
>>> from xai_cnn.models.cnn import resolve_target_layer
"""

from .data.datasets import DatasetBundle, load_dataset
from .explainers import EXPLAINER_REGISTRY, build_explainer
from .models.cnn import build_model, resolve_target_layer

__version__ = "0.1.0"
__all__ = [
    "load_dataset", "DatasetBundle",
    "build_model", "resolve_target_layer",
    "build_explainer", "EXPLAINER_REGISTRY",
]
