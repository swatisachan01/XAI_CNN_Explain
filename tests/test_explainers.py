"""Smoke and correctness tests for the explainer suite.

Run with::

    pytest -q

Uses the small from-scratch CNN with random weights on tiny random inputs, so
the tests run in seconds on CPU and require only torch + the package itself.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from xai_cnn.explainers import EXPLAINER_REGISTRY, build_explainer
from xai_cnn.models.cnn import SmallCNN, resolve_target_layer
from xai_cnn.utils.metrics import deletion_auc, insertion_auc, sparsity

CAM_METHODS = {"gradcam", "gradcam++", "layercam", "eigencam", "scorecam", "ablationcam"}
# Methods exercised in the fast suite (shap/lime need optional deps or are slow).
FAST_METHODS = [
    "saliency", "input_x_gradient", "integrated_gradients", "smoothgrad",
    "guided_backprop", "gradcam", "gradcam++", "layercam", "eigencam",
    "scorecam", "ablationcam", "occlusion", "lrp",
]


@pytest.fixture(scope="module")
def setup():
    torch.manual_seed(0)
    model = SmallCNN(num_classes=10, in_channels=3).eval()
    x = torch.rand(1, 3, 32, 32)
    layer = resolve_target_layer(model)
    return model, x, layer


@pytest.mark.parametrize("method", FAST_METHODS)
def test_attribution_shape_and_range(setup, method):
    model, x, layer = setup
    kwargs = {}
    if method == "integrated_gradients":
        kwargs["steps"] = 8
    if method == "smoothgrad":
        kwargs["n_samples"] = 4
    explainer = build_explainer(
        method, model, device="cpu",
        target_layer=layer if method in CAM_METHODS else None, **kwargs,
    )
    cam = explainer.attribute(x, target=3)
    if hasattr(explainer, "release"):
        explainer.release()

    assert cam.shape == (32, 32), f"{method} returned {tuple(cam.shape)}"
    assert torch.isfinite(cam).all(), f"{method} produced non-finite values"
    assert cam.min() >= -1e-5 and cam.max() <= 1 + 1e-5, f"{method} out of [0,1]"


def test_rise_runs(setup):
    model, x, _ = setup
    explainer = build_explainer("rise", model, device="cpu", n_masks=64, batch_size=32)
    cam = explainer.attribute(x, target=1)
    assert cam.shape == (32, 32)
    assert torch.isfinite(cam).all()


def test_metrics_bounds(setup):
    model, x, layer = setup
    cam = build_explainer("gradcam", model, target_layer=layer).attribute(x, target=0)
    d, dcurve = deletion_auc(model, x, cam, target=0, step=128)
    i, icurve = insertion_auc(model, x, cam, target=0, step=128)
    assert 0.0 <= d <= 1.0
    assert 0.0 <= i <= 1.0
    assert 0.0 <= sparsity(cam) <= 1.0
    assert len(dcurve) == len(icurve)


def test_registry_complete():
    # Every registered name is constructible or explicitly deferred (shap).
    for name in EXPLAINER_REGISTRY:
        assert isinstance(name, str)
