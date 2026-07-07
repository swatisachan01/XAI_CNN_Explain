# XAI_CNN_Explain


A research-grade, PyTorch toolkit for **explaining convolutional neural network (CNN) image classifiers**. It trains (or loads) a CNN on public datasets and attributes its predictions back to the input using a broad set of gradient-based, CAM-based, perturbation-based and reference-based methods — then **evaluates those explanations quantitatively** with faithfulness metrics, so methods can be compared on evidence rather than by eye.

The repository is designed to be a clean starting point for interpretability experiments: every method implements the same narrow interface, so adding a new attribution method or a new evaluation metric is a small, local change.

---

## Table of contents

1. [Why this exists](#why-this-exists)
2. [What's included](#whats-included)
3. [Installation](#installation)
4. [Quick start](#quick-start)
5. [The explanation methods](#the-explanation-methods)
6. [Evaluating explanations](#evaluating-explanations)
7. [Repository layout](#repository-layout)


---

## Why this exists

Saliency maps are easy to produce and easy to over-trust. Two maps can look equally plausible while one is faithful to the model and the other is essentially edge detection. This toolkit therefore pairs **thirteen attribution methods** with **causal faithfulness metrics** (deletion / insertion) and a localisation metric (pointing game), so that "which explanation should I believe?" becomes a measurable question. It targets both a small from-scratch CNN (whose weights you control) and standard ImageNet-pretrained backbones (for real-world images).

## What's included

- **Models** — a compact from-scratch VGG-style CNN plus wrappers around torchvision ResNet / VGG / DenseNet / MobileNet with ImageNet weights.
- **Datasets** — CIFAR-10 and Fashion-MNIST (auto-download), and an `ImageFolder` loader for your own images with pretrained models. Normalisation statistics travel with the data so perturbation methods stay correct.
- **Explainers** — gradient, CAM, perturbation and reference families (full list below), all behind one `build_explainer(...)` factory.
- **Metrics** — deletion / insertion AUC, pointing game, and a Gini sparsity score.
- **Visualisation** — single overlays and multi-method comparison grids (using explicit `subplots_adjust` spacing to avoid panel overlap).
- **CLI scripts** — train a model, explain one image, or compare many methods with metrics in a single command.
- **Tests** — a fast pytest suite that runs every gradient/CAM method on a random-weight model in seconds.

## Installation

```bash
git clone <your-fork-url> xai-cnn-explainability
cd xai-cnn-explainability
python -m venv .venv && source .venv/bin/activate     # optional
pip install -e .            # installs the package + core dependencies
pip install -e ".[shap,dev]"  # optional: SHAP explainer + pytest
```

Requires Python ≥ 3.9. A CUDA GPU is optional; everything runs on CPU (perturbation methods and Score-CAM are simply slower).

## Quick start

**1 — Explain a single image with a pretrained ResNet-50:**

```bash
python scripts/explain_image.py \
    --image path/to/cat.jpg --model resnet50 --pretrained \
    --method gradcam --out assets/cat_gradcam.png
```

**2 — Compare many methods on one image, with faithfulness metrics:**

```bash
python scripts/compare_methods.py \
    --model resnet50 --pretrained \
    --dataset imagenet_sample --data-root ./my_images \
    --methods gradcam gradcam++ scorecam integrated_gradients smoothgrad occlusion rise lrp \
    --index 0 --metrics --out assets/comparison.png
```

**3 — Train the from-scratch CNN on CIFAR-10, then explain it:**

```bash
python scripts/train_model.py --dataset cifar10 --epochs 30
python scripts/compare_methods.py \
    --model small_cnn --ckpt checkpoints/small_cnn.pt \
    --dataset cifar10 --methods gradcam saliency smoothgrad lrp occlusion \
    --index 7 --metrics --out assets/cifar_comparison.png
```

**4 — Use it as a library:**

```python
import torch
from xai_cnn import build_model, load_dataset, build_explainer
from xai_cnn.models.cnn import resolve_target_layer
from xai_cnn.utils.visualization import comparison_grid, tensor_to_image

bundle = load_dataset("cifar10")
model = build_model("small_cnn", num_classes=10).eval()
image, _ = bundle.test_loader.dataset[0]
x = image.unsqueeze(0)

layer = resolve_target_layer(model)
cam = build_explainer("gradcam", model, target_layer=layer).attribute(x)

img = tensor_to_image(x, bundle.inverse_transform)
comparison_grid(img, {"Grad-CAM": cam}, save_path="out.png")
```

## The explanation methods

Every explainer returns a single-channel map, normalised to `[0, 1]`, at input resolution. Call any of them by name through `build_explainer(name, model, ...)`.

### Gradient-based (`xai_cnn/explainers/gradient.py`)

| Name | Key idea |
|---|---|
| `saliency` | Magnitude of the input gradient of the target logit (Simonyan et al., 2013). |
| `input_x_gradient` | Element-wise input × gradient. |
| `integrated_gradients` | Path integral of gradients from a baseline (Sundararajan et al., 2017); satisfies completeness. |
| `smoothgrad` | Averages saliency over Gaussian-noised copies to reduce visual noise (Smilkov et al., 2017). |
| `guided_backprop` | Backprop that only passes positive gradients through ReLUs (Springenberg et al., 2014). |

### CAM family (`xai_cnn/explainers/cam.py`)

| Name | Weighting scheme |
|---|---|
| `gradcam` | Global-average-pooled gradients of the target conv layer (Selvaraju et al., 2017). |
| `gradcam++` | Higher-order gradient weighting; better for multiple instances of a class. |
| `layercam` | Element-wise positive gradients — finer spatial detail, works at any layer. |
| `scorecam` | Gradient-free; each channel weighted by the confidence gain from its masked forward pass (Wang et al., 2020). |
| `ablationcam` | Gradient-free; weight = confidence drop when the channel is ablated. |
| `eigencam` | First principal component of the activation maps — class-agnostic. |

CAM methods need a target convolutional layer. `resolve_target_layer(model)` picks the last `Conv2d` automatically, or pass `--target-layer` (a dotted module path) to choose your own.

### Perturbation-based (`xai_cnn/explainers/perturbation.py`)

| Name | Key idea |
|---|---|
| `occlusion` | Slide an occluding patch and record the drop in target confidence (Zeiler & Fergus, 2014). |
| `rise` | Thousands of random masks, each weighted by the resulting score (Petsiuk et al., 2018). |
| `lime` | Superpixel on/off perturbations fit a local linear surrogate (Ribeiro et al., 2016); uses `lime` if installed, otherwise a built-in SLIC + ridge fallback. |

### Reference-based

| Name | File | Key idea |
|---|---|---|
| `shap` | `shap_explainer.py` | Shapley-value attributions via `shap.GradientExplainer` against a background batch. Requires `pip install shap`. |
| `lrp` | `lrp.py` | Layer-wise Relevance Propagation (Bach et al., 2015), epsilon + z⁺ rules. Supports sequential nets (SmallCNN, VGG); **not** residual/dense backbones. |

## Evaluating explanations

Faithful attributions should actually control the model's output. The metrics live in `xai_cnn/utils/metrics.py`:

- **Deletion AUC** — remove the most salient pixels first; the target probability should fall fast, so **lower is better**.
- **Insertion AUC** — add the most salient pixels onto a blurred baseline; the probability should rise fast, so **higher is better**.
- **Pointing game** — is the map's peak inside a ground-truth bounding box? (localisation).
- **Sparsity (Gini)** — how concentrated is the map; useful as a tie-breaker.

Add `--metrics` to `compare_methods.py` for a scored table across every method you ran.

## Repository layout

```
xai-cnn-explainability/
├── xai_cnn/
│   ├── data/datasets.py          # CIFAR-10, Fashion-MNIST, ImageFolder + norm stats
│   ├── models/cnn.py             # SmallCNN, pretrained wrappers, target-layer resolver
│   ├── explainers/
│   │   ├── base.py               # Explainer ABC + hook manager + map normalisation
│   │   ├── gradient.py           # saliency, IG, smoothgrad, guided backprop, input×grad
│   │   ├── cam.py                # grad-cam, grad-cam++, layer/eigen/score/ablation-cam
│   │   ├── perturbation.py       # occlusion, RISE, LIME
│   │   ├── shap_explainer.py     # SHAP (optional dep)
│   │   ├── lrp.py                # Layer-wise Relevance Propagation
│   │   └── __init__.py           # registry + build_explainer factory
│   ├── utils/
│   │   ├── visualization.py      # overlays and comparison grids
│   │   └── metrics.py            # deletion/insertion/pointing-game/sparsity
│   └── train.py                  # training loop, checkpointing, evaluation
├── scripts/
│   ├── train_model.py            # train SmallCNN on a public dataset
│   ├── explain_image.py          # single method on a single image file
│   └── compare_methods.py        # many methods + metrics on one image
├── tests/test_explainers.py      # fast pytest suite
├── configs/default.yaml          # documented default hyper-parameters
├── requirements.txt · pyproject.toml · LICENSE · .gitignore
```
