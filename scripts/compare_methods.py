#!/usr/bin/env python
"""Run several explainers on one image and produce a comparison figure + metrics.

This is the main demonstration script. It loads a model (from-scratch checkpoint
or an ImageNet-pretrained backbone), picks a test image, runs the requested
explainers, renders a side-by-side overlay grid, and scores each method with the
deletion / insertion faithfulness metrics.

Examples
--------
Pretrained ResNet-50 on a user image folder::

    python scripts/compare_methods.py \
        --model resnet50 --pretrained \
        --dataset imagenet_sample --data-root ./my_images \
        --methods gradcam gradcam++ scorecam integrated_gradients rise occlusion \
        --index 0 --out assets/comparison.png

From-scratch CNN on CIFAR-10::

    python scripts/compare_methods.py \
        --model small_cnn --ckpt checkpoints/small_cnn.pt \
        --dataset cifar10 --methods gradcam saliency smoothgrad lrp occlusion \
        --index 7 --out assets/cifar_comparison.png
"""

from __future__ import annotations

import argparse
import time
from typing import Dict

import torch

from xai_cnn.data.datasets import load_dataset
from xai_cnn.explainers import build_explainer
from xai_cnn.models.cnn import build_model, resolve_target_layer
from xai_cnn.train import load_checkpoint
from xai_cnn.utils.metrics import deletion_auc, insertion_auc, sparsity
from xai_cnn.utils.visualization import comparison_grid, tensor_to_image

CAM_METHODS = {"gradcam", "gradcam++", "layercam", "eigencam", "scorecam", "ablationcam"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="resnet50")
    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--ckpt", default=None, help="checkpoint for small_cnn")
    p.add_argument("--dataset", default="imagenet_sample")
    p.add_argument("--data-root", default="./data")
    p.add_argument("--methods", nargs="+",
                   default=["gradcam", "gradcam++", "scorecam",
                            "integrated_gradients", "smoothgrad", "occlusion"])
    p.add_argument("--index", type=int, default=0, help="test-set image index")
    p.add_argument("--target", type=int, default=None, help="class to explain")
    p.add_argument("--out", default="assets/comparison.png")
    p.add_argument("--metrics", action="store_true", help="compute deletion/insertion")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--target-layer", default=None,
                   help="dotted path to CAM target layer (auto if omitted)")
    return p.parse_args()


def build_the_model(args, bundle) -> torch.nn.Module:
    if args.model == "small_cnn":
        model = build_model("small_cnn", num_classes=bundle.num_classes,
                            in_channels=bundle.in_channels)
        if args.ckpt:
            model = load_checkpoint(model, args.ckpt, args.device)
    else:
        model = build_model(args.model, pretrained=args.pretrained)
    return model.to(args.device).eval()


def main() -> None:
    args = parse_args()

    root_kw = {"root": args.data_root} if args.dataset != "imagenet_sample" else {"root": args.data_root}
    bundle = load_dataset(args.dataset, **root_kw)
    model = build_the_model(args, bundle)

    # Grab a single image.
    dataset = bundle.test_loader.dataset
    image, label = dataset[args.index]
    x = image.unsqueeze(0).to(args.device)

    with torch.no_grad():
        pred = int(model(x).argmax(dim=1)[0].item())
    target = args.target if args.target is not None else pred
    print(f"Image {args.index}: true={bundle.class_names[label] if label < len(bundle.class_names) else label}, "
          f"predicted={bundle.class_names[pred] if pred < len(bundle.class_names) else pred}")

    target_layer = resolve_target_layer(model, args.target_layer)

    attributions: Dict[str, torch.Tensor] = {}
    for name in args.methods:
        kwargs = {}
        if name == "lime":
            kwargs["inverse_transform"] = bundle.inverse_transform
        t0 = time.time()
        explainer = build_explainer(name, model, device=args.device,
                                    target_layer=target_layer if name in CAM_METHODS else None,
                                    **kwargs)
        cam = explainer.attribute(x, target=target)
        attributions[explainer.name] = cam.detach().cpu()
        if hasattr(explainer, "release"):
            explainer.release()
        print(f"  {explainer.name:22s} done in {time.time() - t0:.2f}s")

    image_np = tensor_to_image(x, bundle.inverse_transform)
    title = f"{args.model} | class: {bundle.class_names[target] if target < len(bundle.class_names) else target}"
    comparison_grid(image_np, attributions, title=title, save_path=args.out)
    print(f"Saved comparison figure to {args.out}")

    if args.metrics:
        step = max(1, (image.shape[-1] * image.shape[-2]) // 50)
        print("\nFaithfulness metrics (deletion lower=better, insertion higher=better):")
        print(f"{'method':22s} {'deletion':>10s} {'insertion':>10s} {'sparsity':>10s}")
        for name, cam in attributions.items():
            d, _ = deletion_auc(model, x, cam, target=target, step=step, device=args.device)
            i, _ = insertion_auc(model, x, cam, target=target, step=step, device=args.device)
            s = sparsity(cam)
            print(f"{name:22s} {d:10.4f} {i:10.4f} {s:10.4f}")


if __name__ == "__main__":
    main()
