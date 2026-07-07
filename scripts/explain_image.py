#!/usr/bin/env python
"""Explain a single image file with one method (quick, single-panel output).

Example
-------
    python scripts/explain_image.py --image cat.jpg --model resnet50 \
        --pretrained --method gradcam --out assets/cat_gradcam.png
"""

from __future__ import annotations

import argparse

import torch
from PIL import Image
from torchvision import transforms

from xai_cnn.data.datasets import IMAGENET_MEAN, IMAGENET_STD, _make_inverse
from xai_cnn.explainers import build_explainer
from xai_cnn.models.cnn import build_model, resolve_target_layer
from xai_cnn.utils.visualization import save_single, tensor_to_image

CAM_METHODS = {"gradcam", "gradcam++", "layercam", "eigencam", "scorecam", "ablationcam"}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image", required=True)
    p.add_argument("--model", default="resnet50")
    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--method", default="gradcam")
    p.add_argument("--target", type=int, default=None)
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--out", default="assets/explanation.png")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    tf = transforms.Compose([
        transforms.Resize(int(args.size * 1.14)),
        transforms.CenterCrop(args.size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    img = Image.open(args.image).convert("RGB")
    x = tf(img).unsqueeze(0).to(args.device)

    model = build_model(args.model, pretrained=args.pretrained).to(args.device).eval()
    target_layer = resolve_target_layer(model)

    explainer = build_explainer(
        args.method, model, device=args.device,
        target_layer=target_layer if args.method in CAM_METHODS else None,
    )
    cam = explainer.attribute(x, target=args.target)
    if hasattr(explainer, "release"):
        explainer.release()

    inverse = _make_inverse(IMAGENET_MEAN, IMAGENET_STD)
    image_np = tensor_to_image(x, inverse)
    save_single(image_np, cam.cpu(), args.out, title=explainer.name)
    print(f"Saved {explainer.name} explanation to {args.out}")


if __name__ == "__main__":
    main()
