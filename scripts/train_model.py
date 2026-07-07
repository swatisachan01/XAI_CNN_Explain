#!/usr/bin/env python
"""Train the from-scratch CNN on a public dataset.

Examples
--------
    python scripts/train_model.py --dataset cifar10 --epochs 30
    python scripts/train_model.py --dataset fashion_mnist --epochs 15 --lr 0.05
"""

from __future__ import annotations

import argparse

from xai_cnn.data.datasets import load_dataset
from xai_cnn.models.cnn import build_model
from xai_cnn.train import TrainConfig, train


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="cifar10", choices=["cifar10", "fashion_mnist"])
    p.add_argument("--data-root", default="./data")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--ckpt", default="checkpoints/small_cnn.pt")
    args = p.parse_args()

    bundle = load_dataset(args.dataset, root=args.data_root, batch_size=args.batch_size)
    model = build_model("small_cnn", num_classes=bundle.num_classes,
                        in_channels=bundle.in_channels)
    cfg = TrainConfig(epochs=args.epochs, lr=args.lr, ckpt_path=args.ckpt)
    train(model, bundle.train_loader, bundle.test_loader, cfg)


if __name__ == "__main__":
    main()
