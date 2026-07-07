"""Training utilities for the from-scratch CNN.

A minimal but complete training loop with cosine LR schedule, checkpointing and
top-1 evaluation. Pretrained backbones do not need training; this exists so the
explainers can be demonstrated on a model whose weights the user controls.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@dataclass
class TrainConfig:
    epochs: int = 20
    lr: float = 0.1
    weight_decay: float = 5e-4
    momentum: float = 0.9
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path: str = "checkpoints/small_cnn.pt"
    log_every: int = 100


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    correct = total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / max(total, 1)


def train(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader,
          cfg: Optional[TrainConfig] = None) -> nn.Module:
    cfg = cfg or TrainConfig()
    device = cfg.device
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.SGD(model.parameters(), lr=cfg.lr,
                                momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=cfg.epochs)

    best_acc = 0.0
    os.makedirs(os.path.dirname(cfg.ckpt_path) or ".", exist_ok=True)

    for epoch in range(cfg.epochs):
        model.train()
        running = 0.0
        for step, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimiser.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimiser.step()
            running += loss.item()
            if step % cfg.log_every == 0:
                print(f"epoch {epoch:02d} step {step:04d} loss {loss.item():.4f}")
        scheduler.step()

        acc = evaluate(model, test_loader, device)
        print(f"epoch {epoch:02d} | mean loss {running / len(train_loader):.4f} | test acc {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            torch.save({"model_state": model.state_dict(), "acc": acc}, cfg.ckpt_path)
            print(f"  -> saved checkpoint (acc {acc:.4f}) to {cfg.ckpt_path}")

    print(f"Training complete. Best test accuracy: {best_acc:.4f}")
    return model


def load_checkpoint(model: nn.Module, path: str, device: str = "cpu") -> nn.Module:
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model_state"] if "model_state" in state else state)
    model.to(device).eval()
    return model
