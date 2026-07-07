"""Public-dataset loaders with normalisation metadata.

All loaders return a :class:`DatasetBundle` carrying the ``torch`` data loaders
plus the class names and the normalisation statistics. Explainers need the
normalisation constants to map perturbations and visualisations back into pixel
space, so we keep them attached to the data rather than hard-coded downstream.

Supported datasets
------------------
* ``cifar10``        -- 32x32 RGB, 10 classes (torchvision download).
* ``fashion_mnist``  -- 28x28 greyscale, 10 classes (torchvision download).
* ``imagenet_sample``-- a user-supplied folder of images for pretrained models,
  using the standard ImageNet normalisation. No download required.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader


# ImageNet statistics, reused by pretrained backbones.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class DatasetBundle:
    """Everything an experiment needs about a dataset."""

    train_loader: Optional[DataLoader]
    test_loader: DataLoader
    class_names: List[str]
    mean: Tuple[float, ...]
    std: Tuple[float, ...]
    in_channels: int
    image_size: int
    inverse_transform: Callable[[torch.Tensor], torch.Tensor] = field(repr=False)

    @property
    def num_classes(self) -> int:
        return len(self.class_names)


def _make_inverse(mean: Tuple[float, ...], std: Tuple[float, ...]) -> Callable:
    """Return a function undoing normalisation for visualisation."""
    mean_t = torch.tensor(mean).view(-1, 1, 1)
    std_t = torch.tensor(std).view(-1, 1, 1)

    def inverse(x: torch.Tensor) -> torch.Tensor:
        # Accept (C,H,W) or (B,C,H,W).
        if x.dim() == 4:
            return (x * std_t.to(x.device) + mean_t.to(x.device)).clamp(0, 1)
        return (x * std_t.to(x.device) + mean_t.to(x.device)).clamp(0, 1)

    return inverse


def load_cifar10(root: str = "./data", batch_size: int = 128, num_workers: int = 4,
                 download: bool = True) -> DatasetBundle:
    from torchvision import datasets, transforms

    mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train = datasets.CIFAR10(root, train=True, download=download, transform=train_tf)
    test = datasets.CIFAR10(root, train=False, download=download, transform=test_tf)
    classes = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]
    return DatasetBundle(
        train_loader=DataLoader(train, batch_size=batch_size, shuffle=True,
                                num_workers=num_workers, pin_memory=True),
        test_loader=DataLoader(test, batch_size=batch_size, shuffle=False,
                               num_workers=num_workers, pin_memory=True),
        class_names=classes, mean=mean, std=std, in_channels=3, image_size=32,
        inverse_transform=_make_inverse(mean, std),
    )


def load_fashion_mnist(root: str = "./data", batch_size: int = 128, num_workers: int = 4,
                       download: bool = True) -> DatasetBundle:
    from torchvision import datasets, transforms

    mean, std = (0.2860,), (0.3530,)
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    train = datasets.FashionMNIST(root, train=True, download=download, transform=tf)
    test = datasets.FashionMNIST(root, train=False, download=download, transform=tf)
    classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    return DatasetBundle(
        train_loader=DataLoader(train, batch_size=batch_size, shuffle=True,
                                num_workers=num_workers, pin_memory=True),
        test_loader=DataLoader(test, batch_size=batch_size, shuffle=False,
                               num_workers=num_workers, pin_memory=True),
        class_names=classes, mean=mean, std=std, in_channels=1, image_size=28,
        inverse_transform=_make_inverse(mean, std),
    )


def load_imagenet_folder(root: str, batch_size: int = 16, num_workers: int = 4,
                         image_size: int = 224) -> DatasetBundle:
    """Load a user folder of images (``ImageFolder`` layout) for pretrained nets.

    Expects ``root/<class_name>/*.jpg`` structure. ImageNet class names are not
    inferred; the folder names become the class list. Used only for the test
    loader (no training).
    """
    from torchvision import datasets, transforms

    tf = transforms.Compose([
        transforms.Resize(int(image_size * 1.14)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    ds = datasets.ImageFolder(root, transform=tf)
    return DatasetBundle(
        train_loader=None,
        test_loader=DataLoader(ds, batch_size=batch_size, shuffle=False,
                               num_workers=num_workers),
        class_names=ds.classes, mean=IMAGENET_MEAN, std=IMAGENET_STD,
        in_channels=3, image_size=image_size,
        inverse_transform=_make_inverse(IMAGENET_MEAN, IMAGENET_STD),
    )


DATASETS = {
    "cifar10": load_cifar10,
    "fashion_mnist": load_fashion_mnist,
    "imagenet_sample": load_imagenet_folder,
}


def load_dataset(name: str, **kwargs) -> DatasetBundle:
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Choose from {sorted(DATASETS)}.")
    return DATASETS[name](**kwargs)
