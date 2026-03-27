from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Sequence

from PIL import Image
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms


TINY_IMAGENET_NUM_CLASSES = 200


class TinyImageNetValDataset(Dataset):
    def __init__(self, root: str, transform: Optional[Callable] = None):
        self.root = Path(root)
        self.transform = transform
        self.images_dir = self.root / "images"
        annotations_path = self.root / "val_annotations.txt"
        if not self.images_dir.exists() or not annotations_path.exists():
            raise FileNotFoundError(
                "No se encontro la estructura oficial de validacion de Tiny ImageNet en val/."
            )

        self.samples = []
        with open(annotations_path, "r", encoding="utf-8") as f:
            rows = [line.strip().split("\t") for line in f if line.strip()]

        classes = sorted({row[1] for row in rows})
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

        for row in rows:
            image_name, class_name = row[0], row[1]
            image_path = self.images_dir / image_name
            self.samples.append((image_path, self.class_to_idx[class_name]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        image_path, target = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, target


class UnlabeledImageFolder(Dataset):
    def __init__(self, root: str, transform: Optional[Callable] = None):
        self.root = Path(root)
        self.transform = transform
        if (self.root / "images").exists():
            self.image_dir = self.root / "images"
        else:
            self.image_dir = self.root

        exts = {".jpg", ".jpeg", ".png"}
        self.files = sorted([p for p in self.image_dir.rglob("*") if p.suffix.lower() in exts])
        if not self.files:
            raise FileNotFoundError(f"No se encontraron imagenes en {self.image_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        image_path = self.files[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, image_path.name


def build_transforms(image_size: int = 64, augmentation: bool = True):
    train_tfms = [transforms.Resize((image_size, image_size))]
    if augmentation:
        train_tfms.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(image_size, padding=4),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            ]
        )
    train_tfms.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4802, 0.4481, 0.3975], std=[0.2770, 0.2691, 0.2821]),
        ]
    )

    eval_tfms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4802, 0.4481, 0.3975], std=[0.2770, 0.2691, 0.2821]),
        ]
    )

    return transforms.Compose(train_tfms), eval_tfms


def load_train_dataset(train_dir: str, image_size: int = 64, augmentation: bool = True):
    train_transform, _ = build_transforms(image_size=image_size, augmentation=augmentation)
    dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    return dataset



def load_val_dataset(val_dir: str, image_size: int = 64):
    _, eval_transform = build_transforms(image_size=image_size, augmentation=False)
    val_path = Path(val_dir)

    if (val_path / "images").exists() and (val_path / "val_annotations.txt").exists():
        return TinyImageNetValDataset(val_dir, transform=eval_transform)

    return datasets.ImageFolder(val_dir, transform=eval_transform)



def load_test_dataset(test_dir: str, image_size: int = 64):
    _, eval_transform = build_transforms(image_size=image_size, augmentation=False)
    test_path = Path(test_dir)
    try:
        return datasets.ImageFolder(test_dir, transform=eval_transform)
    except Exception:
        if test_path.exists():
            return UnlabeledImageFolder(test_dir, transform=eval_transform)
        raise



def shard_dataset(dataset: Sequence, worker_id: int, num_workers: int):
    if worker_id < 0 or worker_id >= num_workers:
        raise ValueError("worker_id debe estar entre 0 y num_workers - 1")
    indices = list(range(worker_id, len(dataset), num_workers))
    return Subset(dataset, indices)
