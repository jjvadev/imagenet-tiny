from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMG_EXTENSIONS


def build_transforms(image_size: int = 64, use_augmentation: bool = False):
    if use_augmentation:
        train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
    else:
        train_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    eval_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    return train_transform, eval_transform


class ImageListDataset(Dataset):
    def __init__(
        self,
        root_dir: str | Path,
        samples: Sequence[Tuple[str, int]],
        transform=None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        rel_path, label = self.samples[index]
        image_path = self.root_dir / rel_path

        with Image.open(image_path) as img:
            image = img.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def _read_wnids(root_path: Path) -> List[str]:
    wnids_file = root_path / "wnids.txt"
    if not wnids_file.exists():
        raise FileNotFoundError(f"No existe wnids.txt en: {wnids_file}")

    wnids: List[str] = []
    with open(wnids_file, "r", encoding="utf-8") as f:
        for line in f:
            wnid = line.strip()
            if wnid:
                wnids.append(wnid)

    if not wnids:
        raise ValueError("wnids.txt está vacío o no contiene clases válidas.")

    return wnids


def _scan_train_split(root_path: Path, class_to_idx: dict[str, int]) -> List[Tuple[str, int]]:
    train_dir = root_path / "train"
    if not train_dir.exists():
        raise FileNotFoundError(f"No existe el split de entrenamiento: {train_dir}")

    samples: List[Tuple[str, int]] = []

    for wnid in sorted(class_to_idx.keys()):
        images_dir = train_dir / wnid / "images"
        if not images_dir.exists():
            continue

        label = class_to_idx[wnid]

        for img_path in sorted(images_dir.iterdir()):
            if _is_image_file(img_path):
                rel_path = img_path.relative_to(root_path)
                samples.append((rel_path.as_posix(), label))

    if not samples:
        raise ValueError(f"No se encontraron imágenes de entrenamiento en: {train_dir}")

    return samples


def _scan_val_split(root_path: Path, class_to_idx: dict[str, int]) -> List[Tuple[str, int]]:
    val_dir = root_path / "val"
    images_dir = val_dir / "images"
    annotations_file = val_dir / "val_annotations.txt"

    if not val_dir.exists():
        raise FileNotFoundError(f"No existe el split de validación: {val_dir}")
    if not images_dir.exists():
        raise FileNotFoundError(f"No existe la carpeta de imágenes de validación: {images_dir}")
    if not annotations_file.exists():
        raise FileNotFoundError(f"No existe el archivo de anotaciones: {annotations_file}")

    samples: List[Tuple[str, int]] = []

    with open(annotations_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue

            image_name, wnid = parts[0], parts[1]

            if wnid not in class_to_idx:
                continue

            img_path = images_dir / image_name
            if not _is_image_file(img_path):
                continue

            label = class_to_idx[wnid]
            rel_path = img_path.relative_to(root_path)
            samples.append((rel_path.as_posix(), label))

    if not samples:
        raise ValueError(f"No se encontraron imágenes de validación en: {val_dir}")

    return samples


def load_tiny_imagenet_manifests(
    data_dir: str | Path,
    max_train_samples: int | None = None,
    max_val_samples: int | None = None,
    seed: int = 42,
):
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"No existe el directorio del dataset: {root}")

    class_names = _read_wnids(root)
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    train_samples = _scan_train_split(root, class_to_idx)
    val_samples = _scan_val_split(root, class_to_idx)

    if max_train_samples is not None and max_train_samples > 0:
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(train_samples), generator=g).tolist()
        train_samples = [train_samples[i] for i in perm[:max_train_samples]]

    if max_val_samples is not None and max_val_samples > 0:
        g = torch.Generator().manual_seed(seed + 1)
        perm = torch.randperm(len(val_samples), generator=g).tolist()
        val_samples = [val_samples[i] for i in perm[:max_val_samples]]

    return train_samples, val_samples, class_names