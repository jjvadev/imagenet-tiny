from __future__ import annotations

import pickle
import random
import socket
import struct
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import models, transforms

HEADER_SIZE = 8
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".JPEG"}


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(prefer_mps: bool = True) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if prefer_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def send_msg(sock: socket.socket, obj) -> None:
    payload = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    header = struct.pack("!Q", len(payload))
    sock.sendall(header + payload)


def recv_exact(sock: socket.socket, n: int) -> bytes:
    data = bytearray()
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            raise ConnectionError("La conexion fue cerrada")
        data.extend(chunk)
    return bytes(data)


def recv_msg(sock: socket.socket):
    header = recv_exact(sock, HEADER_SIZE)
    msg_len = struct.unpack("!Q", header)[0]
    payload = recv_exact(sock, msg_len)
    return pickle.loads(payload)


def _is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in {e.lower() for e in IMG_EXTENSIONS}


def _load_class_names(root: Path) -> list[str]:
    wnids_file = root / "wnids.txt"
    if wnids_file.exists():
        with open(wnids_file, "r", encoding="utf-8") as f:
            class_names = [line.strip() for line in f if line.strip()]
        if class_names:
            return class_names

    train_dir = root / "train"
    if train_dir.exists():
        class_names = sorted([p.name for p in train_dir.iterdir() if p.is_dir()])
        if class_names:
            return class_names

    val_dir = root / "val"
    ann_file = val_dir / "val_annotations.txt"
    if ann_file.exists():
        classes = set()
        with open(ann_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    classes.add(parts[1])
        class_names = sorted(classes)
        if class_names:
            return class_names

    raise FileNotFoundError(
        f"No se pudieron obtener las clases desde {root}. "
        f"Falta wnids.txt y no fue posible inferir clases desde train/ o val/."
    )


class TinyImageNetDataset(Dataset):
    def __init__(self, root: str | Path, split: str = "train", transform=None):
        self.root = Path(root)
        self.split = split
        self.transform = transform

        self.class_names = _load_class_names(self.root)
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.samples: list[tuple[str, int]] = []

        if split == "train":
            train_dir = self.root / "train"
            if not train_dir.exists():
                raise FileNotFoundError(f"No existe: {train_dir}")

            for wnid in self.class_names:
                img_dir = train_dir / wnid / "images"
                if not img_dir.exists():
                    continue
                for img_path in sorted(img_dir.iterdir()):
                    if _is_image_file(img_path):
                        self.samples.append((str(img_path), self.class_to_idx[wnid]))

        elif split == "val":
            val_dir = self.root / "val"
            ann_file = val_dir / "val_annotations.txt"
            img_dir = val_dir / "images"

            if not ann_file.exists():
                raise FileNotFoundError(f"No existe: {ann_file}")
            if not img_dir.exists():
                raise FileNotFoundError(f"No existe: {img_dir}")

            with open(ann_file, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) < 2:
                        continue
                    img_name, wnid = parts[0], parts[1]
                    if wnid not in self.class_to_idx:
                        continue
                    img_path = img_dir / img_name
                    if img_path.exists():
                        self.samples.append((str(img_path), self.class_to_idx[wnid]))
        else:
            raise ValueError("split debe ser 'train' o 'val'")

        if not self.samples:
            raise RuntimeError(f"No se encontraron muestras en {self.root} [{split}]")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, target = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, target


def build_transforms(
    image_size: int = 64,
    train: bool = False,
    arch: str = "small_cnn",
    pretrained: bool = False,
):
    arch = arch.lower().strip()

    if arch == "resnet18" and pretrained:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        if train:
            return transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(
                        brightness=0.2,
                        contrast=0.2,
                        saturation=0.2,
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std),
                ]
            )

        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    mean = [0.4802, 0.4481, 0.3975]
    std = [0.2770, 0.2691, 0.2821]

    if train:
        return transforms.Compose(
            [
                transforms.RandomCrop(image_size, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def make_partitioned_train_loader(
    data_dir: str,
    worker_id: int,
    partition_count: int,
    batch_size: int,
    loader_workers: int,
    image_size: int = 64,
    seed: int = 42,
    arch: str = "small_cnn",
    pretrained: bool = False,
):
    dataset = TinyImageNetDataset(
        root=data_dir,
        split="train",
        transform=build_transforms(
            image_size=image_size,
            train=True,
            arch=arch,
            pretrained=pretrained,
        ),
    )

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(dataset), generator=g).tolist()
    indices = perm[worker_id::partition_count]

    subset = Subset(dataset, indices)

    pin_memory = torch.cuda.is_available()
    persistent_workers = loader_workers > 0

    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=loader_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    return loader, len(subset)


def make_val_loader(
    data_dir: str,
    batch_size: int,
    loader_workers: int,
    image_size: int = 64,
    max_samples: int = 0,
    seed: int = 42,
    arch: str = "small_cnn",
    pretrained: bool = False,
):
    dataset = TinyImageNetDataset(
        root=data_dir,
        split="val",
        transform=build_transforms(
            image_size=image_size,
            train=False,
            arch=arch,
            pretrained=pretrained,
        ),
    )

    if max_samples and max_samples > 0 and max_samples < len(dataset):
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(dataset), generator=g).tolist()
        dataset = Subset(dataset, perm[:max_samples])

    pin_memory = torch.cuda.is_available()
    persistent_workers = loader_workers > 0

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=loader_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    return loader


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int = 200, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def freeze_backbone_resnet18(model: nn.Module):
    for name, param in model.named_parameters():
        if not name.startswith("fc."):
            param.requires_grad = False
    return model


def build_model(
    arch: str = "small_cnn",
    num_classes: int = 200,
    pretrained: bool = False,
    freeze_backbone: bool = False,
):
    arch = arch.lower().strip()

    if arch == "small_cnn":
        return SmallCNN(num_classes=num_classes)

    if arch == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        if freeze_backbone:
            model = freeze_backbone_resnet18(model)
        return model

    raise ValueError("arch debe ser 'small_cnn' o 'resnet18'")


def state_dict_to_cpu(state_dict):
    out = {}
    for k, v in state_dict.items():
        out[k] = v.detach().cpu() if torch.is_tensor(v) else v
    return out


def weighted_average_state_dict(worker_payloads):
    total_samples = sum(p["num_samples"] for p in worker_payloads)
    if total_samples <= 0:
        raise ValueError("No hay muestras validas para agregar")

    ref_state = worker_payloads[0]["state_dict"]
    agg = {}

    for key in ref_state.keys():
        ref_tensor = ref_state[key]
        if torch.is_tensor(ref_tensor) and torch.is_floating_point(ref_tensor):
            acc = torch.zeros_like(ref_tensor, dtype=torch.float32)
            for payload in worker_payloads:
                weight = payload["num_samples"] / total_samples
                acc += payload["state_dict"][key].float() * weight
            agg[key] = acc.to(ref_tensor.dtype)
        else:
            agg[key] = ref_tensor.clone() if torch.is_tensor(ref_tensor) else ref_tensor

    return agg


def accuracy_top1(logits: torch.Tensor, targets: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    correct = pred.eq(targets).sum().item()
    return float(correct)


def build_optimizer(
    model: nn.Module,
    optimizer_name: str,
    lr: float,
    weight_decay: float,
    momentum: float,
):
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer_name = optimizer_name.lower()

    if optimizer_name == "sgd":
        return torch.optim.SGD(
            trainable_params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )

    if optimizer_name == "adam":
        return torch.optim.Adam(
            trainable_params,
            lr=lr,
            weight_decay=weight_decay,
        )

    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            trainable_params,
            lr=lr,
            weight_decay=weight_decay,
        )

    raise ValueError("optimizer debe ser 'sgd', 'adam' o 'adamw'")


def train_one_round(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer_name: str,
    lr: float,
    weight_decay: float,
    momentum: float,
    local_epochs: int,
    use_amp: bool = True,
):
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, optimizer_name, lr, weight_decay, momentum)

    amp_enabled = use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    model.to(device)
    model.train()

    total_loss = 0.0
    total_correct = 0.0
    total_samples = 0
    start = time.perf_counter()

    for _ in range(local_epochs):
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=amp_enabled):
                logits = model(images)
                loss = criterion(logits, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            bsz = targets.size(0)
            total_loss += loss.item() * bsz
            total_correct += accuracy_top1(logits, targets)
            total_samples += bsz

    elapsed = time.perf_counter() - start

    return {
        "train_loss": total_loss / max(total_samples, 1),
        "train_acc": total_correct / max(total_samples, 1),
        "num_samples": total_samples,
        "train_time": elapsed,
    }


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool = True,
):
    criterion = nn.CrossEntropyLoss()
    amp_enabled = use_amp and device.type == "cuda"

    model.to(device)
    model.eval()

    total_loss = 0.0
    total_correct = 0.0
    total_samples = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, targets)

        bsz = targets.size(0)
        total_loss += loss.item() * bsz
        total_correct += accuracy_top1(logits, targets)
        total_samples += bsz

    return {
        "loss": total_loss / max(total_samples, 1),
        "acc": total_correct / max(total_samples, 1),
        "samples": total_samples,
    }