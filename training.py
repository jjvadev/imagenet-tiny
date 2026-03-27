from __future__ import annotations

import copy
import random
import time
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
from torch import nn
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass
class EpochMetrics:
    loss: float
    top1: float
    top5: float
    samples: int
    duration: float



def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def get_device(preferred: str = "auto") -> torch.device:
    preferred = preferred.lower()
    if preferred == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(preferred)



def build_optimizer(model: nn.Module, optimizer_name: str, lr: float, weight_decay: float, momentum: float):
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer_name = optimizer_name.lower()
    if optimizer_name == "sgd":
        return SGD(trainable_params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    if optimizer_name == "adam":
        return Adam(trainable_params, lr=lr, weight_decay=weight_decay)
    raise ValueError("optimizer debe ser 'sgd' o 'adam'")



def accuracy_topk(logits: torch.Tensor, target: torch.Tensor, topk: Tuple[int, ...] = (1, 5)):
    with torch.no_grad():
        maxk = min(max(topk), logits.size(1))
        _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        result = []
        for k in topk:
            k = min(k, logits.size(1))
            correct_k = correct[:k].reshape(-1).float().sum(0)
            result.append(correct_k.item())
        return result



def train_one_round(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer_name: str,
    lr: float,
    weight_decay: float,
    momentum: float,
    local_epochs: int,
    use_amp: bool = False,
    verbose: bool = True,
) -> EpochMetrics:
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, optimizer_name, lr, weight_decay, momentum)
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))
    model.to(device)
    model.train()

    start = time.perf_counter()
    total_loss = 0.0
    total_top1 = 0.0
    total_top5 = 0.0
    total_samples = 0

    for _ in range(local_epochs):
        iterator: Iterable = loader
        if verbose:
            iterator = tqdm(loader, desc="Entrenando", leave=False)
        for images, target in iterator:
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(use_amp and device.type == "cuda")):
                logits = model(images)
                loss = criterion(logits, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_size = target.size(0)
            top1, top5 = accuracy_topk(logits, target, topk=(1, 5))
            total_loss += loss.item() * batch_size
            total_top1 += top1
            total_top5 += top5
            total_samples += batch_size

    duration = time.perf_counter() - start
    return EpochMetrics(
        loss=total_loss / max(total_samples, 1),
        top1=total_top1 / max(total_samples, 1),
        top5=total_top5 / max(total_samples, 1),
        samples=total_samples,
        duration=duration,
    )


@torch.inference_mode()
def evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> EpochMetrics:
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    model.eval()

    start = time.perf_counter()
    total_loss = 0.0
    total_top1 = 0.0
    total_top5 = 0.0
    total_samples = 0

    for images, target in loader:
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, target)
        batch_size = target.size(0)
        top1, top5 = accuracy_topk(logits, target, topk=(1, 5))
        total_loss += loss.item() * batch_size
        total_top1 += top1
        total_top5 += top5
        total_samples += batch_size

    duration = time.perf_counter() - start
    return EpochMetrics(
        loss=total_loss / max(total_samples, 1),
        top1=total_top1 / max(total_samples, 1),
        top5=total_top5 / max(total_samples, 1),
        samples=total_samples,
        duration=duration,
    )



def clone_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}



def load_state_dict(model: nn.Module, state_dict: Dict[str, torch.Tensor]) -> None:
    copied = {k: v.clone() for k, v in state_dict.items()}
    model.load_state_dict(copied, strict=True)



def aggregate_state_dicts(weighted_states: list[tuple[Dict[str, torch.Tensor], int]]):
    if not weighted_states:
        raise ValueError("No hay estados para agregar")

    total_samples = sum(samples for _, samples in weighted_states)
    if total_samples <= 0:
        raise ValueError("El total de muestras debe ser mayor que cero")

    base = copy.deepcopy(weighted_states[0][0])
    for key in base.keys():
        base[key] = base[key].float() * 0.0
        for state_dict, samples in weighted_states:
            base[key] += state_dict[key].float() * (samples / total_samples)
    return base
