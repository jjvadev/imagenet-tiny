from __future__ import annotations

import torch.nn as nn
from torchvision import models


SUPPORTED_ARCHS = ["small_cnn", "resnet18", "mobilenet_v3_small"]


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int, dropout: float = 0.2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
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


def _replace_classifier_resnet(model: nn.Module, num_classes: int, dropout: float = 0.0):
    in_features = model.fc.in_features
    if dropout > 0:
        model.fc = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_features, num_classes))
    else:
        model.fc = nn.Linear(in_features, num_classes)
    return model


def _replace_classifier_mobilenet(model: nn.Module, num_classes: int, dropout: float = 0.0):
    if not hasattr(model, "classifier"):
        raise ValueError("El modelo MobileNet no tiene atributo classifier")

    last_linear_idx = None
    for i in range(len(model.classifier) - 1, -1, -1):
        if isinstance(model.classifier[i], nn.Linear):
            last_linear_idx = i
            break

    if last_linear_idx is None:
        raise ValueError("No se encontro la capa lineal final en MobileNet")

    in_features = model.classifier[last_linear_idx].in_features
    if dropout > 0:
        model.classifier[last_linear_idx] = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )
    else:
        model.classifier[last_linear_idx] = nn.Linear(in_features, num_classes)
    return model


def _freeze_backbone_resnet(model: nn.Module):
    for name, param in model.named_parameters():
        if not name.startswith("fc."):
            param.requires_grad = False
    return model


def _freeze_backbone_mobilenet(model: nn.Module):
    for name, param in model.named_parameters():
        if not name.startswith("classifier."):
            param.requires_grad = False
    return model


def build_imagenet_model(
    arch: str,
    num_classes: int,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    dropout: float = 0.0,
):
    arch = arch.lower().strip()

    if arch == "small_cnn":
        return SmallCNN(num_classes=num_classes, dropout=max(dropout, 0.1))

    if arch == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model = _replace_classifier_resnet(model, num_classes=num_classes, dropout=dropout)
        if freeze_backbone:
            model = _freeze_backbone_resnet(model)
        return model

    if arch == "mobilenet_v3_small":
        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        model = _replace_classifier_mobilenet(model, num_classes=num_classes, dropout=dropout)
        if freeze_backbone:
            model = _freeze_backbone_mobilenet(model)
        return model

    raise ValueError(
        f"Arquitectura no soportada: {arch}. Usa una de: {SUPPORTED_ARCHS}"
    )


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
