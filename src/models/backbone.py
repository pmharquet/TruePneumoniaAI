"""
Pretrained backbone factory — DenseNet121 (CheXNet reference) or ResNet50.

The classifier head is replaced with a single logit for binary BCE loss.
"""

import torch
import torch.nn as nn
from torchvision import models


def build_backbone(name: str, pretrained: bool = True, dropout: float = 0.5) -> nn.Module:
    if name == "densenet121":
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.densenet121(weights=weights)
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 1),
        )
    elif name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = models.resnet50(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 1),
        )
    else:
        raise ValueError(f"Unknown backbone: {name}. Choose 'densenet121' or 'resnet50'.")

    return model


def freeze_backbone(model: nn.Module, name: str, train_last_block: bool = True) -> None:
    """Freeze the pretrained feature extractor in place, keeping the classifier
    head (and optionally the last block) trainable.

    Fine-tuning all 7M params on this small, augmentation-duplicated dataset
    memorizes the train sources and fails to generalize to the shifted test
    distribution. Keeping the early ImageNet features fixed reduces overfitting.
    """
    if name == "densenet121":
        for param in model.features.parameters():
            param.requires_grad = False
        if train_last_block:
            for block in ("denseblock4", "norm5"):
                for param in getattr(model.features, block).parameters():
                    param.requires_grad = True
    elif name == "resnet50":
        for param_name, param in model.named_parameters():
            if not param_name.startswith("fc."):
                param.requires_grad = False
        if train_last_block:
            for param in model.layer4.parameters():
                param.requires_grad = True
    else:
        raise ValueError(f"Unknown backbone: {name}.")
    # The classifier head is a separate module and stays trainable.
