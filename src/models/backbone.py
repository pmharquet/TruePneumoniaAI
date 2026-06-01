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
