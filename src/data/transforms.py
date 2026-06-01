"""
Image transforms pipeline.

MONAI handles medical-imaging-specific ops (ScaleIntensity, EnsureChannelFirst).
Albumentations handles classical augmentation (flips, rotations, CLAHE).
"""

import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from monai.transforms import (
    Compose,
    ScaleIntensity,
    EnsureChannelFirst,
    Resize,
    RandFlip,
    RandRotate,
    RandZoom,
    ToTensor,
)


def get_train_transforms_albumentations(image_size: int = 224) -> A.Compose:
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.CLAHE(clip_limit=2.0, p=0.3), # improves contrast in low-density lung regions
        A.GaussNoise(var_limit=(5.0, 30.0), p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_val_transforms_albumentations(image_size: int = 224) -> A.Compose:
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_monai_train_transforms(image_size: int = 224) -> Compose:
    """MONAI pipeline — used when loading DICOM or single-channel arrays."""
    return Compose([
        EnsureChannelFirst(),
        ScaleIntensity(),
        Resize(spatial_size=(image_size, image_size)),
        RandFlip(prob=0.5, spatial_axis=1),
        RandRotate(range_x=0.2, prob=0.5, keep_size=True),
        RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.3),
        ToTensor(),
    ])


def get_monai_val_transforms(image_size: int = 224) -> Compose:
    return Compose([
        EnsureChannelFirst(),
        ScaleIntensity(),
        Resize(spatial_size=(image_size, image_size)),
        ToTensor(),
    ])
