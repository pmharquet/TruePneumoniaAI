"""
PyTorch Lightning DataModule for chest X-ray classification.

Computes pos_weight from training set counts to handle PNEUMONIA/NORMAL imbalance.
"""

import os
from pathlib import Path

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .dataset import ChestXrayDataset
from .transforms import (
    get_preprocessed_transforms_albumentations,
    get_train_transforms_albumentations,
    get_val_transforms_albumentations,
)


class ChestXrayDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "chest_Xray",
        image_size: int = 224,
        batch_size: int = 32,
        num_workers: int = 4,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pos_weight: torch.Tensor | None = None

    def _loader_kwargs(self, shuffle: bool) -> dict:
        use_cuda = torch.cuda.is_available()
        use_workers = use_cuda and os.name != "nt"
        effective_workers = self.num_workers if use_workers else 0
        return {
            "batch_size": self.batch_size,
            "shuffle": shuffle,
            "num_workers": effective_workers,
            "pin_memory": use_cuda,
            "persistent_workers": effective_workers > 0,
        }

    def setup(self, stage: str | None = None):
        is_preprocessed = (Path(self.data_dir) / "augmentation_summary.json").exists()
        train_tf = (
            get_preprocessed_transforms_albumentations()
            if is_preprocessed
            else get_train_transforms_albumentations(self.image_size)
        )
        val_tf = get_val_transforms_albumentations(self.image_size)

        self.train_dataset = ChestXrayDataset(self.data_dir, "train", transform=train_tf)
        self.val_dataset = ChestXrayDataset(self.data_dir, "val", transform=val_tf)
        self.test_dataset = ChestXrayDataset(self.data_dir, "test", transform=val_tf)

        counts = self.train_dataset.class_counts()
        n_normal = counts["NORMAL"]
        n_pneumonia = counts["PNEUMONIA"]
        # pos_weight = n_negative / n_positive — upweights PNEUMONIA in BCEWithLogitsLoss
        self.pos_weight = torch.tensor([n_normal / n_pneumonia], dtype=torch.float32)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, **self._loader_kwargs(shuffle=True))

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, **self._loader_kwargs(shuffle=False))

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, **self._loader_kwargs(shuffle=False))
