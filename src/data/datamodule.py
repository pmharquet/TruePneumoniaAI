"""
PyTorch Lightning DataModule for chest X-ray classification.

Computes pos_weight from training set counts to handle PNEUMONIA/NORMAL imbalance.
"""

import csv
import random
from collections import defaultdict
from pathlib import Path

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .dataset import CLASS_TO_IDX, ChestXrayDataset
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
        prefetch_factor: int = 4,
        val_split: float = 0.15,
        split_seed: int = 42,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.val_split = val_split
        self.split_seed = split_seed
        self.pos_weight: torch.Tensor | None = None

    def _loader_kwargs(self, shuffle: bool) -> dict:
        use_cuda = torch.cuda.is_available()
        workers = max(0, self.num_workers)
        kwargs = {
            "batch_size": self.batch_size,
            "shuffle": shuffle,
            "num_workers": workers,
            "pin_memory": use_cuda,
            "persistent_workers": workers > 0,
        }
        # prefetch_factor is only valid when worker processes are used.
        if workers > 0:
            kwargs["prefetch_factor"] = self.prefetch_factor
        return kwargs

    def _carve_validation_split(self):
        """Carve a leak-free validation set out of the train split.

        Groups every training image (originals + their augmented variants) by
        its source image, so no augmented copy of a validation image can leak
        into train. Validation uses one clean original per held-out source.
        Falls back to a per-file stratified split when no manifest is present.
        """
        root = Path(self.data_dir)
        manifest = root / "augmentation_manifest.csv"

        groups: dict[str, list[tuple[Path, str]]] = defaultdict(list)
        group_label: dict[str, int] = {}

        if manifest.exists():
            with open(manifest, newline="") as f:
                for row in csv.DictReader(f):
                    if row["split"] != "train":
                        continue
                    src = row["source_path"]
                    groups[src].append((root / row["output_path"], row["kind"]))
                    group_label[src] = CLASS_TO_IDX[row["class"]]
        else:
            # No augmentation manifest: each file is its own group.
            base = ChestXrayDataset(self.data_dir, "train")
            for path, label in base.samples:
                groups[str(path)].append((path, "original"))
                group_label[str(path)] = label

        by_class: dict[int, list[str]] = defaultdict(list)
        for src, label in group_label.items():
            by_class[label].append(src)

        rng = random.Random(self.split_seed)
        train_samples: list[tuple[Path, int]] = []
        val_samples: list[tuple[Path, int]] = []
        for label, srcs in by_class.items():
            srcs = sorted(srcs)
            rng.shuffle(srcs)
            n_val = int(round(len(srcs) * self.val_split))
            val_srcs = set(srcs[:n_val])
            for src in srcs:
                if src in val_srcs:
                    originals = [p for p, kind in groups[src] if kind == "original"]
                    chosen = originals[0] if originals else groups[src][0][0]
                    val_samples.append((chosen, label))
                else:
                    train_samples.extend((p, label) for p, _ in groups[src])

        return train_samples, val_samples

    def setup(self, stage: str | None = None):
        is_preprocessed = (Path(self.data_dir) / "augmentation_summary.json").exists()
        train_tf = (
            get_preprocessed_transforms_albumentations()
            if is_preprocessed
            else get_train_transforms_albumentations(self.image_size)
        )
        # Preprocessed images are already letterboxed to model size, so the
        # eval transform only needs normalization; raw images need letterboxing.
        val_tf = (
            get_preprocessed_transforms_albumentations()
            if is_preprocessed
            else get_val_transforms_albumentations(self.image_size)
        )

        if self.val_split and self.val_split > 0:
            train_samples, val_samples = self._carve_validation_split()
            self.train_dataset = ChestXrayDataset(
                self.data_dir, "train", transform=train_tf, samples=train_samples
            )
            self.val_dataset = ChestXrayDataset(
                self.data_dir, "train", transform=val_tf, samples=val_samples
            )
        else:
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
