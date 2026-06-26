"""
ChestXrayDataset — loads <root>/{train,val,test}/{<class>,...}

Default task is binary NORMAL vs PNEUMONIA (0/1). The class layout is
configurable via `class_to_idx` so the same dataset serves the pneumonia
subtype task too (BACTERIA vs VIRUS), which only differs in folder names.
"""

from pathlib import Path
from typing import Callable, Optional

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


# Default binary task. Order defines the label index: NORMAL=0, PNEUMONIA=1.
CLASS_TO_IDX = {"NORMAL": 0, "PNEUMONIA": 1}


class ChestXrayDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str,
        transform: Optional[Callable] = None,
        use_rgb: bool = True,
        samples: Optional[list[tuple[Path, int]]] = None,
        class_to_idx: Optional[dict[str, int]] = None,
    ):
        self.root = Path(root) / split
        self.transform = transform
        self.use_rgb = use_rgb
        self.class_to_idx = dict(class_to_idx) if class_to_idx else dict(CLASS_TO_IDX)
        self.samples: list[tuple[Path, int]] = []

        # Allow constructing from an explicit (path, label) list — used by the
        # DataModule to build a leak-free validation split that groups augmented
        # variants by their source image instead of globbing a directory.
        if samples is not None:
            self.samples = list(samples)
            if not self.samples:
                raise ValueError("Empty samples list passed to ChestXrayDataset")
            return

        for class_name, label in self.class_to_idx.items():
            class_dir = self.root / class_name
            if not class_dir.exists():
                raise FileNotFoundError(f"Expected directory: {class_dir}")
            for img_path in class_dir.glob("*.jpeg"):
                self.samples.append((img_path, label))
            for img_path in class_dir.glob("*.jpg"):
                self.samples.append((img_path, label))
            for img_path in class_dir.glob("*.png"):
                self.samples.append((img_path, label))

        if not self.samples:
            raise ValueError(f"No images found under {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path, label = self.samples[idx]
        img = Image.open(img_path)
        img = img.convert("RGB") if self.use_rgb else img.convert("L")
        img_np = np.array(img)

        if self.transform:
            augmented = self.transform(image=img_np)
            img_tensor = augmented["image"].float()
        else:
            img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float() / 255.0

        return img_tensor, torch.tensor(label, dtype=torch.float32)

    def class_counts(self) -> dict[str, int]:
        counts = {name: 0 for name in self.class_to_idx}
        idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        for _, label in self.samples:
            counts[idx_to_class[label]] += 1
        return counts
