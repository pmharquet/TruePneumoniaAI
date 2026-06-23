"""Lazy-loaded inference for the dashboard "Test des modèles" page.

Loads a trained Lightning checkpoint, runs single-image predictions and
full-split evaluations, and caches loaded models so repeated calls are cheap.
Torch and the model stack are imported lazily to keep dashboard startup fast.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

# checkpoint path (str) -> (model, device)
_CACHE: dict[str, tuple[Any, str]] = {}
_LOCK = threading.Lock()


def _load_model(ckpt_path: Path):
    import torch

    from src.models.classifier import PneumoniaClassifier

    key = str(ckpt_path)
    with _LOCK:
        cached = _CACHE.get(key)
        if cached is not None:
            return cached
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # pretrained=False skips the (unused) ImageNet download — weights come
        # from the checkpoint. pos_weight recreates the criterion buffer that
        # BCEWithLogitsLoss saved, so the state_dict loads strictly.
        model = PneumoniaClassifier.load_from_checkpoint(
            ckpt_path,
            map_location=device,
            pretrained=False,
            pos_weight=torch.tensor([1.0]),
        )
        model.eval().to(device)
        _CACHE[key] = (model, device)
        return model, device


def _clahe_enabled() -> bool:
    """Match the preprocessing used at training time (read from default config)."""
    import yaml

    cfg_path = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"
    try:
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        return bool(cfg.get("data", {}).get("clahe", False))
    except Exception:
        return False


def _val_transform(image_size: int = 224):
    # Letterbox + normalize works for any input size (no-op letterbox on images
    # already at model size), so it covers both dataset and uploaded images.
    from src.data.transforms import get_val_transforms_albumentations

    return get_val_transforms_albumentations(image_size, clahe=_clahe_enabled())


def _to_tensor(image: Image.Image, image_size: int):
    transform = _val_transform(image_size)
    arr = np.array(image.convert("RGB"))
    return transform(image=arr)["image"].float()


def predict_image(
    ckpt_path: Path,
    image: Image.Image,
    threshold: float | None = None,
    image_size: int = 224,
    tta: bool = True,
) -> dict[str, Any]:
    import torch

    model, device = _load_model(ckpt_path)
    tensor = _to_tensor(image, image_size).unsqueeze(0).to(device)
    with torch.no_grad():
        prob = torch.sigmoid(model(tensor))
        if tta:  # average over the horizontal flip
            prob = (prob + torch.sigmoid(model(torch.flip(tensor, dims=[3])))) / 2
        prob = prob.item()

    thr = float(threshold) if threshold is not None else float(getattr(model, "threshold", 0.5))
    return {
        "probability": prob,
        "threshold": thr,
        "prediction": "PNEUMONIA" if prob >= thr else "NORMAL",
    }


def evaluate_split(
    ckpt_path: Path,
    data_dir: str,
    split: str = "test",
    threshold: float | None = None,
    image_size: int = 224,
    batch_size: int = 64,
    tta: bool = True,
) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader
    from torchmetrics.functional.classification import binary_auroc

    from src.data.dataset import ChestXrayDataset

    model, device = _load_model(ckpt_path)
    dataset = ChestXrayDataset(data_dir, split, transform=_val_transform(image_size))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    probs_all, labels_all = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            p = torch.sigmoid(model(imgs))
            if tta:  # average over the horizontal flip
                p = (p + torch.sigmoid(model(torch.flip(imgs, dims=[3])))) / 2
            probs_all.append(p.cpu())
            labels_all.append(labels)
    probs = torch.cat(probs_all)
    labels = torch.cat(labels_all).int()

    thr = float(threshold) if threshold is not None else float(getattr(model, "threshold", 0.5))
    preds = (probs >= thr).int()

    pos, neg = labels == 1, labels == 0
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    tp = int((preds[pos] == 1).sum())
    tn = int((preds[neg] == 0).sum())
    fp = n_neg - tn
    fn = n_pos - tp

    sensitivity = tp / n_pos if n_pos else 0.0
    specificity = tn / n_neg if n_neg else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) else 0.0
    accuracy = (tp + tn) / len(labels) if len(labels) else 0.0
    auroc = float(binary_auroc(probs, labels))

    return {
        "split": split,
        "threshold": thr,
        "count": int(len(labels)),
        "n_normal": n_neg,
        "n_pneumonia": n_pos,
        "metrics": {
            "auroc": auroc,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "f1": f1,
            "accuracy": accuracy,
        },
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }
