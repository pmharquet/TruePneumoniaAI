"""
Post-training threshold tuning.

Finds the decision threshold that meets target sensitivity on the validation set,
then reports the corresponding specificity and AUC-ROC.

Clinical rationale: missing a pneumonia (false negative) is more dangerous than
a false alarm, so we fix sensitivity >= target and maximize specificity.

Usage:
    python -m src.training.threshold_tuning --ckpt checkpoints/normal-pneumonia/<timestamp>/best-loss-epochXX.ckpt --config configs/default.yaml
"""

import argparse

import numpy as np
import torch
import yaml
from sklearn.metrics import roc_auc_score, roc_curve

from src.data.datamodule import ChestXrayDataModule
from src.models.classifier import PneumoniaClassifier


def tune_threshold(
    probs: np.ndarray,
    labels: np.ndarray,
    target_sensitivity: float = 0.95,
) -> dict:
    fpr, tpr, thresholds = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)

    # Among thresholds reaching target sensitivity, pick the one with best specificity
    viable = [(t, 1 - f) for t, f, s in zip(thresholds, fpr, tpr) if s >= target_sensitivity]
    if not viable:
        raise ValueError(f"No threshold achieves sensitivity >= {target_sensitivity}")

    best_threshold, best_specificity = max(viable, key=lambda x: x[1])

    preds = (probs >= best_threshold).astype(int)
    achieved_sensitivity = (preds[labels == 1] == 1).mean()

    return {
        "threshold": float(best_threshold),
        "sensitivity": float(achieved_sensitivity),
        "specificity": float(best_specificity),
        "auc_roc": float(auc),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    dm = ChestXrayDataModule(
        data_dir=cfg["data"]["data_dir"],
        image_size=cfg["data"]["image_size"],
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
    )
    dm.setup()

    model = PneumoniaClassifier.load_from_checkpoint(args.ckpt)
    model.eval()

    all_probs, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in dm.val_dataloader():
            logits = model(imgs)
            all_probs.extend(torch.sigmoid(logits).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    results = tune_threshold(
        np.array(all_probs),
        np.array(all_labels),
        target_sensitivity=cfg["threshold"]["target_sensitivity"],
    )

    print("\n--- Threshold Tuning Results ---")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")

    cfg["threshold"]["default"] = results["threshold"]
    with open(args.config, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    print(f"\nUpdated threshold in {args.config}")


if __name__ == "__main__":
    main()
