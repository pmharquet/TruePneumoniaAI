"""Train a small seed-ensemble and evaluate it on the test set.

Each member is the regularized DenseNet (frozen backbone + label smoothing),
trained with a different seed. We average per-image probabilities (each with
horizontal-flip TTA), calibrate the decision threshold on a held-out 40% of the
test set, and report the other 60% — disjoint, so the numbers are honest.

Usage:
    python -m scripts.ensemble --seeds 42 123 7
"""

from __future__ import annotations

import argparse
import copy
import glob
from pathlib import Path

import torch

from src.data.datamodule import ChestXrayDataModule
from src.training.train import (
    load_config,
    metrics_at,
    pick_threshold,
    stratified_split,
    train,
)


@torch.no_grad()
def tta_probs(model, loader, device):
    model.eval().to(device)
    probs, labels = [], []
    for imgs, lbl in loader:
        imgs = imgs.to(device).contiguous(memory_format=torch.channels_last)
        p = torch.sigmoid(model(imgs))
        p = (p + torch.sigmoid(model(torch.flip(imgs, dims=[3])))) / 2
        probs.append(p.cpu())
        labels.append(lbl.cpu())
    return torch.cat(probs), torch.cat(labels).int()


def best_ckpt(ckpt_dir: str) -> str:
    candidates = glob.glob(f"{ckpt_dir}/best-loss-epoch*.ckpt")
    if not candidates:
        raise FileNotFoundError(f"No best-loss checkpoint in {ckpt_dir}")
    # save_top_k=1 leaves one, but versions may exist — take the newest.
    return max(candidates, key=lambda p: Path(p).stat().st_mtime)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 7])
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_dirs = {}
    for seed in args.seeds:
        ckpt_dir = f"checkpoints/ens_seed{seed}"
        ckpt_dirs[seed] = ckpt_dir
        if args.skip_train:
            continue
        cfg = copy.deepcopy(base_cfg)
        cfg["training"]["seed"] = seed
        cfg["paths"]["checkpoints"] = ckpt_dir
        print(f"\n========== training seed {seed} -> {ckpt_dir} ==========")
        train(cfg)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Shared test set (same preprocessing as training).
    dm = ChestXrayDataModule(
        data_dir=base_cfg["data"]["data_dir"],
        image_size=base_cfg["data"]["image_size"],
        batch_size=base_cfg["data"]["batch_size"],
        num_workers=0,
        val_split=base_cfg["data"].get("val_split", 0.15),
        clahe=base_cfg["data"].get("clahe", False),
    )
    dm.setup()
    test_loader = dm.test_dataloader()

    from src.dashboard.inference import _load_model

    member_probs = []
    labels_ref = None
    print("\n========== per-member test AUROC (TTA) ==========")
    from torchmetrics.functional.classification import binary_auroc

    for seed, ckpt_dir in ckpt_dirs.items():
        ckpt = best_ckpt(ckpt_dir)
        model, _ = _load_model(Path(ckpt))
        probs, labels = tta_probs(model, test_loader, device)
        labels_ref = labels
        member_probs.append(probs)
        print(f"  seed {seed}: AUROC={float(binary_auroc(probs, labels)):.4f}  ({Path(ckpt).name})")

    ensemble = torch.stack(member_probs).mean(0)
    labels = labels_ref

    target_sens = base_cfg["threshold"].get("target_sensitivity", 0.95)
    calib_idx, eval_idx = stratified_split(labels, calib_frac=0.4, seed=42)
    thr = pick_threshold(ensemble[calib_idx], labels[calib_idx], target_sens)

    print("\n========== ENSEMBLE (test eval split, 60%) ==========")
    print(f"members={len(member_probs)}  threshold={thr:.3f}  (target sensitivity={target_sens})")
    em = metrics_at(ensemble[eval_idx], labels[eval_idx], thr)
    print(f"ensemble AUROC={float(binary_auroc(ensemble, labels)):.4f}  (full test)")
    print("@ thr : " + ", ".join(f"{k}={v:.3f}" for k, v in em.items()))
    print("@ 0.5 : " + ", ".join(f"{k}={v:.3f}" for k, v in metrics_at(ensemble[eval_idx], labels[eval_idx], 0.5).items()))


if __name__ == "__main__":
    main()
