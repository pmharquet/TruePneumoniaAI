"""
Entry point for training.

Usage:
    python -m src.training.train --config configs/default.yaml
"""

import argparse
import os
from pathlib import Path

import mlflow
import torch
import yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import MLFlowLogger

from src.data.datamodule import ChestXrayDataModule
from src.models.classifier import PneumoniaClassifier
from src.training.dashboard_callback import DashboardEventLogger


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def configure_runtime_cache() -> None:
    cache_root = Path(os.environ.get("TPAI_CACHE_DIR", "outputs/cache")).resolve()
    torch_home = Path(os.environ.get("TORCH_HOME", cache_root / "torch")).resolve()
    xdg_cache = Path(os.environ.get("XDG_CACHE_HOME", cache_root / "xdg")).resolve()

    os.environ.setdefault("TPAI_CACHE_DIR", str(cache_root))
    os.environ.setdefault("TORCH_HOME", str(torch_home))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache))
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

    torch_home.mkdir(parents=True, exist_ok=True)
    xdg_cache.mkdir(parents=True, exist_ok=True)


@torch.no_grad()
def calibrate_threshold(model, dataloader, target_sensitivity: float) -> float:
    """Pick a decision threshold on the validation set.

    Returns the highest threshold whose sensitivity still meets the target
    (which maximizes specificity at that constraint). If no threshold reaches
    the target, falls back to the one with the highest sensitivity.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval().to(device)

    probs_all, labels_all = [], []
    for imgs, labels in dataloader:
        imgs = imgs.to(device).contiguous(memory_format=torch.channels_last)
        probs_all.append(torch.sigmoid(model(imgs)).cpu())
        labels_all.append(labels.cpu())
    probs = torch.cat(probs_all)
    labels = torch.cat(labels_all).int()

    pos, neg = labels == 1, labels == 0
    n_pos, n_neg = int(pos.sum()), int(neg.sum())

    best_t, best_spec, fallback_t, fallback_sens = 0.5, -1.0, 0.5, -1.0
    for t in torch.linspace(0.01, 0.99, 99).tolist():
        preds = probs >= t
        sens = int((preds & pos).sum()) / n_pos if n_pos else 0.0
        spec = int((~preds & neg).sum()) / n_neg if n_neg else 0.0
        if sens > fallback_sens:
            fallback_sens, fallback_t = sens, t
        if sens >= target_sensitivity and spec >= best_spec:
            best_spec, best_t = spec, t

    return best_t if best_spec >= 0 else fallback_t


def train(cfg: dict):
    configure_runtime_cache()
    # cudnn.benchmark autotunes conv kernels for fixed input sizes — only safe
    # (and only worth it) when we are NOT requesting deterministic training.
    deterministic = bool(cfg["training"].get("deterministic", False))
    pl.seed_everything(42, workers=deterministic)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = not deterministic

    dm = ChestXrayDataModule(
        data_dir=cfg["data"]["data_dir"],
        image_size=cfg["data"]["image_size"],
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        prefetch_factor=cfg["data"].get("prefetch_factor", 4),
        val_split=cfg["data"].get("val_split", 0.15),
    )
    dm.setup()

    model = PneumoniaClassifier(
        backbone=cfg["model"]["backbone"],
        pretrained=cfg["model"]["pretrained"],
        dropout=cfg["model"]["dropout"],
        learning_rate=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
        pos_weight=dm.pos_weight,
        threshold=cfg["threshold"]["default"],
    )
    # channels_last memory format speeds up convnets on tensor cores.
    if torch.cuda.is_available():
        model = model.to(memory_format=torch.channels_last)
    if cfg["training"].get("compile", False):
        model = torch.compile(model)

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    with mlflow.start_run():
        mlflow.log_params({
            "backbone": cfg["model"]["backbone"],
            "pretrained": cfg["model"]["pretrained"],
            "dropout": cfg["model"]["dropout"],
            "lr": cfg["training"]["learning_rate"],
            "batch_size": cfg["data"]["batch_size"],
            "image_size": cfg["data"]["image_size"],
            "pos_weight": dm.pos_weight.item(),
        })

        logger = MLFlowLogger(
            experiment_name=cfg["mlflow"]["experiment_name"],
            tracking_uri=cfg["mlflow"]["tracking_uri"],
        )

        ckpt_dir = cfg["paths"]["checkpoints"]
        callbacks = [
            DashboardEventLogger(
                log_every_n_steps=cfg.get("dashboard", {}).get("log_every_n_steps", 1),
            ),
            # Select/stop on val/loss, not val/auroc: AUROC saturates near 1.0
            # within the first epoch on this dataset, so it can't distinguish a
            # well-calibrated model from an overfit one. val/loss penalizes the
            # overconfident wrong predictions that wreck specificity.
            # auto_insert_metric_name=False so the "/" in "val/loss" is not baked
            # into the filename (which would nest the .ckpt in a subdirectory).
            ModelCheckpoint(
                dirpath=ckpt_dir,
                filename="best-loss-epoch{epoch:02d}",
                monitor="val/loss",
                mode="min",
                save_top_k=1,
                auto_insert_metric_name=False,
            ),
            ModelCheckpoint(
                dirpath=ckpt_dir,
                filename="best-auroc-epoch{epoch:02d}",
                monitor="val/auroc",
                mode="max",
                save_top_k=1,
                auto_insert_metric_name=False,
            ),
            EarlyStopping(monitor="val/loss", patience=5, mode="min"),
            LearningRateMonitor(logging_interval="epoch"),
        ]

        precision = cfg["training"]["precision"]
        if precision == "16-mixed" and not torch.cuda.is_available():
            precision = "32-true"

        trainer = pl.Trainer(
            max_epochs=cfg["training"]["max_epochs"],
            precision=precision,
            callbacks=callbacks,
            logger=logger,
            deterministic=deterministic,
            log_every_n_steps=10,
        )

        trainer.fit(model, datamodule=dm)

        # Calibrate the decision threshold on validation to hit the target
        # sensitivity, then evaluate test at that threshold. AUROC measures
        # ranking; a screening tool still needs a sensible operating point.
        best_ckpt = trainer.checkpoint_callbacks[0].best_model_path
        if best_ckpt:
            model = PneumoniaClassifier.load_from_checkpoint(
                best_ckpt, pos_weight=dm.pos_weight
            )
        target_sens = cfg["threshold"].get("target_sensitivity", 0.95)
        threshold = calibrate_threshold(model, dm.val_dataloader(), target_sens)
        model.threshold = threshold
        print(
            f"Calibrated decision threshold = {threshold:.3f} "
            f"(target sensitivity = {target_sens})"
        )
        mlflow.log_metric("calibrated_threshold", threshold)

        trainer.test(model, datamodule=dm)

        mlflow.pytorch.log_model(model, "model")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
