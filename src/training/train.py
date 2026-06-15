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


def train(cfg: dict):
    configure_runtime_cache()
    pl.seed_everything(42, workers=True)

    dm = ChestXrayDataModule(
        data_dir=cfg["data"]["data_dir"],
        image_size=cfg["data"]["image_size"],
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
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
            ModelCheckpoint(
                dirpath=ckpt_dir,
                filename="best-{epoch:02d}-{val/auroc:.4f}",
                monitor="val/auroc",
                mode="max",
                save_top_k=1,
            ),
            ModelCheckpoint(
                dirpath=ckpt_dir,
                filename="best-sensitivity-{epoch:02d}-{val/sensitivity:.4f}",
                monitor="val/sensitivity",
                mode="max",
                save_top_k=1,
            ),
            EarlyStopping(monitor="val/auroc", patience=7, mode="max"),
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
            deterministic=True,
            log_every_n_steps=10,
        )

        trainer.fit(model, datamodule=dm)
        trainer.test(model, datamodule=dm, ckpt_path="best")

        mlflow.pytorch.log_model(model, "model")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    train(cfg)


if __name__ == "__main__":
    main()
