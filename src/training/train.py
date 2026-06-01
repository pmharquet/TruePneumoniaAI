"""
Entry point for training.

Usage:
    python -m src.training.train --config configs/default.yaml
"""

import argparse
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


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def train(cfg: dict):
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

        trainer = pl.Trainer(
            max_epochs=cfg["training"]["max_epochs"],
            precision=cfg["training"]["precision"],
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
