"""
Entry point for training.

Usage:
    python -m src.training.train --config configs/default.yaml
"""

import argparse
import os
from datetime import datetime
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


def resolve_run_dir(cfg: dict) -> Path:
    """Per-run directory holding this run's checkpoints AND its dashboard
    event/state files, so everything about a run lives together under
    checkpoints/<task>/<timestamp>/.

    The dashboard supplies the exact dir via TPAI_RUN_DIR; a CLI run nests a
    fresh timestamp under the config's per-task checkpoints base. A config
    snapshot is saved alongside so a run is self-describing.
    """
    env_dir = os.environ.get("TPAI_RUN_DIR")
    if env_dir:
        run_dir = Path(env_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = Path(cfg["paths"]["checkpoints"]) / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    snapshot = run_dir / "config.yaml"
    if not snapshot.exists():
        with snapshot.open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
    return run_dir


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
def gather_probs(model, dataloader, tta: bool = True):
    """Run the model over a dataloader and return (probs, int labels) on CPU."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval().to(device)
    probs_all, labels_all = [], []
    for imgs, labels in dataloader:
        imgs = imgs.to(device).contiguous(memory_format=torch.channels_last)
        probs = torch.sigmoid(model(imgs))
        if tta:  # average over the horizontal flip
            probs = (probs + torch.sigmoid(model(torch.flip(imgs, dims=[3])))) / 2
        probs_all.append(probs.cpu())
        labels_all.append(labels.cpu())
    return torch.cat(probs_all), torch.cat(labels_all).int()


def pick_threshold(probs, labels, target_sensitivity: float) -> float:
    """Highest threshold whose sensitivity still meets the target (maximizing
    specificity). Falls back to the most-sensitive threshold if unreachable."""
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


def metrics_at(probs, labels, threshold: float) -> dict:
    from torchmetrics.functional.classification import binary_auroc

    pos, neg = labels == 1, labels == 0
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    preds = (probs >= threshold).int()
    tp = int((preds[pos] == 1).sum())
    tn = int((preds[neg] == 0).sum())
    sens = tp / n_pos if n_pos else 0.0
    spec = tn / n_neg if n_neg else 0.0
    prec = tp / int((preds == 1).sum()) if int((preds == 1).sum()) else 0.0
    f1 = 2 * prec * sens / (prec + sens) if (prec + sens) else 0.0
    acc = (tp + tn) / len(labels) if len(labels) else 0.0
    return {
        "auroc": float(binary_auroc(probs, labels)),
        "sensitivity": sens,
        "specificity": spec,
        "f1": f1,
        "accuracy": acc,
    }


def stratified_split(labels, calib_frac: float, seed: int = 42):
    """Split sample indices into (calib, eval) keeping class balance in each."""
    g = torch.Generator().manual_seed(seed)
    calib_idx, eval_idx = [], []
    for cls in (0, 1):
        idx = torch.where(labels == cls)[0]
        idx = idx[torch.randperm(len(idx), generator=g)]
        n_calib = int(round(len(idx) * calib_frac))
        calib_idx.append(idx[:n_calib])
        eval_idx.append(idx[n_calib:])
    return torch.cat(calib_idx), torch.cat(eval_idx)


class TestCurveLogger(pl.Callback):
    """Evaluate the test set after every validation epoch and log test/accuracy,
    test/specificity, test/auroc so the dashboard can plot real test curves.

    The internal val set (from train) saturates near 1.0 and is uninformative;
    the test curve actually shows generalization. This is monitoring only —
    model selection still uses val/loss, so test is not used for tuning.
    """

    def __init__(self, test_dataloader, threshold: float = 0.5) -> None:
        super().__init__()
        self.loader = test_dataloader
        self.threshold = threshold

    @torch.no_grad()
    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if trainer.sanity_checking:
            return
        from torchmetrics.functional.classification import binary_auroc

        was_training = pl_module.training
        pl_module.eval()
        probs, labels = [], []
        for imgs, lbl in self.loader:
            imgs = imgs.to(pl_module.device).contiguous(memory_format=torch.channels_last)
            probs.append(torch.sigmoid(pl_module(imgs)).cpu())
            labels.append(lbl.cpu())
        probs = torch.cat(probs)
        labels = torch.cat(labels).int()
        preds = (probs >= self.threshold).int()
        neg = labels == 0
        acc = (preds == labels).float().mean().item()
        spec = (preds[neg] == 0).float().mean().item() if neg.any() else 0.0

        pl_module.log("test/accuracy", acc)
        pl_module.log("test/specificity", spec)
        pl_module.log("test/auroc", float(binary_auroc(probs, labels)))
        if was_training:
            pl_module.train()


def train(cfg: dict):
    configure_runtime_cache()
    # cudnn.benchmark autotunes conv kernels for fixed input sizes — only safe
    # (and only worth it) when we are NOT requesting deterministic training.
    deterministic = bool(cfg["training"].get("deterministic", False))
    pl.seed_everything(cfg["training"].get("seed", 42), workers=deterministic)
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
        clahe=cfg["data"].get("clahe", False),
        classes=cfg["data"].get("classes"),
    )
    dm.setup()

    run_dir = resolve_run_dir(cfg)

    model = PneumoniaClassifier(
        backbone=cfg["model"]["backbone"],
        pretrained=cfg["model"]["pretrained"],
        dropout=cfg["model"]["dropout"],
        learning_rate=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
        pos_weight=dm.pos_weight,
        threshold=cfg["threshold"]["default"],
        label_smoothing=cfg["training"].get("label_smoothing", 0.0),
        freeze_backbone=cfg["training"].get("freeze_backbone", False),
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

        ckpt_dir = run_dir
        callbacks = [
            # Before DashboardEventLogger so the test metrics it logs are present
            # in callback_metrics when the event logger reads them.
            TestCurveLogger(dm.test_dataloader()),
            DashboardEventLogger(
                output_dir=run_dir,
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

        # The internal val set (split from train) does NOT reflect the shifted
        # Kermany test distribution, so a val-tuned threshold doesn't transfer.
        # Calibrate on a held-out HALF of the test set and report the OTHER
        # half — disjoint, so the reported operating point is honest.
        best_ckpt = trainer.checkpoint_callbacks[0].best_model_path
        if best_ckpt:
            model = PneumoniaClassifier.load_from_checkpoint(
                best_ckpt, pos_weight=dm.pos_weight
            )
        target_sens = cfg["threshold"].get("target_sensitivity", 0.95)

        probs, labels = gather_probs(model, dm.test_dataloader())
        calib_idx, eval_idx = stratified_split(labels, calib_frac=0.4, seed=42)
        threshold = pick_threshold(probs[calib_idx], labels[calib_idx], target_sens)
        model.threshold = threshold

        eval_metrics = metrics_at(probs[eval_idx], labels[eval_idx], threshold)
        eval_at_half = metrics_at(probs[eval_idx], labels[eval_idx], 0.5)
        print(
            f"\n[calibration] threshold={threshold:.3f} on {len(calib_idx)} calib images "
            f"(target sensitivity={target_sens})"
        )
        print(f"[test-eval @ {threshold:.3f}] " + ", ".join(f"{k}={v:.3f}" for k, v in eval_metrics.items()))
        print(f"[test-eval @ 0.500] " + ", ".join(f"{k}={v:.3f}" for k, v in eval_at_half.items()))

        mlflow.log_metric("calibrated_threshold", threshold)
        mlflow.log_metrics({f"test_{k}": v for k, v in eval_metrics.items()})

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
