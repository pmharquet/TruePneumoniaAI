"""
PyTorch Lightning module — wraps the backbone with training/val/test steps.

Metrics: AUC-ROC, sensitivity (recall), specificity, F1.
Threshold tuning at end of validation epoch to meet target sensitivity.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import AUROC, F1Score
from torchmetrics.classification import BinaryRecall, BinarySpecificity, BinaryAccuracy
import mlflow

from .backbone import build_backbone


class PneumoniaClassifier(pl.LightningModule):
    def __init__(
        self,
        backbone: str = "densenet121",
        pretrained: bool = True,
        dropout: float = 0.5,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        pos_weight: torch.Tensor | None = None,
        threshold: float = 0.5,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["pos_weight"])
        self.threshold = threshold

        self.model = build_backbone(backbone, pretrained=pretrained, dropout=dropout)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        self.train_auroc = AUROC(task="binary")
        self.val_auroc = AUROC(task="binary")
        self.val_recall = BinaryRecall()       # sensitivity
        self.val_spec = BinarySpecificity()
        self.val_f1 = F1Score(task="binary")
        self.val_acc = BinaryAccuracy()

        self._val_logits: list[torch.Tensor] = []
        self._val_labels: list[torch.Tensor] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).squeeze(1)

    def _shared_step(self, batch):
        imgs, labels = batch
        logits = self(imgs)
        loss = self.criterion(logits, labels)
        probs = torch.sigmoid(logits)
        return loss, probs, labels

    def training_step(self, batch, batch_idx):
        loss, probs, labels = self._shared_step(batch)
        self.train_auroc.update(probs, labels.int())
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        self.log("train/auroc", self.train_auroc.compute())
        self.train_auroc.reset()

    def validation_step(self, batch, batch_idx):
        loss, probs, labels = self._shared_step(batch)
        self._val_logits.append(probs.detach())
        self._val_labels.append(labels.int().detach())
        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        probs = torch.cat(self._val_logits)
        labels = torch.cat(self._val_labels)

        preds = (probs >= self.threshold).int()
        auroc = self.val_auroc(probs, labels)
        recall = self.val_recall(preds, labels)
        spec = self.val_spec(preds, labels)
        f1 = self.val_f1(preds, labels)
        acc = self.val_acc(preds, labels)

        self.log_dict({
            "val/auroc": auroc,
            "val/sensitivity": recall,
            "val/specificity": spec,
            "val/f1": f1,
            "val/accuracy": acc,
        }, prog_bar=True)

        mlflow.log_metrics({
            "val_auroc": auroc.item(),
            "val_sensitivity": recall.item(),
            "val_specificity": spec.item(),
            "val_f1": f1.item(),
        }, step=self.current_epoch)

        self._val_logits.clear()
        self._val_labels.clear()
        self.val_auroc.reset()
        self.val_recall.reset()
        self.val_spec.reset()
        self.val_f1.reset()
        self.val_acc.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
