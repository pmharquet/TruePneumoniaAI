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

from .backbone import build_backbone, freeze_backbone as apply_backbone_freeze


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
        label_smoothing: float = 0.0,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["pos_weight"])
        self.threshold = threshold
        self.label_smoothing = label_smoothing

        self.model = build_backbone(backbone, pretrained=pretrained, dropout=dropout)
        if freeze_backbone:
            apply_backbone_freeze(self.model, backbone)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        self.train_auroc = AUROC(task="binary")

        self.val_auroc = AUROC(task="binary")
        self.val_recall = BinaryRecall()       # sensitivity
        self.val_spec = BinarySpecificity()
        self.val_f1 = F1Score(task="binary")
        self.val_acc = BinaryAccuracy()

        # Dedicated test metrics so test/val never share state.
        self.test_auroc = AUROC(task="binary")
        self.test_recall = BinaryRecall()
        self.test_spec = BinarySpecificity()
        self.test_f1 = F1Score(task="binary")
        self.test_acc = BinaryAccuracy()

        self._val_logits: list[torch.Tensor] = []
        self._val_labels: list[torch.Tensor] = []
        self._test_logits: list[torch.Tensor] = []
        self._test_labels: list[torch.Tensor] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).squeeze(1)

    def _shared_step(self, batch):
        imgs, labels = batch
        imgs = imgs.contiguous(memory_format=torch.channels_last)
        logits = self(imgs)
        # Label smoothing pulls targets off {0,1} so the model can't drive
        # logits to ±inf — this keeps probabilities calibrated (less
        # overconfidence), which matters under the train->test domain shift.
        if self.label_smoothing > 0:
            target = labels * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        else:
            target = labels
        loss = self.criterion(logits, target)
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

    def _eval_step(self, batch, logits_store, labels_store, loss_key):
        loss, probs, labels = self._shared_step(batch)
        logits_store.append(probs.detach())
        labels_store.append(labels.int().detach())
        self.log(loss_key, loss, on_epoch=True, prog_bar=True)
        return loss

    def _eval_epoch_end(self, prefix, logits_store, labels_store, metrics):
        probs = torch.cat(logits_store)
        labels = torch.cat(labels_store)

        preds = (probs >= self.threshold).int()
        auroc = metrics["auroc"](probs, labels)
        recall = metrics["recall"](preds, labels)
        spec = metrics["spec"](preds, labels)
        f1 = metrics["f1"](preds, labels)
        acc = metrics["acc"](preds, labels)

        self.log_dict({
            f"{prefix}/auroc": auroc,
            f"{prefix}/sensitivity": recall,
            f"{prefix}/specificity": spec,
            f"{prefix}/f1": f1,
            f"{prefix}/accuracy": acc,
        }, prog_bar=True)

        mlflow.log_metrics({
            f"{prefix}_auroc": auroc.item(),
            f"{prefix}_sensitivity": recall.item(),
            f"{prefix}_specificity": spec.item(),
            f"{prefix}_f1": f1.item(),
        }, step=self.current_epoch)

        logits_store.clear()
        labels_store.clear()
        for metric in metrics.values():
            metric.reset()

    @property
    def _val_metrics(self):
        return {
            "auroc": self.val_auroc, "recall": self.val_recall,
            "spec": self.val_spec, "f1": self.val_f1, "acc": self.val_acc,
        }

    @property
    def _test_metrics(self):
        return {
            "auroc": self.test_auroc, "recall": self.test_recall,
            "spec": self.test_spec, "f1": self.test_f1, "acc": self.test_acc,
        }

    def validation_step(self, batch, batch_idx):
        return self._eval_step(batch, self._val_logits, self._val_labels, "val/loss")

    def on_validation_epoch_end(self):
        self._eval_epoch_end("val", self._val_logits, self._val_labels, self._val_metrics)

    def test_step(self, batch, batch_idx):
        return self._eval_step(batch, self._test_logits, self._test_labels, "test/loss")

    def on_test_epoch_end(self):
        self._eval_epoch_end("test", self._test_logits, self._test_labels, self._test_metrics)

    def configure_optimizers(self):
        # Only optimize trainable params — a frozen backbone must not receive
        # weight-decay updates or AdamW moment buffers.
        trainable = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable,
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_epochs
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
