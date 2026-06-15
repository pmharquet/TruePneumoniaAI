"""
Lightning callback that writes live training state for the local dashboard.
"""

from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from pytorch_lightning import Callback


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return float(value.detach().cpu().item())
        return value.detach().cpu().tolist()
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return str(value)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    if not math.isfinite(number):
        return default
    return int(number)


def _trainer_train_batches(trainer) -> int:
    return _safe_int(getattr(trainer, "num_training_batches", 0))


def _trainer_val_batches(trainer) -> int:
    batches = getattr(trainer, "num_val_batches", 0)
    if isinstance(batches, (list, tuple)):
        batches = batches[0] if batches else 0
    return _safe_int(batches)


class DashboardEventLogger(Callback):
    def __init__(
        self,
        output_dir: str | Path | None = None,
        log_every_n_steps: int = 1,
    ) -> None:
        super().__init__()
        self.output_dir = Path(
            output_dir or os.environ.get("TPAI_DASHBOARD_DIR", "outputs/dashboard/current")
        )
        self.log_every_n_steps = max(1, log_every_n_steps)
        self.events_path = self.output_dir / "events.jsonl"
        self.state_path = self.output_dir / "state.json"
        self.started_at = time.time()
        self._state: dict[str, Any] = {
            "status": "initializing",
            "started_at": self.started_at,
            "last_event_at": self.started_at,
            "latest_metrics": {},
        }
        self._initialized = False

    def setup(self, trainer, pl_module, stage: str) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if not self._initialized:
            self.events_path.write_text("", encoding="utf-8")
            self._initialized = True
        self._write_event("setup", {"stage": stage})

    def _metrics(self, trainer) -> dict[str, Any]:
        return {
            key: _to_jsonable(value)
            for key, value in trainer.callback_metrics.items()
            if not str(key).endswith("_step")
        }

    def _write_state(self) -> None:
        payload = json.dumps(self._state, indent=2)

        for attempt in range(8):
            tmp_path = self.output_dir / f"state.{os.getpid()}.{time.time_ns()}.tmp"
            try:
                tmp_path.write_text(payload, encoding="utf-8")
                tmp_path.replace(self.state_path)
                return
            except OSError:
                try:
                    tmp_path.unlink(missing_ok=True)
                except OSError:
                    pass
                time.sleep(0.05 * (attempt + 1))

        # The dashboard may briefly lock state.json on Windows. Missing one state
        # refresh is better than crashing the training run.
        try:
            self.state_path.write_text(payload, encoding="utf-8")
        except OSError:
            pass

    def _write_event(self, event_type: str, payload: dict[str, Any]) -> None:
        now = time.time()
        event = {
            "time": now,
            "type": event_type,
            **_to_jsonable(payload),
        }
        with self.events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")

        self._state.update({
            "status": payload.get("status", self._state.get("status")),
            "last_event_at": now,
            "last_event": event_type,
        })
        if "metrics" in payload:
            self._state["latest_metrics"] = payload["metrics"]
        for key in (
            "epoch",
            "max_epochs",
            "global_step",
            "train_batch",
            "train_batches",
            "val_batch",
            "val_batches",
            "stage",
            "message",
        ):
            if key in payload:
                self._state[key] = payload[key]
        self._write_state()

    def on_fit_start(self, trainer, pl_module) -> None:
        self.started_at = time.time()
        self._state.update({
            "status": "running",
            "started_at": self.started_at,
            "max_epochs": _safe_int(trainer.max_epochs),
            "train_batches": _trainer_train_batches(trainer),
            "val_batches": _trainer_val_batches(trainer),
            "global_step": _safe_int(trainer.global_step),
            "stage": "fit",
        })
        self._write_event("fit_start", self._state)

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        self._write_event("train_epoch_start", {
            "status": "running",
            "stage": "train",
            "epoch": _safe_int(trainer.current_epoch),
            "max_epochs": _safe_int(trainer.max_epochs),
            "train_batches": _trainer_train_batches(trainer),
            "train_batch": 0,
            "global_step": _safe_int(trainer.global_step),
        })

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx: int) -> None:
        if batch_idx % self.log_every_n_steps != 0:
            return

        metrics = self._metrics(trainer)
        lr = None
        if trainer.optimizers:
            lr = trainer.optimizers[0].param_groups[0].get("lr")
            metrics["lr"] = float(lr) if lr is not None else None

        self._write_event("train_batch_end", {
            "status": "running",
            "stage": "train",
            "epoch": _safe_int(trainer.current_epoch),
            "max_epochs": _safe_int(trainer.max_epochs),
            "train_batch": _safe_int(batch_idx + 1),
            "train_batches": _trainer_train_batches(trainer),
            "global_step": _safe_int(trainer.global_step),
            "metrics": metrics,
        })

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        self._write_event("train_epoch_end", {
            "status": "running",
            "stage": "train",
            "epoch": _safe_int(trainer.current_epoch),
            "max_epochs": _safe_int(trainer.max_epochs),
            "global_step": _safe_int(trainer.global_step),
            "metrics": self._metrics(trainer),
        })

    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        self._write_event("validation_epoch_start", {
            "status": "running",
            "stage": "validation",
            "epoch": _safe_int(trainer.current_epoch),
            "val_batch": 0,
            "val_batches": _trainer_val_batches(trainer),
            "global_step": _safe_int(trainer.global_step),
        })

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx: int) -> None:
        self._write_event("validation_batch_end", {
            "status": "running",
            "stage": "validation",
            "epoch": _safe_int(trainer.current_epoch),
            "val_batch": _safe_int(batch_idx + 1),
            "val_batches": _trainer_val_batches(trainer),
            "global_step": _safe_int(trainer.global_step),
        })

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        self._write_event("validation_epoch_end", {
            "status": "running",
            "stage": "validation",
            "epoch": _safe_int(trainer.current_epoch),
            "global_step": _safe_int(trainer.global_step),
            "metrics": self._metrics(trainer),
        })

    def on_test_end(self, trainer, pl_module) -> None:
        self._write_event("test_end", {
            "status": "running",
            "stage": "test",
            "global_step": _safe_int(trainer.global_step),
            "metrics": self._metrics(trainer),
        })

    def on_fit_end(self, trainer, pl_module) -> None:
        self._write_event("fit_end", {
            "status": "completed",
            "stage": "done",
            "epoch": _safe_int(trainer.current_epoch),
            "global_step": _safe_int(trainer.global_step),
            "metrics": self._metrics(trainer),
            "elapsed_seconds": round(time.time() - self.started_at, 2),
        })

    def on_exception(self, trainer, pl_module, exception: BaseException) -> None:
        self._write_event("exception", {
            "status": "failed",
            "stage": "error",
            "message": f"{type(exception).__name__}: {exception}",
            "global_step": _safe_int(trainer.global_step),
            "elapsed_seconds": round(time.time() - self.started_at, 2),
        })
