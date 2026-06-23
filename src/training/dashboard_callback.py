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


def _batch_num_samples(batch: Any) -> int:
    """Best-effort count of images in a training batch."""
    sample = batch
    if isinstance(batch, (list, tuple)) and batch:
        sample = batch[0]
    if isinstance(sample, torch.Tensor):
        return int(sample.shape[0]) if sample.ndim else 0
    if isinstance(sample, Mapping):
        for value in sample.values():
            if isinstance(value, torch.Tensor) and value.ndim:
                return int(value.shape[0])
    return 0


def _gpu_memory_mb() -> tuple[float | None, float | None]:
    """Return (allocated, peak) GPU memory in MB, or (None, None) on CPU."""
    if not torch.cuda.is_available():
        return None, None
    allocated = torch.cuda.memory_allocated() / 1024 ** 2
    peak = torch.cuda.max_memory_allocated() / 1024 ** 2
    return round(allocated, 1), round(peak, 1)


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
            "performance": {},
        }
        self._initialized = False
        # Throughput / timing trackers
        self._batch_start: float | None = None
        self._batch_samples: int = 0
        self._img_ema: float | None = None
        self._ms_ema: float | None = None
        self._epoch_samples: int = 0
        self._epoch_time: float = 0.0
        self._total_samples: int = 0

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
        self._epoch_samples = 0
        self._epoch_time = 0.0
        self._write_event("train_epoch_start", {
            "status": "running",
            "stage": "train",
            "epoch": _safe_int(trainer.current_epoch),
            "max_epochs": _safe_int(trainer.max_epochs),
            "train_batches": _trainer_train_batches(trainer),
            "train_batch": 0,
            "global_step": _safe_int(trainer.global_step),
        })

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx: int) -> None:
        self._batch_start = time.perf_counter()
        self._batch_samples = _batch_num_samples(batch)

    def _update_performance(self, trainer, batch_idx: int) -> dict[str, Any]:
        """Compute throughput/timing stats from the batch that just finished."""
        now = time.perf_counter()
        dt = now - self._batch_start if self._batch_start is not None else 0.0
        samples = self._batch_samples
        perf = dict(self._state.get("performance") or {})

        if dt > 0 and samples > 0:
            img_s = samples / dt
            ms = dt * 1000.0
            alpha = 0.3
            self._img_ema = img_s if self._img_ema is None else (1 - alpha) * self._img_ema + alpha * img_s
            self._ms_ema = ms if self._ms_ema is None else (1 - alpha) * self._ms_ema + alpha * ms
            self._epoch_samples += samples
            self._epoch_time += dt
            self._total_samples += samples

        avg_img_s = self._epoch_samples / self._epoch_time if self._epoch_time > 0 else None
        elapsed = time.time() - self.started_at

        # ETA from remaining batches at the current smoothed batch time.
        train_batches = _trainer_train_batches(trainer)
        max_epochs = _safe_int(trainer.max_epochs)
        epoch = _safe_int(trainer.current_epoch)
        eta = None
        if train_batches > 0 and max_epochs > 0 and self._ms_ema:
            done = epoch * train_batches + (batch_idx + 1)
            remaining = max(0, max_epochs * train_batches - done)
            eta = remaining * (self._ms_ema / 1000.0)

        gpu_alloc, gpu_peak = _gpu_memory_mb()
        perf.update({
            "img_per_s": round(self._img_ema, 1) if self._img_ema is not None else None,
            "img_per_s_avg": round(avg_img_s, 1) if avg_img_s is not None else None,
            "ms_per_batch": round(self._ms_ema, 1) if self._ms_ema is not None else None,
            "steps_per_s": round(1000.0 / self._ms_ema, 2) if self._ms_ema else None,
            "gpu_mem_mb": gpu_alloc,
            "gpu_mem_peak_mb": gpu_peak,
            "elapsed_seconds": round(elapsed, 1),
            "eta_seconds": round(eta, 1) if eta is not None else None,
            "total_images": self._total_samples,
        })
        self._state["performance"] = perf
        return perf

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx: int) -> None:
        perf = self._update_performance(trainer, batch_idx)

        if batch_idx % self.log_every_n_steps != 0:
            return

        metrics = self._metrics(trainer)
        lr = None
        if trainer.optimizers:
            lr = trainer.optimizers[0].param_groups[0].get("lr")
            metrics["lr"] = float(lr) if lr is not None else None
        if perf.get("img_per_s") is not None:
            metrics["perf/img_per_s"] = perf["img_per_s"]

        self._write_event("train_batch_end", {
            "status": "running",
            "stage": "train",
            "epoch": _safe_int(trainer.current_epoch),
            "max_epochs": _safe_int(trainer.max_epochs),
            "train_batch": _safe_int(batch_idx + 1),
            "train_batches": _trainer_train_batches(trainer),
            "global_step": _safe_int(trainer.global_step),
            "metrics": metrics,
            "performance": perf,
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
