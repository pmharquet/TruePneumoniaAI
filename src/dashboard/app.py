"""
FastAPI dashboard for controlling and monitoring TruePneumoniaAI training.

Usage:
    uvicorn src.dashboard.app:app --host 127.0.0.1 --port 8501
"""

from __future__ import annotations

import base64
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
STATIC_DIR = Path(__file__).resolve().parent / "static"
DASHBOARD_ROOT = ROOT / "outputs" / "dashboard"
RUNS_DIR = DASHBOARD_ROOT / "runs"
LATEST_RUN_FILE = DASHBOARD_ROOT / "latest_run.txt"
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
DATASETS = ("chest_Xray_augmented", "chest_Xray")
CLASSES = ("NORMAL", "PNEUMONIA")
SPLITS = ("train", "val", "test")
PRETRAINED_WEIGHTS = {
    "densenet121": "densenet121-a639ec97.pth",
    "resnet50": "resnet50-11ad3fa6.pth",
}


class TrainingProcess:
    def __init__(self) -> None:
        self.process: subprocess.Popen[str] | None = None
        self.run_dir: Path | None = None
        self.started_at: float | None = None
        self.log_lines: list[str] = []
        self.lock = threading.Lock()

    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def returncode(self) -> int | None:
        return self.process.poll() if self.process is not None else None

    def append_log(self, line: str) -> None:
        with self.lock:
            self.log_lines.append(line.rstrip())
            self.log_lines = self.log_lines[-300:]

    def tail_logs(self, count: int = 120) -> list[str]:
        with self.lock:
            return self.log_lines[-count:]


TRAINING = TrainingProcess()
app = FastAPI(title="TruePneumoniaAI Dashboard", version="0.1.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.middleware("http")
async def no_cache(request: Request, call_next):
    """Disable browser caching so static assets always reflect the latest build."""
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return response


def _now_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def _safe_relative_path(path: Path) -> Path:
    resolved = (ROOT / path).resolve() if not path.is_absolute() else path.resolve()
    if ROOT.resolve() not in (resolved, *resolved.parents):
        raise HTTPException(status_code=400, detail=f"Path outside project: {path}")
    return resolved


def _dataset_counts(dataset: Path) -> dict[str, Any]:
    counts: dict[str, dict[str, int]] = {}
    total = 0
    for split in SPLITS:
        counts[split] = {}
        for class_name in CLASSES:
            class_dir = dataset / split / class_name
            count = 0
            if class_dir.exists():
                count = sum(
                    1 for item in class_dir.iterdir()
                    if item.is_file() and item.suffix.lower() in IMAGE_EXTS
                )
            counts[split][class_name] = count
            total += count

    summary_path = dataset / "augmentation_summary.json"
    summary = None
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            summary = None

    return {
        "name": dataset.name,
        "path": str(dataset.relative_to(ROOT)) if ROOT in (dataset, *dataset.parents) else str(dataset),
        "exists": dataset.exists(),
        "total": total,
        "counts": counts,
        "summary": summary,
    }


def _available_datasets() -> list[dict[str, Any]]:
    return [_dataset_counts(ROOT / name) for name in DATASETS]


def _default_data_dir() -> str:
    augmented = ROOT / "chest_Xray_augmented"
    return "chest_Xray_augmented" if augmented.exists() else "chest_Xray"


def _read_state(run_dir: Path | None) -> dict[str, Any]:
    if run_dir is None:
        run_dir = _latest_run_dir()
    if run_dir is None:
        return {}

    state_path = run_dir / "state.json"
    if not state_path.exists():
        return {"status": "starting", "run_dir": str(run_dir.relative_to(ROOT))}
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        state = {"status": "starting"}
    state["run_dir"] = str(run_dir.relative_to(ROOT))
    return state


def _read_events(run_dir: Path | None, limit: int = 500) -> list[dict[str, Any]]:
    if run_dir is None:
        run_dir = _latest_run_dir()
    if run_dir is None:
        return []

    events_path = run_dir / "events.jsonl"
    if not events_path.exists():
        return []

    lines = events_path.read_text(encoding="utf-8", errors="replace").splitlines()
    events: list[dict[str, Any]] = []
    for line in lines[-limit:]:
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def _latest_run_dir() -> Path | None:
    if TRAINING.run_dir is not None:
        return TRAINING.run_dir
    if LATEST_RUN_FILE.exists():
        candidate = (ROOT / LATEST_RUN_FILE.read_text(encoding="utf-8").strip()).resolve()
        if candidate.exists() and ROOT.resolve() in (candidate, *candidate.parents):
            return candidate
    if RUNS_DIR.exists():
        runs = sorted((p for p in RUNS_DIR.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime)
        if runs:
            return runs[-1]
    return None


def _tail_file(path: Path, lines: int = 120) -> list[str]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return text[-lines:]


def _artifact_snapshot(cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = cfg or _load_yaml(DEFAULT_CONFIG)
    ckpt_dir = _safe_relative_path(Path(cfg["paths"]["checkpoints"]))
    export_path = _safe_relative_path(Path(cfg["paths"]["onnx_export"]))
    mlruns = _safe_relative_path(Path(cfg["mlflow"]["tracking_uri"]))
    torch_cache = ROOT / "outputs" / "cache" / "torch" / "hub" / "checkpoints"

    checkpoints = []
    if ckpt_dir.exists():
        for path in sorted(ckpt_dir.glob("*.ckpt"), key=lambda p: p.stat().st_mtime, reverse=True):
            checkpoints.append({
                "name": path.name,
                "path": str(path.relative_to(ROOT)),
                "size_mb": round(path.stat().st_size / 1024 / 1024, 2),
                "modified": path.stat().st_mtime,
            })

    return {
        "checkpoints": checkpoints,
        "onnx": {
            "path": str(export_path.relative_to(ROOT)) if ROOT in (export_path, *export_path.parents) else str(export_path),
            "exists": export_path.exists(),
            "size_mb": round(export_path.stat().st_size / 1024 / 1024, 2) if export_path.exists() else None,
        },
        "mlflow": {
            "tracking_uri": cfg["mlflow"]["tracking_uri"],
            "exists": mlruns.exists(),
        },
        "pretrained_weights": {
            name: {
                "file": filename,
                "exists": (torch_cache / filename).exists(),
                "path": str((torch_cache / filename).relative_to(ROOT)),
            }
            for name, filename in PRETRAINED_WEIGHTS.items()
        },
    }


def _system_snapshot() -> dict[str, Any]:
    usage = shutil.disk_usage(ROOT)
    gpu: list[dict[str, Any]] = []
    cuda_available = False
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        if cuda_available:
            for index in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(index)
                gpu.append({
                    "index": index,
                    "name": props.name,
                    "total_vram_gb": round(props.total_memory / 1024 ** 3, 2),
                    "allocated_gb": round(torch.cuda.memory_allocated(index) / 1024 ** 3, 2),
                    "reserved_gb": round(torch.cuda.memory_reserved(index) / 1024 ** 3, 2),
                })
    except Exception as exc:
        gpu.append({"error": str(exc)})

    return {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "cuda_available": cuda_available,
        "gpu": gpu,
        "disk": {
            "total_gb": round(usage.total / 1024 ** 3, 1),
            "free_gb": round(usage.free / 1024 ** 3, 1),
        },
    }


def _apply_overrides(cfg: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    cfg = json.loads(json.dumps(cfg))
    cfg["data"]["data_dir"] = overrides.get("data_dir") or _default_data_dir()

    for key in ("batch_size", "num_workers", "image_size"):
        if key in overrides and overrides[key] not in (None, ""):
            cfg["data"][key] = int(overrides[key])

    for key in ("backbone", "pretrained", "dropout"):
        if key in overrides and overrides[key] not in (None, ""):
            cfg["model"][key] = overrides[key]

    for key in ("max_epochs",):
        if key in overrides and overrides[key] not in (None, ""):
            cfg["training"][key] = int(overrides[key])

    for key in ("learning_rate", "weight_decay"):
        if key in overrides and overrides[key] not in (None, ""):
            cfg["training"][key] = float(overrides[key])

    if "precision" in overrides and overrides["precision"]:
        cfg["training"]["precision"] = overrides["precision"]

    cfg.setdefault("dashboard", {})
    cfg["dashboard"]["log_every_n_steps"] = int(overrides.get("log_every_n_steps") or 10)
    return cfg


def _read_process_output(process: subprocess.Popen[str], run_dir: Path) -> None:
    log_path = run_dir / "train.log"
    with log_path.open("a", encoding="utf-8", errors="replace") as log:
        assert process.stdout is not None
        for line in process.stdout:
            TRAINING.append_log(line)
            log.write(line)
            log.flush()
    code = process.wait()
    TRAINING.append_log(f"[dashboard] training process exited with code {code}")


def _running_status() -> dict[str, Any]:
    running = TRAINING.is_running()
    run_dir = TRAINING.run_dir or _latest_run_dir()
    state = _read_state(run_dir)
    logs = TRAINING.tail_logs()
    if not logs and run_dir is not None:
        logs = _tail_file(run_dir / "train.log")

    if not running and TRAINING.process is not None:
        code = TRAINING.returncode()
        state.setdefault("status", "completed" if code == 0 else "failed")
        state["returncode"] = code

    return {
        "running": running,
        "returncode": TRAINING.returncode(),
        "run_dir": str(run_dir.relative_to(ROOT)) if run_dir else None,
        "state": state,
        "events": _read_events(run_dir, limit=500),
        "logs": logs,
        "artifacts": _artifact_snapshot(),
        "system": _system_snapshot(),
    }


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/project")
def project() -> dict[str, Any]:
    cfg = _load_yaml(DEFAULT_CONFIG)
    cfg["data"]["data_dir"] = _default_data_dir()
    return {
        "root": str(ROOT),
        "config": cfg,
        "datasets": _available_datasets(),
        "artifacts": _artifact_snapshot(cfg),
        "system": _system_snapshot(),
    }


@app.get("/api/status")
def status() -> dict[str, Any]:
    return _running_status()


@app.post("/api/train/start")
async def start_training(request: Request) -> dict[str, Any]:
    if TRAINING.is_running():
        raise HTTPException(status_code=409, detail="Training is already running.")

    overrides = await request.json()
    base_cfg = _load_yaml(DEFAULT_CONFIG)
    cfg = _apply_overrides(base_cfg, overrides)
    data_dir = _safe_relative_path(Path(cfg["data"]["data_dir"]))
    if not data_dir.exists():
        raise HTTPException(status_code=400, detail=f"Dataset not found: {cfg['data']['data_dir']}")

    backbone = cfg["model"]["backbone"]
    if cfg["model"].get("pretrained", False) and backbone in PRETRAINED_WEIGHTS:
        weights_path = ROOT / "outputs" / "cache" / "torch" / "hub" / "checkpoints" / PRETRAINED_WEIGHTS[backbone]
        if not weights_path.exists():
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Pretrained weights are missing for {backbone}: {weights_path}. "
                    "Download them once or disable ImageNet pretrained."
                ),
            )

    run_id = _now_id()
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    run_config = run_dir / "config.yaml"
    _write_yaml(run_config, cfg)
    LATEST_RUN_FILE.parent.mkdir(parents=True, exist_ok=True)
    LATEST_RUN_FILE.write_text(str(run_dir.relative_to(ROOT)), encoding="utf-8")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    env["PYTHONUNBUFFERED"] = "1"
    env["NO_ALBUMENTATIONS_UPDATE"] = "1"
    env["MLFLOW_ALLOW_FILE_STORE"] = "true"
    env["TPAI_DASHBOARD_DIR"] = str(run_dir)
    cache_root = ROOT / "outputs" / "cache"
    torch_home = cache_root / "torch"
    xdg_cache = cache_root / "xdg"
    torch_home.mkdir(parents=True, exist_ok=True)
    xdg_cache.mkdir(parents=True, exist_ok=True)
    env["TPAI_CACHE_DIR"] = str(cache_root)
    env["TORCH_HOME"] = str(torch_home)
    env["XDG_CACHE_HOME"] = str(xdg_cache)

    command = [
        sys.executable,
        "-m",
        "src.training.train",
        "--config",
        str(run_config),
    ]
    (run_dir / "command.json").write_text(json.dumps(command, indent=2), encoding="utf-8")

    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        errors="replace",
    )
    TRAINING.process = process
    TRAINING.run_dir = run_dir
    TRAINING.started_at = time.time()
    TRAINING.log_lines = [f"[dashboard] started run {run_id}"]
    thread = threading.Thread(target=_read_process_output, args=(process, run_dir), daemon=True)
    thread.start()

    return {"started": True, "run_id": run_id, "run_dir": str(run_dir.relative_to(ROOT))}


@app.post("/api/train/stop")
def stop_training() -> dict[str, Any]:
    if not TRAINING.is_running():
        return {"stopped": False, "message": "No training process is running."}
    assert TRAINING.process is not None
    TRAINING.process.terminate()
    TRAINING.append_log("[dashboard] termination requested")
    return {"stopped": True}


@app.get("/api/dataset/sample")
def dataset_sample(
    dataset: str = Query(default="chest_Xray_augmented"),
    split: str = Query(default="train"),
    class_name: str = Query(default="NORMAL"),
    limit: int = Query(default=4, ge=1, le=12),
) -> dict[str, Any]:
    dataset_path = _safe_relative_path(Path(dataset))
    class_dir = dataset_path / split / class_name
    if not class_dir.exists():
        raise HTTPException(status_code=404, detail=f"Missing class directory: {class_dir}")

    images = sorted(
        path for path in class_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )[:limit]
    payload = []
    for path in images:
        with path.open("rb") as f:
            encoded = base64.b64encode(f.read()).decode("ascii")
        payload.append({
            "name": path.name,
            "src": f"data:image/jpeg;base64,{encoded}",
        })
    return {"images": payload}


@app.websocket("/ws/status")
async def websocket_status(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            await websocket.send_json(_running_status())
            await asyncio_sleep(1.0)
    except WebSocketDisconnect:
        return


async def asyncio_sleep(seconds: float) -> None:
    import asyncio

    await asyncio.sleep(seconds)
