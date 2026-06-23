"""Speed/accuracy sweep over training image size.

Trains one regularized DenseNet per image size (224, 128, 64, 32) on the
patient-level split, then reports test accuracy/AUROC and training throughput
(images/s) so the speed-vs-quality trade-off is explicit.

Usage:
    python -m scripts.size_sweep --sizes 224 128 64 32
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import torch

from src.training.train import load_config, train

STATE = Path("outputs/dashboard/current/state.json")


def read_state() -> dict:
    try:
        return json.loads(STATE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/patient.yaml")
    parser.add_argument("--sizes", type=int, nargs="+", default=[224, 128, 64, 32])
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    results = []

    for size in args.sizes:
        cfg = copy.deepcopy(base_cfg)
        cfg["data"]["image_size"] = size
        cfg["paths"]["checkpoints"] = f"checkpoints/size{size}"
        cfg["training"]["seed"] = 42
        cfg["mlflow"]["experiment_name"] = "TruePneumoniaAI-size-sweep"

        print(f"\n{'=' * 60}\n  IMAGE SIZE {size}x{size}\n{'=' * 60}")
        t0 = time.time()
        train(cfg)
        elapsed = time.time() - t0
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        state = read_state()
        perf = state.get("performance", {})
        tm = state.get("test_metrics", {})
        results.append({
            "size": size,
            "accuracy": tm.get("test/accuracy"),
            "auroc": tm.get("test/auroc"),
            "sensitivity": tm.get("test/sensitivity"),
            "specificity": tm.get("test/specificity"),
            "threshold": state.get("test_threshold"),
            "img_per_s": perf.get("img_per_s_avg"),
            "ms_per_batch": perf.get("ms_per_batch"),
            "gpu_mem_peak_mb": perf.get("gpu_mem_peak_mb"),
            "train_seconds": round(elapsed, 1),
        })

    def f(v, d=3):
        return f"{v:.{d}f}" if isinstance(v, (int, float)) else "-"

    print(f"\n\n{'=' * 78}\n  SIZE SWEEP REPORT (patient split, DenseNet121, single seed)\n{'=' * 78}")
    header = f"{'size':>6} | {'accuracy':>8} | {'auroc':>6} | {'sens':>6} | {'spec':>6} | {'img/s':>8} | {'ms/batch':>8} | {'VRAM MB':>8} | {'train s':>8}"
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['size']:>6} | {f(r['accuracy']):>8} | {f(r['auroc']):>6} | "
            f"{f(r['sensitivity']):>6} | {f(r['specificity']):>6} | "
            f"{f(r['img_per_s'], 0):>8} | {f(r['ms_per_batch'], 0):>8} | "
            f"{f(r['gpu_mem_peak_mb'], 0):>8} | {f(r['train_seconds'], 0):>8}"
        )

    out = Path("outputs/size_sweep_report.json")
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
