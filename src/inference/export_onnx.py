"""
Export the trained Lightning model to ONNX for fast inference.

Usage:
    python -m src.inference.export_onnx --ckpt checkpoints/best.ckpt --output exports/model.onnx
"""

import argparse
from pathlib import Path

import torch

from src.models.classifier import PneumoniaClassifier


def export(ckpt_path: str, output_path: str, image_size: int = 224):
    model = PneumoniaClassifier.load_from_checkpoint(ckpt_path)
    model.eval()

    dummy = torch.randn(1, 3, image_size, image_size)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        output_path,
        input_names=["image"],
        output_names=["logit"],
        dynamic_axes={"image": {0: "batch_size"}, "logit": {0: "batch_size"}},
        opset_version=17,
    )
    print(f"Exported to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--output", default="exports/model.onnx")
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()
    export(args.ckpt, args.output, args.image_size)


if __name__ == "__main__":
    main()
