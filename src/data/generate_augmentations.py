"""
Generate an offline augmented chest X-ray dataset.

Default behavior is conservative:
- reads from chest_Xray/
- writes to chest_Xray_augmented/
- copies train/val/test as letterboxed 224x224 JPEGs
- augments only train/
- balances train classes by augmenting the minority class only

Usage:
    python -m src.data.generate_augmentations
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from src.data.transforms import get_offline_resize_transform, get_offline_train_transforms


CLASSES = ("NORMAL", "PNEUMONIA")
SPLITS = ("train", "val", "test")
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}


def list_images(class_dir: Path) -> list[Path]:
    return sorted(
        path for path in class_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )


def read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        return np.array(img)


def save_rgb(path: Path, image: np.ndarray, quality: int) -> None:
    Image.fromarray(image).save(path, format="JPEG", quality=quality, optimize=True)


def source_key(path: Path, input_root: Path) -> str:
    rel = path.relative_to(input_root).as_posix()
    digest = hashlib.sha1(rel.encode("utf-8")).hexdigest()[:10]
    return f"{path.stem}__{digest}"


def ensure_output_dir(output_root: Path, input_root: Path, overwrite: bool) -> None:
    output_root_resolved = output_root.resolve()
    input_root_resolved = input_root.resolve()

    if output_root_resolved == input_root_resolved:
        raise ValueError("Output directory must be different from input directory.")
    if input_root_resolved in output_root_resolved.parents:
        raise ValueError("Output directory must not be inside the input dataset.")
    if output_root_resolved in input_root_resolved.parents:
        raise ValueError("Output directory must not contain the input dataset.")

    if output_root.exists() and any(output_root.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"{output_root} already exists and is not empty. "
                "Use --overwrite or choose another --output."
            )
        generated_markers = [
            output_root / "augmentation_summary.json",
            output_root / "augmentation_manifest.csv",
        ]
        if not any(marker.exists() for marker in generated_markers):
            raise ValueError(
                f"Refusing to overwrite {output_root}: no augmentation manifest found."
            )
        shutil.rmtree(output_root)

    output_root.mkdir(parents=True, exist_ok=True)


def write_record(
    writer: csv.DictWriter,
    input_root: Path,
    output_root: Path,
    source: Path,
    output: Path,
    split: str,
    class_name: str,
    kind: str,
    augmentation_index: int,
    image_size: int,
) -> None:
    writer.writerow({
        "split": split,
        "class": class_name,
        "kind": kind,
        "augmentation_index": augmentation_index,
        "source_path": source.relative_to(input_root).as_posix(),
        "output_path": output.relative_to(output_root).as_posix(),
        "width": image_size,
        "height": image_size,
    })


def generate_dataset(
    input_root: Path,
    output_root: Path,
    image_size: int,
    mode: str,
    copies_per_image: int,
    seed: int,
    jpeg_quality: int,
    overwrite: bool,
) -> dict:
    random.seed(seed)
    np.random.seed(seed)

    ensure_output_dir(output_root, input_root, overwrite)

    resize_tf = get_offline_resize_transform(image_size)
    aug_tf = get_offline_train_transforms(image_size)
    manifest_path = output_root / "augmentation_manifest.csv"
    summary_counts: dict[str, Counter] = defaultdict(Counter)
    source_counts: dict[str, dict[str, int]] = defaultdict(dict)

    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "split",
            "class",
            "kind",
            "augmentation_index",
            "source_path",
            "output_path",
            "width",
            "height",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        train_sources: dict[str, list[Path]] = {}

        for split in SPLITS:
            for class_name in CLASSES:
                src_dir = input_root / split / class_name
                if not src_dir.exists():
                    raise FileNotFoundError(f"Expected directory: {src_dir}")

                out_dir = output_root / split / class_name
                out_dir.mkdir(parents=True, exist_ok=True)

                sources = list_images(src_dir)
                source_counts[split][class_name] = len(sources)
                if split == "train":
                    train_sources[class_name] = sources

                for source in sources:
                    image = read_rgb(source)
                    resized = resize_tf(image=image)["image"]
                    key = source_key(source, input_root)
                    out_path = out_dir / f"{key}__orig.jpg"
                    save_rgb(out_path, resized, jpeg_quality)
                    summary_counts[f"{split}/{class_name}"]["original"] += 1
                    write_record(
                        writer,
                        input_root,
                        output_root,
                        source,
                        out_path,
                        split,
                        class_name,
                        "original",
                        0,
                        image_size,
                    )

        if mode == "fixed":
            plan = {
                class_name: len(sources) * copies_per_image
                for class_name, sources in train_sources.items()
            }
        elif mode == "balance":
            target = max(len(sources) for sources in train_sources.values())
            plan = {
                class_name: target - len(sources)
                for class_name, sources in train_sources.items()
            }
        else:
            raise ValueError(f"Unknown mode: {mode}")

        for class_name, total_to_generate in plan.items():
            if total_to_generate <= 0:
                continue

            sources = train_sources[class_name]
            out_dir = output_root / "train" / class_name

            for index in range(total_to_generate):
                source = sources[index % len(sources)]
                image = read_rgb(source)
                augmented = aug_tf(image=image)["image"]
                key = source_key(source, input_root)
                aug_number = (index // len(sources)) + 1
                out_path = out_dir / f"{key}__aug{aug_number:02d}.jpg"

                while out_path.exists():
                    aug_number += 1
                    out_path = out_dir / f"{key}__aug{aug_number:02d}.jpg"

                save_rgb(out_path, augmented, jpeg_quality)
                summary_counts[f"train/{class_name}"]["augmented"] += 1
                write_record(
                    writer,
                    input_root,
                    output_root,
                    source,
                    out_path,
                    "train",
                    class_name,
                    "augmented",
                    aug_number,
                    image_size,
                )

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_root": str(input_root),
        "output_root": str(output_root),
        "image_size": image_size,
        "resize_strategy": "letterbox",
        "mode": mode,
        "copies_per_image": copies_per_image if mode == "fixed" else None,
        "seed": seed,
        "jpeg_quality": jpeg_quality,
        "source_counts": source_counts,
        "output_counts": {
            key: dict(counts)
            for key, counts in sorted(summary_counts.items())
        },
        "manifest": str(manifest_path),
    }

    with (output_root / "augmentation_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="chest_Xray", type=Path)
    parser.add_argument("--output", default="chest_Xray_augmented", type=Path)
    parser.add_argument("--image-size", default=224, type=int)
    parser.add_argument("--mode", choices=("balance", "fixed"), default="balance")
    parser.add_argument("--copies-per-image", default=1, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--jpeg-quality", default=95, type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    summary = generate_dataset(
        input_root=args.input,
        output_root=args.output,
        image_size=args.image_size,
        mode=args.mode,
        copies_per_image=args.copies_per_image,
        seed=args.seed,
        jpeg_quality=args.jpeg_quality,
        overwrite=args.overwrite,
    )

    print(f"Generated dataset: {summary['output_root']}")
    for key, counts in summary["output_counts"].items():
        total = sum(counts.values())
        details = ", ".join(f"{kind}={count}" for kind, count in sorted(counts.items()))
        print(f"  {key}: {total} ({details})")
    print(f"Summary: {Path(summary['output_root']) / 'augmentation_summary.json'}")
    print(f"Manifest: {summary['manifest']}")


if __name__ == "__main__":
    main()
