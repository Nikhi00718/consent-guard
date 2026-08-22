"""Run a 100-sample data-loader integrity smoke test."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from consentguard.stage_01_data.dataset import VisualRedactionsDataset


def main() -> None:
    dataset = VisualRedactionsDataset(
        ROOT / "data" / "processed" / "visual_redactions_verified_visual" / "records_train2017.jsonl",
        short_side=512,
        max_long_side=1024,
        limit=100,
    )
    total_instances = 0
    total_positive_pixels = 0
    shapes = set()
    ids = set()

    for index in range(len(dataset)):
        image, target = dataset[index]
        if image.dtype != torch.float32 or image.ndim != 3 or image.shape[0] != 3:
            raise ValueError(f"bad image tensor at index {index}: {tuple(image.shape)} {image.dtype}")
        if float(image.min()) < 0 or float(image.max()) > 1:
            raise ValueError(f"image range outside [0, 1] at index {index}")
        instance_count = target["labels"].shape[0]
        if target["boxes"].shape != (instance_count, 4):
            raise ValueError(f"box/label mismatch at index {index}")
        if target["masks"].shape[0] != instance_count:
            raise ValueError(f"mask/label mismatch at index {index}")
        if torch.any(target["area"] <= 0):
            raise ValueError(f"empty mask at index {index}")
        height, width = image.shape[1:]
        if torch.any(target["boxes"][:, 0::2] < 0) or torch.any(target["boxes"][:, 0::2] > width):
            raise ValueError(f"x box outside image at index {index}")
        if torch.any(target["boxes"][:, 1::2] < 0) or torch.any(target["boxes"][:, 1::2] > height):
            raise ValueError(f"y box outside image at index {index}")
        total_instances += instance_count
        total_positive_pixels += int(target["masks"].sum())
        shapes.add(f"{width}x{height}")
        ids.add(target["image_key"])

    report = {
        "samples_checked": len(dataset),
        "unique_image_ids": len(ids),
        "total_instances": total_instances,
        "total_positive_mask_pixels": total_positive_pixels,
        "resized_shapes": sorted(shapes),
        "passed": True,
    }
    output = ROOT / "reports" / "loader_smoke_test.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
