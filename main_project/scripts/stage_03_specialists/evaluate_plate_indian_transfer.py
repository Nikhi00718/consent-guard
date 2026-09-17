"""Compare plate checkpoints on the local Indian INDO-ALPR image set.

INDO-ALPR originals do not ship with localization annotations in this project,
so this is a transfer/sanity evaluation, not an accuracy or mAP benchmark.  It
reports score stability and box geometry separately for the train/test folders
and for scene-like versus crop-like images.  The checkpoint is loaded one at a
time to keep laptop GPU memory bounded.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

# Make direct ``python main_project/scripts/...`` execution work without an
# editable installation of the package.
SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch
from PIL import Image

from consentguard.shared.paths import project_path
from consentguard.shared.runtime import atomic_json_dump, select_device
from consentguard.stage_02_baseline_model.config import load_training_config
from consentguard.stage_05_review_export.ingest import normalize_image
from consentguard.stage_05_review_export.provider_factory import load_torchvision_provider


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inventory(root: Path, max_images: int | None) -> list[dict[str, Any]]:
    paths = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    if max_images is not None:
        if max_images < 1:
            raise ValueError("max_images must be positive")
        paths = paths[:max_images]
    records: list[dict[str, Any]] = []
    for path in paths:
        split = path.parent.name.lower() if path.parent.name.lower() in {"train", "test", "val", "validation"} else "unknown"
        try:
            with Image.open(path) as image:
                width, height = int(image.width), int(image.height)
        except Exception as error:  # pragma: no cover - defensive inventory guard
            records.append({"path": path, "split": split, "error": f"inventory:{type(error).__name__}: {error}"})
            continue
        records.append(
            {
                "path": path,
                "split": split,
                "width": width,
                "height": height,
                # These are dataset-shape categories, not labels.
                "scene_like": bool(width >= 640 and height >= 360),
                "crop_like": bool(width / max(height, 1) >= 2.0),
            }
        )
    return records


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    index = min(len(values) - 1, max(0, round((len(values) - 1) * q)))
    return float(values[index])


def _aggregate(items: list[dict[str, Any]], thresholds: list[float]) -> dict[str, Any]:
    valid = [item for item in items if not item.get("error")]
    confidences = [float(item["max_confidence"]) for item in valid]
    areas = [float(item["top_box_area_ratio"]) for item in valid if item.get("top_box_area_ratio") is not None]
    result: dict[str, Any] = {
        "images": len(items),
        "valid_images": len(valid),
        "errors": len(items) - len(valid),
        "detections_any": sum(int(item["detection_count"]) > 0 for item in valid),
        "detection_rate_any": (sum(int(item["detection_count"]) > 0 for item in valid) / len(valid)) if valid else None,
        "max_confidence_mean": (sum(confidences) / len(confidences)) if confidences else None,
        "max_confidence_median": median(confidences) if confidences else None,
        "max_confidence_p90": _percentile(confidences, 0.90),
        "top_box_area_ratio_mean": (sum(areas) / len(areas)) if areas else None,
        "top_box_area_ratio_median": median(areas) if areas else None,
        "top_box_area_ratio_p90": _percentile(areas, 0.90),
        "thresholds": {},
    }
    for threshold in thresholds:
        key = f"{threshold:.2f}"
        count = sum(bool(item["thresholds"].get(key, False)) for item in valid)
        result["thresholds"][key] = {
            "images_with_detection": count,
            "detection_rate": (count / len(valid)) if valid else None,
        }
    return result


def _grouped_summary(items: list[dict[str, Any]], thresholds: list[float]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {"all": items}
    for split in sorted({str(item.get("split", "unknown")) for item in items}):
        groups[f"split:{split}"] = [item for item in items if item.get("split") == split]
    groups["scene_like"] = [item for item in items if item.get("scene_like")]
    groups["crop_like"] = [item for item in items if item.get("crop_like")]
    groups["scene_like_and_crop_like"] = [item for item in items if item.get("scene_like") and item.get("crop_like")]
    return {name: _aggregate(group, thresholds) for name, group in groups.items()}


def _box_area_ratio(evidence: object, width: int, height: int) -> float | None:
    geometry = getattr(evidence, "geometry", None)
    box = getattr(geometry, "box_xyxy", None)
    if box is None:
        return None
    left, top, right, bottom = (float(value) for value in box)
    return max(0.0, min(1.0, ((right - left) * (bottom - top)) / max(width * height, 1)))


def _evaluate_one_model(
    *,
    label: str,
    config_path: Path,
    checkpoint_path: Path,
    inventory: list[dict[str, Any]],
    device: torch.device,
    thresholds: list[float],
) -> dict[str, Any]:
    config = load_training_config(config_path, require_validation_data=False)
    print(f"Loading {label} ({config.section('model')['name']}) on {device}...", flush=True)
    provider = load_torchvision_provider(config, checkpoint_path, device, provider_name=label)
    per_image: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, record in enumerate(inventory, start=1):
        base = {
            "path": str(record["path"]),
            "split": record.get("split", "unknown"),
            "width": record.get("width"),
            "height": record.get("height"),
            "scene_like": bool(record.get("scene_like", False)),
            "crop_like": bool(record.get("crop_like", False)),
        }
        if record.get("error"):
            base["error"] = str(record["error"])
            per_image.append(base)
            continue
        try:
            image = normalize_image(record["path"])
            evidence = list(provider.analyze(image))
            confidences = [float(item.confidence) for item in evidence]
            max_confidence = max(confidences, default=0.0)
            top = max(evidence, key=lambda item: float(item.confidence), default=None)
            base.update(
                {
                    "detection_count": len(evidence),
                    "max_confidence": max_confidence,
                    "thresholds": {f"{threshold:.2f}": bool(max_confidence >= threshold) for threshold in thresholds},
                    "top_box_area_ratio": _box_area_ratio(top, image.width, image.height) if top else None,
                }
            )
        except Exception as error:  # Keep the batch auditable if one file is malformed.
            base.update(
                {
                    "error": f"{type(error).__name__}: {error}",
                    "detection_count": 0,
                    "max_confidence": 0.0,
                    "thresholds": {f"{threshold:.2f}": False for threshold in thresholds},
                    "top_box_area_ratio": None,
                }
            )
        per_image.append(base)
        if index % 100 == 0 or index == len(inventory):
            elapsed = time.perf_counter() - started
            print(f"{label}: {index}/{len(inventory)} images ({elapsed / 60:.1f} min)", flush=True)

    summary = {
        "label": label,
        "model_name": str(config.section("model")["name"]),
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "checkpoint_sha256": _sha256(checkpoint_path),
        "summary": _grouped_summary(per_image, thresholds),
        "per_image": per_image,
    }
    del provider
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-config", default="main_project/configs/stage_03_specialists/train_plate_ccpd2020_fasterrcnn.yaml")
    parser.add_argument("--new-checkpoint", default="artifacts/checkpoints/specialist_plate_ccpd2020_fasterrcnn/best.pt")
    parser.add_argument("--old-config", default="main_project/configs/stage_03_specialists/train_plate_maskrcnn_5ep.yaml")
    parser.add_argument("--old-checkpoint", default="artifacts/checkpoints/specialist_plate_maskrcnn_5ep/last.pt")
    parser.add_argument("--image-root", default="data/raw/indo_alpr/original")
    parser.add_argument("--output", default="artifacts/evaluations/plate_indian_transfer_2026-08-28.json")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--thresholds", default="0.50,0.70,0.90")
    args = parser.parse_args()

    thresholds = sorted({float(value.strip()) for value in args.thresholds.split(",") if value.strip()})
    if not thresholds or any(not 0.0 <= value <= 1.0 for value in thresholds):
        raise ValueError("thresholds must be comma-separated values in [0, 1]")
    image_root = project_path(args.image_root)
    inventory = _inventory(image_root, args.max_images)
    if not inventory:
        raise ValueError(f"No supported images found below {image_root}")
    device = select_device(args.device)
    print(f"Inventory: {len(inventory)} images under {image_root}; device={device}", flush=True)

    models = []
    for label, config_arg, checkpoint_arg in (
        ("plate_ccpd2020_fasterrcnn", args.new_config, args.new_checkpoint),
        ("plate_india_maskrcnn_baseline", args.old_config, args.old_checkpoint),
    ):
        config_path = project_path(config_arg)
        checkpoint_path = project_path(checkpoint_arg)
        if not config_path.is_file():
            raise FileNotFoundError(config_path)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        models.append(
            _evaluate_one_model(
                label=label,
                config_path=config_path,
                checkpoint_path=checkpoint_path,
                inventory=inventory,
                device=device,
                thresholds=thresholds,
            )
        )

    output = project_path(args.output)
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_type": "qualitative_transfer_sanity_check",
        "accuracy_claim_allowed": False,
        "ground_truth_annotations_available": False,
        "image_root": str(image_root),
        "image_count": len(inventory),
        "inventory_counts": {
            "by_split": dict(sorted(Counter(str(item.get("split", "unknown")) for item in inventory).items())),
            "scene_like": sum(bool(item.get("scene_like")) for item in inventory),
            "crop_like": sum(bool(item.get("crop_like")) for item in inventory),
        },
        "definition_scene_like": "width >= 640 and height >= 360",
        "definition_crop_like": "width / height >= 2.0",
        "score_thresholds": thresholds,
        "models": models,
    }
    atomic_json_dump(payload, output)
    print(f"Wrote {output}", flush=True)


if __name__ == "__main__":
    main()
