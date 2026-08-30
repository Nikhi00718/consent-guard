"""Evaluate one or more plate checkpoints on a frozen labeled box challenge.

This reports thresholded IoU matching rather than a score-only transfer proxy.
Predictions are matched greedily to at most one ground-truth box, and every
unmatched prediction counts as a false positive.  The record file is never
used for training or threshold selection by this script.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import torch

from consentguard.shared.paths import project_path
from consentguard.shared.runtime import atomic_json_dump, select_device
from consentguard.stage_02_baseline_model.config import load_training_config
from consentguard.stage_05_review_export.ingest import normalize_image
from consentguard.stage_05_review_export.provider_factory import load_torchvision_provider


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iou(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
    intersection_width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    intersection_height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    intersection = intersection_width * intersection_height
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def _ground_truth(record: dict[str, Any]) -> list[tuple[float, float, float, float]]:
    result = []
    for instance in record.get("instances", []):
        x, y, width, height = (float(value) for value in instance["bbox"])
        if width > 0 and height > 0:
            result.append((x, y, x + width, y + height))
    return result


def _match(
    predictions: list[tuple[float, tuple[float, float, float, float]]],
    truth: list[tuple[float, float, float, float]],
    *,
    score_threshold: float,
    iou_threshold: float,
) -> tuple[int, int, int]:
    unmatched = set(range(len(truth)))
    true_positives = 0
    false_positives = 0
    for score, box in sorted(predictions, key=lambda item: item[0], reverse=True):
        if score < score_threshold:
            continue
        candidates = [(index, _iou(box, truth[index])) for index in unmatched]
        best = max(candidates, key=lambda item: item[1], default=None)
        if best is not None and best[1] >= iou_threshold:
            unmatched.remove(best[0])
            true_positives += 1
        else:
            false_positives += 1
    return true_positives, false_positives, len(unmatched)


def _metrics(true_positives: int, false_positives: int, false_negatives: int, images: int) -> dict[str, Any]:
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives else 0.0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positives_per_image": false_positives / images if images else 0.0,
    }


def _evaluate_candidate(
    label: str,
    config_path: Path,
    checkpoint_path: Path,
    records: list[dict[str, Any]],
    thresholds: list[float],
    iou_threshold: float,
    device: torch.device,
) -> dict[str, Any]:
    config = load_training_config(config_path, require_validation_data=False)
    provider = load_torchvision_provider(config, checkpoint_path, device, provider_name=f"plate_{label}")
    counts = {threshold: [0, 0, 0] for threshold in thresholds}
    per_image = []
    started = time.perf_counter()
    for index, record in enumerate(records, start=1):
        image_path = project_path(record["image_path"])
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        expected_hash = record.get("image_sha256")
        if expected_hash and _sha256(image_path).lower() != str(expected_hash).lower():
            raise RuntimeError(f"Challenge image hash mismatch: {image_path}")
        image = normalize_image(image_path)
        evidence = provider.analyze(image)
        predictions = [
            (float(item.confidence), tuple(float(value) for value in item.geometry.box_xyxy))
            for item in evidence
            if item.geometry.box_xyxy is not None
        ]
        truth = _ground_truth(record)
        image_counts = {}
        for threshold in thresholds:
            matched = _match(
                predictions,
                truth,
                score_threshold=threshold,
                iou_threshold=iou_threshold,
            )
            for position, value in enumerate(matched):
                counts[threshold][position] += value
            image_counts[f"{threshold:.2f}"] = {
                "true_positives": matched[0],
                "false_positives": matched[1],
                "false_negatives": matched[2],
            }
        per_image.append(
            {
                "image_id": record.get("image_id"),
                "image_path": str(image_path),
                "ground_truth_boxes": len(truth),
                "raw_predictions": len(predictions),
                "thresholds": image_counts,
            }
        )
        if index % 25 == 0 or index == len(records):
            print(f"{label}: {index}/{len(records)} images ({(time.perf_counter() - started) / 60:.1f} min)", flush=True)
    result = {
        "label": label,
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "metrics": {
            f"{threshold:.2f}": _metrics(*counts[threshold], images=len(records))
            for threshold in thresholds
        },
        "per_image": per_image,
    }
    del provider
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        nargs=3,
        action="append",
        metavar=("LABEL", "CONFIG", "CHECKPOINT"),
        required=True,
    )
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--thresholds", default="0.25,0.50,0.70")
    parser.add_argument("--iou-threshold", type=float, default=0.50)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args()
    thresholds = sorted({float(value.strip()) for value in args.thresholds.split(",") if value.strip()})
    if not thresholds or any(not 0 <= value <= 1 for value in thresholds):
        raise ValueError("thresholds must contain values in [0, 1]")
    if not 0 < args.iou_threshold <= 1:
        raise ValueError("iou-threshold must be in (0, 1]")

    records_path = args.records.resolve()
    records = [json.loads(line) for line in records_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        raise ValueError(f"Challenge record file is empty: {records_path}")
    device = select_device(args.device)
    candidates = []
    for label, config_arg, checkpoint_arg in args.candidate:
        config_path = project_path(config_arg)
        checkpoint_path = project_path(checkpoint_arg)
        if not config_path.is_file() or not checkpoint_path.is_file():
            raise FileNotFoundError(f"Candidate inputs are missing: {config_path}, {checkpoint_path}")
        candidates.append(
            _evaluate_candidate(
                label,
                config_path,
                checkpoint_path,
                records,
                thresholds,
                args.iou_threshold,
                device,
            )
        )
    payload = {
        "schema_version": "plate-detection-challenge-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "records": str(records_path),
        "records_sha256": _sha256(records_path),
        "images": len(records),
        "ground_truth_boxes": sum(len(_ground_truth(record)) for record in records),
        "iou_threshold": args.iou_threshold,
        "score_thresholds": thresholds,
        "test_split_used_for_training": False,
        "candidates": candidates,
    }
    atomic_json_dump(payload, project_path(args.output))
    print(json.dumps({key: payload[key] for key in ("images", "ground_truth_boxes", "iou_threshold")}, indent=2))


if __name__ == "__main__":
    main()
