"""Evaluate the fused evidence bundle on V2 validation records only.

The evaluator measures privacy-oriented coverage after provider thresholds and
fusion, rather than reporting detector mAP alone.  It intentionally refuses
test records and writes an explicit ``test_split_used: false`` provenance flag.
The optional specialist and OpenCV providers are loaded exactly as they are in
the review pipeline, so this report is also an integration check.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from consentguard.shared.paths import project_path
from consentguard.shared.runtime import select_device
from consentguard.stage_03_specialists.orchestrator import AnalysisOrchestrator
from consentguard.stage_04_fusion_calibration.evidence import EvidenceFusion, ThresholdRegistry
from consentguard.stage_04_fusion_calibration.evidence.geometry import decode_binary_mask
from consentguard.stage_05_review_export.ingest import normalize_image

ROOT = project_path(".")
SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

# Reuse the exact model/provider construction used by the production CLI.
from stage_05_review_export.run_analysis_pipeline import _build_provider  # type: ignore
from consentguard.stage_02_baseline_model.config import load_training_config
from consentguard.stage_03_specialists.barcode_zxing import ZXingBarcodeProvider
from consentguard.stage_03_specialists.face_yunet import YuNetFaceProvider
from consentguard.stage_03_specialists.plate_yunet import LPDYuNetPlateProvider
from consentguard.stage_03_specialists.ppocr_onnx import PPOCRTextGeometryProvider

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _polygon_mask(instance: dict[str, Any], *, width: int, height: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for polygon in instance.get("polygons", []):
        points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
        if len(points) >= 3:
            points[:, 0] = np.clip(points[:, 0], 0, width - 1)
            points[:, 1] = np.clip(points[:, 1], 0, height - 1)
            cv2.fillPoly(mask, [np.round(points).astype(np.int32)], 1)
    return mask.astype(bool)


def _bootstrap_interval(values: list[float], *, seed: int, draws: int = 1000) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "lower_95": None, "upper_95": None}
    rng = random.Random(seed)
    array = np.asarray(values, dtype=np.float64)
    means = np.asarray(
        [float(array[[rng.randrange(len(array)) for _ in range(len(array))]].mean()) for _ in range(draws)],
        dtype=np.float64,
    )
    return {
        "count": len(values),
        "mean": float(array.mean()),
        "lower_95": float(np.quantile(means, 0.025)),
        "upper_95": float(np.quantile(means, 0.975)),
    }


def _build_providers(args: argparse.Namespace, device: torch.device) -> tuple[object, ...]:
    base_config = load_training_config(args.config, require_validation_data=False)
    providers: list[object] = [
        _build_provider(base_config, project_path(args.checkpoint), device),
    ]
    for checkpoint, config_path, name in (
        (args.face_checkpoint, args.face_config, "face_maskrcnn"),
        (args.plate_checkpoint, args.plate_config, "plate_maskrcnn"),
        (args.handwriting_checkpoint, args.handwriting_config, "handwriting_maskrcnn"),
    ):
        if checkpoint:
            config = load_training_config(config_path, require_validation_data=False)
            providers.append(
                _build_provider(config, project_path(checkpoint), device, provider_name=name)
            )
    if args.yunet_model:
        path = project_path(args.yunet_model)
        providers.append(YuNetFaceProvider(path, version=f"{path.name}:{_sha256(path)[:16]}"))
    if args.plate_yunet_model:
        path = project_path(args.plate_yunet_model)
        providers.append(LPDYuNetPlateProvider(path, version=f"{path.name}:{_sha256(path)[:16]}"))
    if args.ppocr_model:
        path = project_path(args.ppocr_model)
        providers.append(PPOCRTextGeometryProvider(path, version=f"{path.name}:{_sha256(path)[:16]}"))
    if args.with_barcode:
        providers.append(ZXingBarcodeProvider())
    return tuple(providers)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    records_path = project_path(args.records)
    if "test" in records_path.name.lower() or "test2017" in str(records_path).lower():
        raise ValueError("The fused evaluator refuses test-split records")
    records = _read_jsonl(records_path)
    if args.max_images is not None:
        records = records[: args.max_images]
    if not records:
        raise ValueError("No validation records found")

    device = select_device(args.device)
    providers = _build_providers(args, device)
    orchestrator = AnalysisOrchestrator(tuple(providers))  # type: ignore[arg-type]
    thresholds = ThresholdRegistry.load(project_path(args.threshold_profile))
    fusion = EvidenceFusion(thresholds)
    class_map = json.loads(project_path(args.class_map).read_text(encoding="utf-8"))
    id_to_name = {int(value): str(key) for key, value in class_map.items()}

    per_class_instances: Counter[str] = Counter()
    per_class_found: Counter[str] = Counter()
    per_class_pixel_total: Counter[str] = Counter()
    per_class_pixel_covered: Counter[str] = Counter()
    image_pixel_recall: list[float] = []
    negative_flags: list[float] = []
    candidate_counts: list[int] = []
    provider_counts: Counter[str] = Counter()
    unavailable_counts: Counter[str] = Counter()
    provider_errors: Counter[str] = Counter()
    processed = 0

    for record in records:
        image_path = project_path(record["image_path"])
        image = normalize_image(image_path)
        analysis = orchestrator.analyze(image)
        snapshot_evidence = list(analysis.evidence)
        fused = fusion.combine(
            snapshot_evidence,
            width=image.width,
            height=image.height,
            unavailable_providers=analysis.unavailable_providers,
        )
        for item in snapshot_evidence:
            provider_counts[item.provider] += 1
        unavailable_counts.update(analysis.unavailable_providers)
        provider_errors.update(analysis.provider_errors)

        candidate_masks: list[tuple[np.ndarray, tuple[str, ...]]] = []
        union = np.zeros((image.height, image.width), dtype=bool)
        for candidate in fused.candidates:
            mask = decode_binary_mask(candidate.mask_rle, image.height, image.width).astype(bool)
            union |= mask
            candidate_masks.append((mask, candidate.privacy_classes))
        candidate_counts.append(len(fused.candidates))

        target_union = np.zeros_like(union)
        instances = record.get("instances", [])
        for instance in instances:
            class_name = id_to_name.get(int(instance["class_id"]), str(instance["class_id"]))
            target = _polygon_mask(instance, width=image.width, height=image.height)
            target_union |= target
            pixels = int(target.sum())
            covered = int((target & union).sum())
            per_class_instances[class_name] += 1
            per_class_pixel_total[class_name] += pixels
            per_class_pixel_covered[class_name] += covered
            class_union = np.zeros_like(union)
            for candidate_mask, classes in candidate_masks:
                if class_name in classes:
                    class_union |= candidate_mask
            if pixels and int((target & class_union).sum()) / pixels >= args.instance_coverage_threshold:
                per_class_found[class_name] += 1
        gt_pixels = int(target_union.sum())
        covered_pixels = int((target_union & union).sum())
        image_pixel_recall.append(covered_pixels / gt_pixels if gt_pixels else 1.0)
        negative_flags.append(1.0 if not instances and len(fused.candidates) else 0.0)
        processed += 1
        if processed == 1 or processed % 25 == 0 or processed == len(records):
            print(f"[fused] processed {processed}/{len(records)}", flush=True)

    per_class: dict[str, dict[str, Any]] = {}
    for class_name in sorted(per_class_instances):
        total = per_class_instances[class_name]
        pixels = per_class_pixel_total[class_name]
        covered = per_class_pixel_covered[class_name]
        per_class[class_name] = {
            "instances": total,
            "found_instances": per_class_found[class_name],
            "instance_recall": per_class_found[class_name] / total if total else 0.0,
            "sensitive_pixels": pixels,
            "covered_sensitive_pixels": covered,
            "pixel_recall": covered / pixels if pixels else 0.0,
            "pixel_leakage": 1.0 - covered / pixels if pixels else 0.0,
        }

    report = {
        "schema_version": "fused-validation-evaluation-v1",
        "split": "v2_validation_only",
        "test_split_used": False,
        "records_path": str(records_path),
        "records_sha256": _sha256(records_path),
        "evaluated_images": processed,
        "device": str(device),
        "threshold_profile": {
            "path": str(thresholds.profile.source_path),
            "profile_id": thresholds.profile.profile_id,
            "release_ready": thresholds.profile.release_ready,
            "sha256": thresholds.profile.source_sha256,
        },
        "providers": sorted(provider.name for provider in providers),
        "provider_evidence_counts": dict(sorted(provider_counts.items())),
        "unavailable_provider_counts": dict(sorted(unavailable_counts.items())),
        "provider_error_counts": dict(sorted(provider_errors.items())),
        "overall": {
            "pixel_recall": float(np.mean(image_pixel_recall)),
            "pixel_recall_bootstrap_95": _bootstrap_interval(image_pixel_recall, seed=args.seed),
            "negative_image_false_positive_rate": float(np.mean(negative_flags)),
            "negative_fpr_bootstrap_95": _bootstrap_interval(negative_flags, seed=args.seed + 1),
            "mean_candidates_per_image": float(np.mean(candidate_counts)),
            "images_with_candidates": int(sum(count > 0 for count in candidate_counts)),
        },
        "per_class": per_class,
        "notes": [
            "This is validation evidence, not a locked-test result.",
            "A Visual Redactions V2 record does not carry the general/India domain labels required by the release gates.",
            "Missing providers and provider errors are reported explicitly and never treated as empty scenes.",
        ],
    }
    output = project_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--records", default="data/processed/visual_redactions_verified_visual_v2_negatives/records_val2017.jsonl")
    parser.add_argument("--class-map", default="data/processed/visual_redactions_verified_visual_v2_negatives/class_map.json")
    parser.add_argument("--threshold-profile", required=True)
    parser.add_argument("--face-checkpoint", type=Path)
    parser.add_argument("--face-config", default="main_project/configs/stage_03_specialists/train_face_maskrcnn_5ep.yaml")
    parser.add_argument("--plate-checkpoint", type=Path)
    parser.add_argument("--plate-config", default="main_project/configs/stage_03_specialists/train_plate_maskrcnn_5ep.yaml")
    parser.add_argument("--handwriting-checkpoint", type=Path)
    parser.add_argument("--handwriting-config", default="main_project/configs/stage_03_specialists/train_handwriting_maskrcnn_5ep.yaml")
    parser.add_argument("--yunet-model", type=Path)
    parser.add_argument("--plate-yunet-model", type=Path)
    parser.add_argument("--ppocr-model", type=Path)
    parser.add_argument("--with-barcode", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--instance-coverage-threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.max_images is not None and args.max_images < 1:
        parser.error("--max-images must be positive")
    if not 0.0 <= args.instance_coverage_threshold <= 1.0:
        parser.error("--instance-coverage-threshold must be in [0, 1]")
    report = evaluate(args)
    print(json.dumps({"output": str(project_path(args.output)), "overall": report["overall"]}, indent=2))


if __name__ == "__main__":
    main()
