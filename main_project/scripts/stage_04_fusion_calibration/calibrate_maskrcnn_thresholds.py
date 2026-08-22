"""Calibrate Mask R-CNN score thresholds on one locked validation split only."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from consentguard.shared.paths import project_path
from consentguard.shared.runtime import atomic_json_dump, select_device
from consentguard.stage_02_baseline_model.config import load_training_config, validate_checkpoint_inference_compatibility
from consentguard.stage_02_baseline_model.data_loading import build_data_loaders
from consentguard.stage_02_baseline_model.models import build_instance_segmentation_model
from consentguard.stage_04_fusion_calibration.calibration import MatchCounts, match_class_masks


THRESHOLDS = tuple(float(round(float(value), 2)) for value in np.arange(0.05, 1.0, 0.05))


def _resize_masks(masks: torch.Tensor, height: int, width: int) -> list[np.ndarray]:
    if masks.numel() == 0:
        return []
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim != 3:
        raise ValueError(f"expected prediction masks [N,H,W] or [N,1,H,W], received {tuple(masks.shape)}")
    if tuple(masks.shape[-2:]) != (height, width):
        masks = F.interpolate(masks[:, None].float(), size=(height, width), mode="bilinear", align_corners=False)[:, 0]
    return [(mask.detach().cpu().numpy() >= 0.5) for mask in masks]


def _merge_counts(left: MatchCounts, right: MatchCounts) -> MatchCounts:
    return MatchCounts(
        left.true_positive + right.true_positive,
        left.false_positive + right.false_positive,
        left.false_negative + right.false_negative,
    )


def _metrics(counts: MatchCounts) -> dict[str, float | int]:
    return {
        "true_positive": counts.true_positive,
        "false_positive": counts.false_positive,
        "false_negative": counts.false_negative,
        "precision": counts.precision,
        "recall": counts.recall,
        "f2": counts.f2,
    }


def calibrate(
    model: torch.nn.Module,
    data_loader,
    device: torch.device,
    *,
    class_map: dict[str, int],
    target_recall: float,
    precision_floor: float,
    iou_threshold: float,
    max_batches: int | None = None,
) -> dict[str, Any]:
    if not 0.0 < target_recall <= 1.0 or not 0.0 <= precision_floor <= 1.0:
        raise ValueError("target_recall must be in (0,1] and precision_floor in [0,1]")
    counts: dict[int, dict[float, MatchCounts]] = {
        int(class_id): {threshold: MatchCounts(0, 0, 0) for threshold in THRESHOLDS}
        for class_id in class_map.values()
        if int(class_id) != 0
    }
    images_seen = 0
    batches_seen = 0
    model.eval()
    with torch.inference_mode():
        for images, targets in data_loader:
            if max_batches is not None and batches_seen >= max_batches:
                break
            predictions = model([image.to(device, non_blocking=True) for image in images])
            if len(predictions) != len(targets):
                raise RuntimeError("model prediction count does not match targets")
            for prediction, target in zip(predictions, targets):
                gt_masks = target["masks"].detach().cpu().numpy().astype(bool, copy=False)
                gt_labels = target["labels"].detach().cpu().numpy().astype(int, copy=False)
                height, width = gt_masks.shape[-2:]
                pred_masks = _resize_masks(prediction["masks"].detach(), height, width)
                pred_scores = [float(value) for value in prediction["scores"].detach().cpu().tolist()]
                pred_labels = [int(value) for value in prediction["labels"].detach().cpu().tolist()]
                for class_id in counts:
                    predicted_for_class = [
                        mask for mask, label in zip(pred_masks, pred_labels) if label == class_id
                    ]
                    scores_for_class = [
                        score for score, label in zip(pred_scores, pred_labels) if label == class_id
                    ]
                    gt_for_class = [mask for mask, label in zip(gt_masks, gt_labels) if label == class_id]
                    for threshold in THRESHOLDS:
                        counts[class_id][threshold] = _merge_counts(
                            counts[class_id][threshold],
                            match_class_masks(
                                predicted_for_class,
                                scores_for_class,
                                gt_for_class,
                                score_threshold=threshold,
                                iou_threshold=iou_threshold,
                            ),
                        )
                images_seen += 1
            batches_seen += 1
    if not images_seen:
        raise RuntimeError("validation loader produced no images")

    id_to_name = {int(value): str(name) for name, value in class_map.items()}
    per_class: dict[str, Any] = {}
    selected_rules: dict[str, float] = {}
    release_ready = True
    for class_id, threshold_counts in sorted(counts.items()):
        metrics = {str(threshold): _metrics(value) for threshold, value in threshold_counts.items()}
        eligible = [
            threshold
            for threshold, value in threshold_counts.items()
            if value.recall >= target_recall and value.precision >= precision_floor
        ]
        if eligible:
            selected = max(eligible)
            selection_reason = "highest_threshold_meeting_recall_and_precision_floors"
        else:
            # When scores are uninformative, choose the lower threshold on an
            # F2 tie so the privacy-oriented fallback does not hide evidence.
            selected = max(THRESHOLDS, key=lambda threshold: (threshold_counts[threshold].f2, -threshold))
            selection_reason = "max_f2_fallback_gates_not_met"
            release_ready = False
        selected_rules[id_to_name[class_id]] = selected
        selected_metrics = threshold_counts[selected]
        if selected_metrics.recall < target_recall or selected_metrics.precision < precision_floor:
            release_ready = False
        per_class[id_to_name[class_id]] = {
            "class_id": class_id,
            "selected_threshold": selected,
            "selection_reason": selection_reason,
            "selected_metrics": _metrics(selected_metrics),
            "thresholds": metrics,
        }
    return {
        "images": images_seen,
        "batches": batches_seen,
        "iou_threshold": iou_threshold,
        "target_recall": target_recall,
        "precision_floor": precision_floor,
        "threshold_grid": list(THRESHOLDS),
        "release_ready": release_ready,
        "per_class": per_class,
        "selected_thresholds": selected_rules,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/maskrcnn_v2_threshold_calibration.json"))
    parser.add_argument("--profile-output", type=Path, default=Path("main_project/configs/stage_04_fusion_calibration/threshold_profile_v2_validation.yaml"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--target-recall", type=float, default=0.80)
    parser.add_argument("--precision-floor", type=float, default=0.05)
    parser.add_argument("--iou-threshold", type=float, default=0.50)
    parser.add_argument("--max-batches", type=int, default=None)
    args = parser.parse_args()
    config = load_training_config(args.config)
    checkpoint_path = project_path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    validate_checkpoint_inference_compatibility(checkpoint, config)
    _, val_loader = build_data_loaders(config)
    if val_loader is None:
        raise RuntimeError("validation is disabled; calibration requires V2 validation records")
    device = select_device(args.device)
    model_config = copy.deepcopy(config.section("model"))
    model_config["pretrained"] = False
    model_config["trainable_backbone_layers"] = 5
    data = config.section("data")
    model = build_instance_segmentation_model(
        model_config,
        num_classes=config.num_classes,
        min_size=int(data["short_side"]),
        max_size=int(data["max_long_side"]),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    result = calibrate(
        model,
        val_loader,
        device,
        class_map=config.class_map,
        target_recall=args.target_recall,
        precision_floor=args.precision_floor,
        iou_threshold=args.iou_threshold,
        max_batches=args.max_batches,
    )
    records_path = project_path(data["val_records"])
    report = {
        "schema_version": "threshold-calibration-v1",
        "split": "v2_validation_only",
        "test_split_used": False,
        "config": str(config.path),
        "config_sha256": hashlib.sha256(config.path.read_bytes()).hexdigest(),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        "validation_records": str(records_path),
        "validation_records_sha256": hashlib.sha256(records_path.read_bytes()).hexdigest(),
        "class_map": config.class_map,
        "calibration": result,
    }
    output = project_path(args.output)
    atomic_json_dump(report, output)

    import yaml

    candidate_profile_path = project_path("main_project/configs/stage_04_fusion_calibration/threshold_profile_candidate_v1.yaml")
    existing_profile = yaml.safe_load(candidate_profile_path.read_text(encoding="utf-8"))
    existing_rules = {
        (str(rule.get("provider")), str(rule.get("privacy_class"))): rule
        for rule in existing_profile.get("rules", [])
    }
    rules: list[dict[str, Any]] = []
    for privacy_class, threshold in result["selected_thresholds"].items():
        base = dict(existing_rules.get(("maskrcnn", privacy_class), {}))
        base.update({"provider": "maskrcnn", "privacy_class": privacy_class, "score_threshold": threshold})
        base.setdefault("min_area_pixels", 1)
        base.setdefault("mandatory_review", True)
        rules.append(base)
    # Preserve optional-provider and wildcard rules from the candidate profile;
    # this calibration run only changes Mask R-CNN score thresholds.
    rules.extend(
        dict(rule)
        for key, rule in existing_rules.items()
        if key[0] != "maskrcnn"
    )
    profile = {
        "profile_id": "threshold-profile-v2-validation-calibrated",
        "release_ready": bool(result["release_ready"]),
        "calibration_report": str(output),
        "calibration_report_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "rules": rules,
    }
    profile_path = project_path(args.profile_output)
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text(yaml.safe_dump(profile, sort_keys=False), encoding="utf-8")
    print(json.dumps({"report": str(output), "profile": str(profile_path), "release_ready": result["release_ready"]}, indent=2))


if __name__ == "__main__":
    main()
