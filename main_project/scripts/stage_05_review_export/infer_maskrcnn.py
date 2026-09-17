"""Create a newly encoded solid-redacted image from a trained checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from consentguard.stage_02_baseline_model.config import (
    load_training_config,
    project_path,
    validate_checkpoint_inference_compatibility,
)
from consentguard.stage_02_baseline_model.models import build_instance_segmentation_model
from consentguard.stage_05_review_export.redaction.prediction_renderer import (
    load_rgb_image,
    predict_union_mask,
    write_inference_report,
    write_metadata_free_redaction,
)
from consentguard.shared.runtime import select_device
from consentguard.stage_04_fusion_calibration.evidence import ThresholdRegistry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--threshold-profile",
        default="main_project/configs/stage_04_fusion_calibration/threshold_profile_candidate_v1.yaml",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=None,
        help="Manual global override; recorded in the audit report.",
    )
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--dilate-pixels", type=int, default=3)
    args = parser.parse_args()

    config = load_training_config(args.config, require_validation_data=False)
    checkpoint_path = project_path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    validate_checkpoint_inference_compatibility(checkpoint, config)
    data = config.section("data")
    model = build_instance_segmentation_model(
        config.section("model"),
        num_classes=config.num_classes,
        min_size=int(data["short_side"]),
        max_size=int(data["max_long_side"]),
    )
    model.load_state_dict(checkpoint["model_state"])
    device = select_device(args.device)
    model.to(device)
    source_path = project_path(args.input)
    output_path = project_path(args.output)
    image = load_rgb_image(source_path)
    registry = ThresholdRegistry.load(project_path(args.threshold_profile))
    class_rules = {
        class_id: registry.get("maskrcnn", class_name)
        for class_name, class_id in config.class_map.items()
        if class_id != 0
    }
    default_score_threshold = 0.5 if args.score_threshold is None else args.score_threshold
    score_thresholds = {
        class_id: default_score_threshold if args.score_threshold is not None else rule.score_threshold
        for class_id, rule in class_rules.items()
    }
    mask, detections = predict_union_mask(
        model,
        image,
        device,
        short_side=int(data["short_side"]),
        max_long_side=int(data["max_long_side"]),
        score_threshold=default_score_threshold,
        score_thresholds=score_thresholds,
        min_area_pixels_by_class={
            class_id: rule.min_area_pixels for class_id, rule in class_rules.items()
        },
        dilation_pixels_by_class={
            class_id: rule.dilation_pixels for class_id, rule in class_rules.items()
        },
        mask_threshold=args.mask_threshold,
        dilation_pixels=args.dilate_pixels,
    )
    export = write_metadata_free_redaction(source_path, output_path, image, mask)
    id_to_name = {value: key for key, value in config.class_map.items()}
    for detection in detections:
        detection["class_name"] = id_to_name.get(detection["class_id"], "unknown")
    report = {
        "checkpoint": str(checkpoint_path),
        "output": str(output_path),
        "score_threshold": default_score_threshold,
        "manual_global_threshold_override": args.score_threshold is not None,
        "threshold_profile": {
            "profile_id": registry.profile.profile_id,
            "release_ready": registry.profile.release_ready,
            "path": str(registry.profile.source_path),
            "sha256": registry.profile.source_sha256,
        },
        "release_status": "REVIEW_REQUIRED" if not registry.profile.release_ready else "PROFILE_RELEASE_READY",
        "mask_threshold": args.mask_threshold,
        "dilation_pixels": args.dilate_pixels,
        "detection_count": len(detections),
        "detections": detections,
        "export": export,
    }
    report_path = output_path.with_suffix(output_path.suffix + ".json")
    write_inference_report(report, report_path)
    print(json.dumps({"report": str(report_path), **report}, indent=2))


if __name__ == "__main__":
    main()
