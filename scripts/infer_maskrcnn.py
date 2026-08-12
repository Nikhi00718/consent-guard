"""Create a newly encoded solid-redacted image from a trained checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from consentguard.config import (
    load_training_config,
    project_path,
    validate_checkpoint_inference_compatibility,
)
from consentguard.perception.models import build_instance_segmentation_model
from consentguard.redaction.prediction_renderer import (
    load_rgb_image,
    predict_union_mask,
    write_inference_report,
    write_metadata_free_redaction,
)
from consentguard.runtime import select_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--score-threshold", type=float, default=0.5)
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
    mask, detections = predict_union_mask(
        model,
        image,
        device,
        short_side=int(data["short_side"]),
        max_long_side=int(data["max_long_side"]),
        score_threshold=args.score_threshold,
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
        "score_threshold": args.score_threshold,
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
