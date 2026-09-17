"""Evaluate RPN, RoI-independent privacy coverage, and redaction leakage diagnostics."""

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
from consentguard.stage_02_baseline_model.diagnostics import (
    evaluate_privacy_coverage,
    evaluate_rpn_proposal_recall,
)
from consentguard.shared.runtime import atomic_json_dump, select_device
from consentguard.stage_02_baseline_model.models import build_instance_segmentation_model
from consentguard.stage_02_baseline_model.data_loading import build_data_loaders


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path, default=Path("reports/maskrcnn_diagnostics.json"))
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--dilate-pixels", type=int, default=0)
    parser.add_argument("--instance-coverage-threshold", type=float, default=0.8)
    args = parser.parse_args()
    if args.max_batches is not None and args.max_batches < 1:
        parser.error("--max-batches must be positive")

    config = load_training_config(args.config)
    checkpoint_path = project_path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    validate_checkpoint_inference_compatibility(checkpoint, config)
    _, val_loader = build_data_loaders(config)
    if val_loader is None:
        raise RuntimeError("Evaluation is disabled in this configuration")

    device = select_device(args.device)
    data = config.section("data")
    model = build_instance_segmentation_model(
        config.section("model"),
        num_classes=config.num_classes,
        min_size=int(data["short_side"]),
        max_size=int(data["max_long_side"]),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    diagnostics = {
        "checkpoint": str(checkpoint_path),
        "config": str(config.path),
        "device": str(device),
        "rpn": evaluate_rpn_proposal_recall(
            model,
            val_loader,
            device,
            max_batches=args.max_batches,
            class_map=config.class_map,
        ),
        "privacy": evaluate_privacy_coverage(
            model,
            val_loader,
            device,
            score_threshold=args.score_threshold,
            mask_threshold=args.mask_threshold,
            dilation_pixels=args.dilate_pixels,
            instance_coverage_threshold=args.instance_coverage_threshold,
            max_batches=args.max_batches,
            class_map=config.class_map,
        ),
    }
    output = project_path(args.output)
    atomic_json_dump(diagnostics, output)
    print(json.dumps({"output": str(output), "diagnostics": diagnostics}, indent=2))


if __name__ == "__main__":
    main()
