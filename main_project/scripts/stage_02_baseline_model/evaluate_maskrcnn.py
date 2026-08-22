"""Evaluate a trained Mask R-CNN checkpoint on deterministic validation data."""

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
from consentguard.shared.runtime import atomic_json_dump, select_device
from consentguard.stage_02_baseline_model.metrics import evaluate_instance_segmentation
from consentguard.stage_02_baseline_model.models import build_instance_segmentation_model
from consentguard.stage_02_baseline_model.data_loading import build_data_loaders


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path, default=Path("reports/maskrcnn_evaluation.json"))
    parser.add_argument("--max-batches", type=int, default=None)
    args = parser.parse_args()
    if args.max_batches is not None and args.max_batches < 1:
        parser.error("--max-batches must be positive")

    config = load_training_config(args.config)
    checkpoint = torch.load(project_path(args.checkpoint), map_location="cpu", weights_only=True)
    validate_checkpoint_inference_compatibility(checkpoint, config)
    _, val_loader = build_data_loaders(config)
    if val_loader is None:
        raise RuntimeError("Evaluation is disabled in this configuration")
    device = select_device(args.device)
    model = build_instance_segmentation_model(
        config.section("model"),
        num_classes=config.num_classes,
        min_size=int(config.section("data")["short_side"]),
        max_size=int(config.section("data")["max_long_side"]),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    evaluation = config.section("evaluation")
    metrics = evaluate_instance_segmentation(
        model,
        val_loader,
        device,
        score_threshold=float(evaluation["score_threshold"]),
        class_metrics=bool(evaluation["class_metrics"]),
        max_batches=args.max_batches if args.max_batches is not None else evaluation["max_batches"],
        class_map=config.class_map,
    )
    output = project_path(args.output)
    atomic_json_dump(metrics, output)
    print(json.dumps({"output": str(output), "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
