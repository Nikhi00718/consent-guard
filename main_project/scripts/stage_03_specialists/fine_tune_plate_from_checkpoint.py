"""Fine-tune a one-class plate detector from an existing TorchVision checkpoint.

This intentionally initializes only ``model_state``.  Optimizer, scheduler,
epoch, and best-metric state are reset so a domain-adaptation run does not
silently resume the source-domain experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

SRC_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from consentguard.shared.paths import project_path
from consentguard.shared.runtime import atomic_json_dump, environment_snapshot, seed_everything, select_device
from consentguard.stage_02_baseline_model.config import (
    load_training_config,
    validate_checkpoint_initialization_compatibility,
)
from consentguard.stage_02_baseline_model.data_loading import build_data_loaders
from consentguard.stage_02_baseline_model.models import build_instance_segmentation_model, model_summary
from consentguard.stage_02_baseline_model.training_loop import MaskRCNNTrainer


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="main_project/configs/stage_03_specialists/train_plate_ccpd2020_india_finetune_5ep.yaml")
    parser.add_argument("--init-checkpoint", default="artifacts/checkpoints/specialist_plate_ccpd2020_fasterrcnn/best.pt")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.epochs is not None and args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.max_steps is not None and args.max_steps < 1:
        parser.error("--max-steps must be positive")

    config = load_training_config(args.config)
    if args.device is not None:
        config.values["training"]["device"] = args.device
    if args.epochs is not None:
        config.values["training"]["epochs"] = args.epochs
    if args.max_steps is not None:
        config.values["training"]["max_steps"] = args.max_steps

    checkpoint_path = project_path(args.init_checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Initialization checkpoint does not exist: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    validate_checkpoint_initialization_compatibility(checkpoint, config)

    seed_everything(
        int(config.section("experiment")["seed"]),
        deterministic=bool(config.section("experiment").get("deterministic", False)),
        cudnn_benchmark=bool(config.section("experiment").get("cudnn_benchmark", False)),
    )
    device = select_device(config.section("training")["device"])
    train_loader, val_loader = build_data_loaders(config)
    # The initialization checkpoint contains the complete detector state. Do
    # not ask TorchVision for COCO weights here: Kaggle runs offline and a
    # download would be redundant. ``pretrained=False`` with all backbone
    # layers trainable preserves the same parameter/key layout; the next
    # ``load_state_dict`` replaces every parameter with the source checkpoint.
    model_config = dict(config.section("model"))
    model_config["pretrained"] = False
    model_config["trainable_backbone_layers"] = 5
    model = build_instance_segmentation_model(
        model_config,
        num_classes=config.num_classes,
        min_size=int(config.section("data")["short_side"]),
        max_size=int(config.section("data")["max_long_side"]),
    )
    missing, unexpected = model.load_state_dict(checkpoint["model_state"], strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint model state mismatch: missing={missing}, unexpected={unexpected}")

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json_dump(
        {
            "initialization_checkpoint": str(checkpoint_path),
            "initialization_checkpoint_sha256": _sha256(checkpoint_path),
            "initialization_checkpoint_bytes": checkpoint_path.stat().st_size,
            "source_checkpoint_epoch": checkpoint.get("epoch"),
            "source_checkpoint_best_map": checkpoint.get("best_map"),
            "optimizer_state_reset": True,
            "scheduler_state_reset": True,
            "epoch_counter_reset": True,
        },
        output_dir / "initialization.json",
    )
    preflight = {
        "config": str(config.path),
        "device": str(device),
        "environment": environment_snapshot(),
        "train_images": len(train_loader.dataset),
        "validation_images": len(val_loader.dataset) if val_loader is not None else 0,
        "num_classes_including_background": config.num_classes,
        "model": model_summary(model),
    }
    print(json.dumps(preflight, indent=2), flush=True)
    if args.preflight_only:
        return
    trainer = MaskRCNNTrainer(model, config, train_loader, val_loader, device)
    print(json.dumps(trainer.train(), indent=2), flush=True)


if __name__ == "__main__":
    main()
