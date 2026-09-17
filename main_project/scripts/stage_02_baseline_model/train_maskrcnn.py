"""Train the ConsentGuard Visual Redactions Mask R-CNN localizer."""

from __future__ import annotations

import argparse
import json

from consentguard.shared.runtime import environment_snapshot, seed_everything, select_device
from consentguard.stage_02_baseline_model.config import load_training_config
from consentguard.stage_02_baseline_model.models import build_instance_segmentation_model, model_summary
from consentguard.stage_02_baseline_model.data_loading import build_data_loaders
from consentguard.stage_02_baseline_model.training_loop import MaskRCNNTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="main_project/configs/stage_02_baseline_model/train_maskrcnn_4gb.yaml")
    parser.add_argument("--resume", default=None, help="Checkpoint produced by this trainer")
    parser.add_argument("--max-steps", type=int, default=None, help="Override optimizer-step limit")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch limit (useful for resume smoke tests)")
    parser.add_argument("--output-dir", default=None, help="Override experiment output directory")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None)
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    if args.max_steps is not None and args.max_steps < 1:
        parser.error("--max-steps must be positive")
    config = load_training_config(args.config, require_validation_data=not args.skip_evaluation)
    if args.max_steps is not None:
        config.values["training"]["max_steps"] = args.max_steps
    if args.epochs is not None:
        if args.epochs < 1:
            parser.error("--epochs must be positive")
        config.values["training"]["epochs"] = args.epochs
    if args.device is not None:
        config.values["training"]["device"] = args.device
    if args.output_dir is not None:
        config.values["experiment"]["output_dir"] = args.output_dir
    if args.skip_evaluation:
        config.values["evaluation"]["enabled"] = False

    seed_everything(
        int(config.section("experiment")["seed"]),
        deterministic=bool(config.section("experiment").get("deterministic", False)),
        cudnn_benchmark=bool(config.section("experiment").get("cudnn_benchmark", False)),
    )
    device = select_device(config.section("training")["device"])
    train_loader, val_loader = build_data_loaders(config)
    model = build_instance_segmentation_model(
        config.section("model"),
        num_classes=config.num_classes,
        min_size=int(config.section("data")["short_side"]),
        max_size=int(config.section("data")["max_long_side"]),
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
    print(json.dumps(preflight, indent=2))
    if args.preflight_only:
        return

    trainer = MaskRCNNTrainer(model, config, train_loader, val_loader, device)
    resume_path = args.resume or config.section("training").get("resume_from")
    if resume_path:
        trainer.resume(resume_path)
    print(json.dumps(trainer.train(), indent=2))


if __name__ == "__main__":
    main()
