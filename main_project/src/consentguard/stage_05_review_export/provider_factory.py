"""Load trained TorchVision checkpoints into stable evidence providers."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import torch

from consentguard.stage_02_baseline_model.config import (
    TrainingConfig,
    validate_checkpoint_inference_compatibility,
)
from consentguard.stage_02_baseline_model.models import build_instance_segmentation_model
from consentguard.stage_03_specialists.box_detector import BoxDetectorEvidenceProvider
from consentguard.stage_03_specialists.maskrcnn import MaskRCNNEvidenceProvider


def load_torchvision_provider(
    config: TrainingConfig,
    checkpoint_path: str | Path,
    device: torch.device,
    *,
    provider_name: str = "maskrcnn",
) -> object:
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    validate_checkpoint_inference_compatibility(checkpoint, config)
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
    version = f"{checkpoint_path.name}:{hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()[:16]}"
    if str(config.section("model")["name"]).startswith("fasterrcnn"):
        overrides = {}
        if provider_name.startswith("face"):
            overrides = {1: "face"}
        elif provider_name.startswith("plate"):
            overrides = {1: "license_plate"}
        return BoxDetectorEvidenceProvider(
            model,
            device,
            class_map=config.class_map,
            version=version,
            provider_name=provider_name,
            privacy_class_overrides=overrides,
            short_side=int(data["short_side"]),
            max_long_side=int(data["max_long_side"]),
        )
    return MaskRCNNEvidenceProvider(
        model,
        device,
        class_map=config.class_map,
        version=version,
        provider_name=provider_name,
        short_side=int(data["short_side"]),
        max_long_side=int(data["max_long_side"]),
    )
