"""Validated YAML configuration for localizer experiments."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


from consentguard.shared.paths import PROJECT_ROOT, project_path


class ConfigError(ValueError):
    """Raised when an experiment configuration is unsafe or incomplete."""


DEFAULTS: dict[str, Any] = {
    "experiment": {
        "name": "maskrcnn_experiment",
        "seed": 1337,
        "output_dir": "artifacts/checkpoints/default",
        "deterministic": False,
        "cudnn_benchmark": False,
    },
    "data": {
        "short_side": 512,
        "max_long_side": 768,
        "crop_size": None,
        "crop_probability": 0.0,
        "crop_context_factor": 4.0,
        "min_crop_visibility": 0.25,
        "horizontal_flip_probability": 0.0,
        "brightness_contrast_probability": 0.0,
        "num_workers": 0,
        "max_train_samples": None,
        "max_val_samples": None,
        "sampling": "uniform",
        "class_balance_power": 0.5,
        "max_class_balance_ratio": 10.0,
        "negative_sampling_weight": 0.25,
    },
    "model": {
        "name": "maskrcnn_resnet50_fpn_v2",
        "pretrained": True,
        "trainable_backbone_layers": 3,
        "small_object_anchors": False,
        "class_agnostic_mask": False,
        "detections_per_image": 100,
    },
    "optimizer": {
        "name": "sgd",
        "learning_rate": 0.0025,
        "momentum": 0.9,
        "weight_decay": 0.0005,
        "gradient_clip_norm": 10.0,
        "warmup_steps": 500,
        "warmup_start_factor": 0.001,
    },
    "training": {
        "device": "auto",
        "epochs": 20,
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
        "amp": True,
        "log_every_steps": 20,
        "save_every_epochs": 1,
        "evaluate_every_epochs": 1,
        "max_steps": None,
        "resume_from": None,
    },
    "scheduler": {"name": "step", "step_size": 6, "gamma": 0.1},
    "evaluation": {"enabled": True, "score_threshold": 0.0, "class_metrics": True, "max_batches": None},
}


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


@dataclass(frozen=True)
class TrainingConfig:
    path: Path
    values: dict[str, Any]
    class_map: dict[str, int]

    def section(self, name: str) -> dict[str, Any]:
        return self.values[name]

    @property
    def num_classes(self) -> int:
        return max(self.class_map.values()) + 1

    @property
    def output_dir(self) -> Path:
        return project_path(self.values["experiment"]["output_dir"])

    def as_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self.values)


def validate_checkpoint_inference_compatibility(
    checkpoint: dict[str, Any],
    config: TrainingConfig,
) -> None:
    """Reject a checkpoint/config pairing that could silently change inference."""

    if checkpoint.get("class_map") != config.class_map:
        raise ConfigError("Checkpoint class_map does not match the selected configuration")
    saved = checkpoint.get("config")
    if not isinstance(saved, dict):
        raise ConfigError("Checkpoint does not contain its resolved training configuration")
    saved_model = copy.deepcopy(saved.get("model", {}))
    active_model = copy.deepcopy(config.values.get("model", {}))
    # This flag was added after the verified control checkpoints were created.
    # An omitted value in an older checkpoint is semantically equivalent to the
    # current default of False and must not invalidate control evaluation.
    saved_model.setdefault("class_agnostic_mask", False)
    active_model.setdefault("class_agnostic_mask", False)
    if saved_model != active_model:
        raise ConfigError("Checkpoint model configuration does not match the selected configuration")
    saved_data = saved.get("data", {})
    active_data = config.section("data")
    for field in ("short_side", "max_long_side"):
        if saved_data.get(field) != active_data.get(field):
            raise ConfigError(
                f"Checkpoint data.{field} does not match the selected configuration"
            )


def load_training_config(path: str | Path, *, require_validation_data: bool = True) -> TrainingConfig:
    config_path = project_path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Training configuration does not exist: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigError(f"Configuration must contain a YAML mapping: {config_path}")
    unknown_sections = set(raw) - set(DEFAULTS)
    if unknown_sections:
        raise ConfigError(f"Unknown top-level configuration sections: {sorted(unknown_sections)}")
    values = _merge(DEFAULTS, raw)

    for field in ("train_records", "val_records", "class_map"):
        if field not in values["data"]:
            raise ConfigError(f"data.{field} is required")
    train_path = project_path(values["data"]["train_records"])
    if not train_path.is_file():
        raise FileNotFoundError(f"Training records are missing: {train_path}")
    if train_path.stat().st_size == 0:
        raise ConfigError(f"Training records file is empty: {train_path}")
    class_map_path = project_path(values["data"]["class_map"])
    if not class_map_path.is_file():
        raise FileNotFoundError(f"Class map is missing: {class_map_path}")
    class_map = json.loads(class_map_path.read_text(encoding="utf-8"))
    if not isinstance(class_map, dict) or not class_map:
        raise ConfigError("class_map must contain a non-empty JSON object")
    if class_map.get("background") != 0:
        raise ConfigError("class_map must map 'background' to 0")
    try:
        labels = sorted(int(value) for value in class_map.values())
    except (TypeError, ValueError) as error:
        raise ConfigError("class_map values must be integer class IDs") from error
    if labels != list(range(max(labels) + 1)):
        raise ConfigError("class_map labels must be contiguous integers beginning at 0")

    evaluation_enabled = bool(values["evaluation"]["enabled"])
    val_path = project_path(values["data"]["val_records"])
    if require_validation_data and evaluation_enabled and not val_path.is_file():
        raise FileNotFoundError(
            f"Validation records are missing: {val_path}. Finish/extract VISPR validation and rerun preprocessing, "
            "or use main_project/configs/stage_02_baseline_model/train_smoke.yaml."
        )
    if require_validation_data and evaluation_enabled and val_path.stat().st_size == 0:
        raise ConfigError(f"Validation records file is empty: {val_path}")

    if values["model"]["name"] not in {"maskrcnn_resnet50_fpn_v2", "maskrcnn_resnet50_fpn"}:
        raise ConfigError(f"Unsupported model.name: {values['model']['name']}")
    for section, field in (
        ("training", "epochs"),
        ("training", "batch_size"),
        ("training", "gradient_accumulation_steps"),
        ("data", "num_workers"),
    ):
        value = int(values[section][field])
        minimum = 0 if field == "num_workers" else 1
        if value < minimum:
            raise ConfigError(f"{section}.{field} must be >= {minimum}")
    if values["optimizer"]["learning_rate"] <= 0:
        raise ConfigError("optimizer.learning_rate must be positive")
    if values["optimizer"]["name"] not in {"sgd", "adamw"}:
        raise ConfigError("optimizer.name must be sgd or adamw")
    if int(values["optimizer"]["warmup_steps"]) < 0:
        raise ConfigError("optimizer.warmup_steps must be non-negative")
    if not 0.0 < float(values["optimizer"]["warmup_start_factor"]) <= 1.0:
        raise ConfigError("optimizer.warmup_start_factor must be in (0, 1]")
    if values["training"]["device"] not in {"auto", "cpu", "cuda"}:
        raise ConfigError("training.device must be auto, cpu, or cuda")
    if bool(values["experiment"].get("deterministic", False)) and bool(
        values["experiment"].get("cudnn_benchmark", False)
    ):
        raise ConfigError("experiment.deterministic and cudnn_benchmark cannot both be true")
    for field in ("log_every_steps", "save_every_epochs", "evaluate_every_epochs"):
        if int(values["training"][field]) < 1:
            raise ConfigError(f"training.{field} must be positive")
    if values["training"]["max_steps"] is not None and int(values["training"]["max_steps"]) < 1:
        raise ConfigError("training.max_steps must be positive when provided")

    for field in ("short_side", "max_long_side"):
        if int(values["data"][field]) < 32:
            raise ConfigError(f"data.{field} must be at least 32")
    if int(values["data"]["max_long_side"]) < int(values["data"]["short_side"]):
        raise ConfigError("data.max_long_side must be >= data.short_side")
    crop_size = values["data"]["crop_size"]
    if crop_size is not None and int(crop_size) < 32:
        raise ConfigError("data.crop_size must be at least 32 when provided")
    if float(values["data"]["crop_probability"]) > 0 and crop_size is None:
        raise ConfigError("data.crop_size is required when crop_probability is positive")
    if float(values["data"]["crop_context_factor"]) <= 0:
        raise ConfigError("data.crop_context_factor must be positive")
    for field in ("max_train_samples", "max_val_samples"):
        if values["data"][field] is not None and int(values["data"][field]) < 1:
            raise ConfigError(f"data.{field} must be positive when provided")
    for name in ("crop_probability", "min_crop_visibility", "horizontal_flip_probability", "brightness_contrast_probability"):
        value = float(values["data"][name])
        if not 0.0 <= value <= 1.0:
            raise ConfigError(f"data.{name} must be in [0, 1]")
    if values["data"]["sampling"] not in {"uniform", "class_balanced"}:
        raise ConfigError("data.sampling must be uniform or class_balanced")
    if not 0.0 <= float(values["data"]["class_balance_power"]) <= 1.0:
        raise ConfigError("data.class_balance_power must be in [0, 1]")
    if float(values["data"]["max_class_balance_ratio"]) < 1.0:
        raise ConfigError("data.max_class_balance_ratio must be >= 1")
    if not 0.0 < float(values["data"]["negative_sampling_weight"]) <= 1.0:
        raise ConfigError("data.negative_sampling_weight must be in (0, 1]")

    trainable_layers = int(values["model"]["trainable_backbone_layers"])
    if not 0 <= trainable_layers <= 5:
        raise ConfigError("model.trainable_backbone_layers must be in [0, 5]")
    if not bool(values["model"]["pretrained"]) and trainable_layers != 5:
        raise ConfigError(
            "model.trainable_backbone_layers must be 5 when model.pretrained is false"
        )
    if int(values["model"]["detections_per_image"]) < 1:
        raise ConfigError("model.detections_per_image must be positive")
    if not 0.0 <= float(values["model"].get("internal_score_threshold", 0.0)) <= 1.0:
        raise ConfigError("model.internal_score_threshold must be in [0, 1]")
    for field in (
        "rpn_pre_nms_top_n_train",
        "rpn_post_nms_top_n_train",
        "rpn_pre_nms_top_n_test",
        "rpn_post_nms_top_n_test",
        "box_batch_size_per_image",
    ):
        if values["model"].get(field) is not None and int(values["model"][field]) < 1:
            raise ConfigError(f"model.{field} must be positive when provided")

    if values["scheduler"]["name"] not in {"step", "cosine", "none"}:
        raise ConfigError("scheduler.name must be step, cosine, or none")
    if values["scheduler"]["name"] == "step" and int(values["scheduler"]["step_size"]) < 1:
        raise ConfigError("scheduler.step_size must be positive")
    if not 0.0 <= float(values["evaluation"]["score_threshold"]) <= 1.0:
        raise ConfigError("evaluation.score_threshold must be in [0, 1]")
    if values["evaluation"]["max_batches"] is not None and int(values["evaluation"]["max_batches"]) < 1:
        raise ConfigError("evaluation.max_batches must be positive when provided")
    return TrainingConfig(config_path, values, {str(key): int(value) for key, value in class_map.items()})
