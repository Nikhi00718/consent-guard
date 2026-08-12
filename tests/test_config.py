from __future__ import annotations

import copy

import pytest

from consentguard.config import (
    ConfigError,
    load_training_config,
    validate_checkpoint_inference_compatibility,
)


def test_smoke_config_and_class_map_are_valid() -> None:
    config = load_training_config("configs/train_smoke.yaml")
    assert config.num_classes == 10
    assert config.class_map["background"] == 0
    assert set(config.class_map) == {
        "background",
        "a105_face_all",
        "a108_license_plate_all",
        "a109_person_body",
        "a110_nudity_all",
        "a26_handwriting",
        "a39_disability_physical",
        "a43_medicine",
        "a7_fingerprint",
        "a8_signature",
    }
    assert config.section("training")["max_steps"] == 1


def test_checkpoint_configuration_mismatch_is_rejected() -> None:
    config = load_training_config("configs/train_smoke.yaml")
    checkpoint = {"class_map": config.class_map, "config": config.as_dict()}
    validate_checkpoint_inference_compatibility(checkpoint, config)

    incompatible = copy.deepcopy(checkpoint)
    incompatible["config"]["model"]["small_object_anchors"] = False
    with pytest.raises(ConfigError, match="model configuration"):
        validate_checkpoint_inference_compatibility(incompatible, config)
