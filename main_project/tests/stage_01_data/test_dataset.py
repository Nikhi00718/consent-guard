from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch

from consentguard.stage_01_data.dataset import VisualRedactionsDataset, target_for_model


def _fixture(tmp_path: Path) -> Path:
    image_path = tmp_path / "image.jpg"
    image = np.full((100, 160, 3), 127, dtype=np.uint8)
    assert cv2.imwrite(str(image_path), image)
    record = {
        "image_id": "fixture-1",
        "image_path": str(image_path),
        "height": 100,
        "width": 160,
        "instances": [
            {
                "class_id": 1,
                "bbox": [40.0, 20.0, 50.0, 40.0],
                "polygons": [[40.0, 20.0, 90.0, 20.0, 90.0, 60.0, 40.0, 60.0]],
                "iscrowd": False,
            }
        ],
    }
    records_path = tmp_path / "records.jsonl"
    records_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    return records_path


def _negative_fixture(tmp_path: Path) -> Path:
    image_path = tmp_path / "negative.jpg"
    image = np.full((80, 120, 3), 90, dtype=np.uint8)
    assert cv2.imwrite(str(image_path), image)
    record = {
        "image_id": "negative-1",
        "image_path": str(image_path),
        "height": 80,
        "width": 120,
        "negative_for_profile": True,
        "instances": [],
    }
    records_path = tmp_path / "negative-records.jsonl"
    records_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    return records_path


def test_full_resize_produces_valid_torchvision_target(tmp_path: Path) -> None:
    dataset = VisualRedactionsDataset(_fixture(tmp_path), short_side=64, max_long_side=96)
    image, target = dataset[0]
    assert tuple(image.shape) == (3, 60, 96)
    assert target["masks"].shape == (1, 60, 96)
    assert target["labels"].tolist() == [1]
    assert torch.all(target["boxes"][:, 2:] > target["boxes"][:, :2])
    assert torch.equal(target["area"], target["masks"].flatten(1).sum(1).float())
    assert set(target_for_model(target, torch.device("cpu"))) == {"boxes", "labels", "masks", "image_id", "area", "iscrowd"}


def test_instance_crop_is_square_and_nonempty(tmp_path: Path) -> None:
    torch.manual_seed(3)
    dataset = VisualRedactionsDataset(
        _fixture(tmp_path),
        short_side=64,
        max_long_side=96,
        crop_size=64,
        crop_probability=1.0,
        crop_context_factor=2.0,
        training=True,
    )
    image, target = dataset[0]
    assert tuple(image.shape) == (3, 64, 64)
    assert target["used_instance_crop"].item() is True
    assert target["masks"].sum() > 0
    assert target["boxes"].min() >= 0
    assert target["boxes"].max() <= 64


def test_profile_negative_produces_valid_empty_torchvision_target(tmp_path: Path) -> None:
    dataset = VisualRedactionsDataset(
        _negative_fixture(tmp_path),
        short_side=64,
        max_long_side=96,
        crop_size=64,
        crop_probability=1.0,
        training=True,
    )
    image, target = dataset[0]
    assert tuple(image.shape) == (3, 64, 96)
    assert target["boxes"].shape == (0, 4)
    assert target["labels"].shape == (0,)
    assert target["masks"].shape == (0, 64, 96)
    assert target["used_instance_crop"].item() is False
