from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from consentguard.stage_05_review_export.redaction.prediction_renderer import write_metadata_free_redaction
from consentguard.stage_05_review_export.redaction.prediction_renderer import predict_union_mask


class _PredictionModel(torch.nn.Module):
    def forward(self, images):
        height, width = images[0].shape[-2:]
        masks = torch.zeros((2, 1, height, width), device=images[0].device)
        masks[0, 0, 2:8, 2:8] = 1
        masks[1, 0, 10:16, 10:16] = 1
        return [{
            "scores": torch.tensor([0.3, 0.3], device=images[0].device),
            "labels": torch.tensor([1, 2], device=images[0].device),
            "boxes": torch.tensor([[2, 2, 8, 8], [10, 10, 16, 16]], device=images[0].device),
            "masks": masks,
        }]


def test_redaction_is_newly_encoded_and_reopens(tmp_path: Path) -> None:
    source = tmp_path / "source.jpg"
    destination = tmp_path / "redacted.png"
    bgr = np.full((24, 32, 3), 200, dtype=np.uint8)
    assert cv2.imwrite(str(source), bgr)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mask = np.zeros((24, 32), dtype=np.uint8)
    mask[5:15, 7:20] = 255
    report = write_metadata_free_redaction(source, destination, rgb, mask)
    reopened = cv2.imread(str(destination), cv2.IMREAD_COLOR)
    assert reopened is not None
    assert np.all(reopened[5:15, 7:20] == 0)
    assert report["newly_encoded"] is True
    assert report["source_sha256"] != report["output_sha256"]


def test_redaction_refuses_to_overwrite_source(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    image = np.full((8, 8, 3), 127, dtype=np.uint8)
    assert cv2.imwrite(str(source), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    mask = np.ones((8, 8), dtype=np.uint8) * 255
    with pytest.raises(ValueError, match="overwrite the source"):
        write_metadata_free_redaction(source, source, image, mask)


def test_prediction_uses_per_class_score_thresholds() -> None:
    image = np.zeros((24, 24, 3), dtype=np.uint8)
    mask, detections = predict_union_mask(
        _PredictionModel(),
        image,
        torch.device("cpu"),
        short_side=32,
        max_long_side=32,
        score_threshold=0.5,
        score_thresholds={1: 0.25, 2: 0.4},
        dilation_pixels=0,
    )
    assert [item["class_id"] for item in detections] == [1]
    assert detections[0]["score_threshold_used"] == 0.25
    assert mask.sum() > 0
