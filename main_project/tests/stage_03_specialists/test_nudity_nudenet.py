from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from consentguard.stage_03_specialists.nudity_nudenet import NudeNetIntimateContentProvider
from consentguard.stage_04_fusion_calibration.evidence.base import ProviderUnavailableError
from consentguard.stage_05_review_export.ingest import NormalizedImage


class FakeDetector:
    """Stands in for nudenet.NudeDetector, whose weights are AGPL and optional."""

    def __init__(self, detections: list[dict] | None = None, error: Exception | None = None) -> None:
        self.detections = detections or []
        self.error = error

    def detect(self, _pixels):
        if self.error is not None:
            raise self.error
        return self.detections


def _image(tmp_path: Path, width: int = 40, height: int = 30) -> NormalizedImage:
    source = tmp_path / "photo.jpg"
    source.write_bytes(b"placeholder")
    return NormalizedImage(
        source_path=source,
        source_sha256="a" * 64,
        pixel_sha256="b" * 64,
        pixels_rgb=np.zeros((height, width, 3), dtype=np.uint8),
        source_format="JPEG",
        metadata_categories=(),
        orientation_applied=False,
    )


def _provider(detections: list[dict] | None = None, error: Exception | None = None):
    return NudeNetIntimateContentProvider(
        version="test-v1",
        detector_factory=lambda: FakeDetector(detections, error),
    )


def test_intimate_regions_become_nudity_evidence(tmp_path: Path) -> None:
    provider = _provider(
        [
            {"class": "FEMALE_BREAST_EXPOSED", "score": 0.82, "box": [4, 6, 10, 8]},
            {"class": "BUTTOCKS_COVERED", "score": 0.44, "box": [20, 10, 6, 6]},
        ]
    )
    evidence = provider.analyze(_image(tmp_path))

    assert [item.privacy_class for item in evidence] == ["nudity", "nudity"]
    assert evidence[0].geometry.box_xyxy == (4.0, 6.0, 14.0, 14.0), "box is x,y,w,h in NudeNet output"
    assert evidence[0].uncertainty_flags == ()
    assert evidence[1].uncertainty_flags == ("covered_region",)
    assert all(item.sensitivity_tier == "high" for item in evidence)
    assert len({item.evidence_id for item in evidence}) == 2


def test_faces_and_other_labels_are_left_to_their_own_providers(tmp_path: Path) -> None:
    provider = _provider(
        [
            {"class": "FACE_FEMALE", "score": 0.99, "box": [1, 1, 5, 5]},
            {"class": "FEET_EXPOSED", "score": 0.9, "box": [2, 2, 5, 5]},
            {"class": "BELLY_EXPOSED", "score": 0.9, "box": [3, 3, 5, 5]},
        ]
    )
    assert provider.analyze(_image(tmp_path)) == []


def test_boxes_are_clamped_and_degenerate_ones_dropped(tmp_path: Path) -> None:
    provider = _provider(
        [
            {"class": "ANUS_EXPOSED", "score": 0.7, "box": [35, 25, 500, 500]},
            {"class": "MALE_GENITALIA_EXPOSED", "score": 0.7, "box": [10, 10, 0, 4]},
            {"class": "BUTTOCKS_EXPOSED", "score": 0.7, "box": [1, 2, 3]},
        ]
    )
    evidence = provider.analyze(_image(tmp_path))
    assert len(evidence) == 1
    assert evidence[0].geometry.box_xyxy == (35.0, 25.0, 40.0, 30.0)


def test_a_missing_dependency_is_unavailable_not_empty(tmp_path: Path) -> None:
    provider = NudeNetIntimateContentProvider(version="test-v1", model_path=tmp_path / "absent.onnx")
    with pytest.raises(ProviderUnavailableError, match="weights are missing"):
        provider.analyze(_image(tmp_path))


def test_a_runtime_failure_is_unavailable_not_empty(tmp_path: Path) -> None:
    provider = _provider(error=RuntimeError("onnx session died"))
    with pytest.raises(ProviderUnavailableError, match="inference failed"):
        provider.analyze(_image(tmp_path))
