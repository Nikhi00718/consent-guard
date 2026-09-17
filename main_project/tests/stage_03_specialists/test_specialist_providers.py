from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from consentguard.stage_03_specialists.face_yunet import YuNetFaceProvider
from consentguard.stage_03_specialists.maskrcnn import MaskRCNNEvidenceProvider
from consentguard.stage_03_specialists.orchestrator import AnalysisOrchestrator
from consentguard.stage_04_fusion_calibration.evidence.base import ProviderUnavailableError
from consentguard.stage_05_review_export.ingest import normalize_image


class _UnavailableProvider:
    name = "missing"
    version = "test"

    def analyze(self, image):
        del image
        raise ProviderUnavailableError("weights missing")


class _FakeMaskRCNN(torch.nn.Module):
    def forward(self, images):
        height, width = images[0].shape[-2:]
        mask = torch.zeros((1, 1, height, width), dtype=torch.float32)
        mask[:, :, height // 4 : height // 2, width // 4 : width // 2] = 1.0
        return [
            {
                "boxes": torch.tensor([[4.0, 5.0, width - 4.0, height - 5.0]]),
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([1]),
                "masks": mask,
            }
        ]


def _image(tmp_path: Path):
    path = tmp_path / "image.jpg"
    assert cv2.imwrite(str(path), np.zeros((24, 32, 3), dtype=np.uint8))
    return normalize_image(path)


def test_orchestrator_preserves_unavailable_provider_state(tmp_path: Path) -> None:
    result = AnalysisOrchestrator((_UnavailableProvider(),)).analyze(_image(tmp_path))
    assert result.evidence == ()
    assert result.unavailable_providers == ("missing",)
    assert result.provider_errors == {"missing": "weights missing"}


def test_yunet_missing_weights_is_not_silently_reported_as_no_faces(tmp_path: Path) -> None:
    provider = YuNetFaceProvider(tmp_path / "missing.onnx", version="test")
    result = AnalysisOrchestrator((provider,)).analyze(_image(tmp_path))
    assert result.unavailable_providers == ("yunet",)


def test_maskrcnn_provider_returns_original_coordinate_mask_evidence(tmp_path: Path) -> None:
    provider = MaskRCNNEvidenceProvider(
        _FakeMaskRCNN(),
        torch.device("cpu"),
        class_map={"background": 0, "face": 1},
        version="checkpoint-test",
        short_side=32,
        max_long_side=64,
    )
    result = provider.analyze(_image(tmp_path))
    assert len(result) == 1
    item = result[0]
    assert item.provider == "maskrcnn"
    assert item.provider_version == "checkpoint-test"
    assert item.privacy_class == "face"
    assert abs(item.confidence - 0.9) < 1e-6
    assert item.geometry.width == 32
    assert item.geometry.height == 24
    assert item.geometry.mask_rle
    assert sum(item.geometry.mask_rle) == 32 * 24
    assert item.geometry.box_xyxy is not None
    left, top, right, bottom = item.geometry.box_xyxy
    assert 0 <= left < right <= 32
    assert 0 <= top < bottom <= 24
