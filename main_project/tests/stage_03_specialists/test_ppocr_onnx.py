from pathlib import Path

import pytest

from consentguard.stage_03_specialists.ppocr_onnx import PPOCRTextGeometryProvider
from consentguard.stage_04_fusion_calibration.evidence.base import ProviderUnavailableError
from consentguard.stage_05_review_export.ingest import normalize_image


def test_ppocr_missing_asset_is_explicit(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    from PIL import Image

    Image.new("RGB", (64, 48), "white").save(image_path)
    image = normalize_image(image_path)
    provider = PPOCRTextGeometryProvider(tmp_path / "missing.onnx", version="test")
    with pytest.raises(ProviderUnavailableError, match="weights are missing"):
        provider.analyze(image)
