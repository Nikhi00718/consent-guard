from pathlib import Path

import pytest

from consentguard.stage_03_specialists.plate_yunet import LPDYuNetPlateProvider
from consentguard.stage_04_fusion_calibration.evidence.base import ProviderUnavailableError
from consentguard.stage_05_review_export.ingest import normalize_image


def test_plate_yunet_missing_asset_is_explicit(tmp_path: Path, tmp_path_factory) -> None:
    image_path = tmp_path / "image.png"
    from PIL import Image

    Image.new("RGB", (64, 48), "white").save(image_path)
    image = normalize_image(image_path)
    provider = LPDYuNetPlateProvider(tmp_path / "missing.onnx", version="test")
    with pytest.raises(ProviderUnavailableError, match="weights are missing"):
        provider.analyze(image)
