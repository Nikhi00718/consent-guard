from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from consentguard.stage_03_specialists.tiled import TiledEvidenceProvider, tile_origins
from consentguard.stage_04_fusion_calibration.domain import Evidence, EvidenceGeometry
from consentguard.stage_04_fusion_calibration.evidence.base import ProviderUnavailableError
from consentguard.stage_04_fusion_calibration.evidence.geometry import (
    decode_binary_mask,
    encode_binary_mask,
)
from consentguard.stage_05_review_export.ingest import NormalizedImage


class CornerProvider:
    """Report one box and mask covering the top-left 4x4 pixels of every tile."""

    name = "corner"
    version = "corner-v1"

    def __init__(self) -> None:
        self.seen: list[tuple[int, int]] = []

    def analyze(self, image: NormalizedImage) -> list[Evidence]:
        self.seen.append((image.width, image.height))
        mask = np.zeros((image.height, image.width), dtype=np.uint8)
        mask[0:4, 0:4] = 1
        return [
            Evidence(
                evidence_id=f"corner-{image.pixel_sha256}",
                provider=self.name,
                provider_version=self.version,
                privacy_class="face",
                geometry=EvidenceGeometry(
                    width=image.width,
                    height=image.height,
                    box_xyxy=(0.0, 0.0, 4.0, 4.0),
                    mask_rle=encode_binary_mask(mask),
                ),
                confidence=0.9,
            )
        ]


class BrokenProvider:
    name = "broken"
    version = "broken-v1"

    def analyze(self, _image: NormalizedImage) -> list[Evidence]:
        raise ProviderUnavailableError("weights missing")


def _image(width: int, height: int, tmp_path: Path) -> NormalizedImage:
    source = tmp_path / "source.png"
    source.write_bytes(b"not-read-by-the-provider")
    return NormalizedImage(
        source_path=source,
        source_sha256="a" * 64,
        pixel_sha256="b" * 64,
        pixels_rgb=np.zeros((height, width, 3), dtype=np.uint8),
        source_format="PNG",
        metadata_categories=(),
        orientation_applied=False,
    )


def test_tile_origins_cover_the_extent_and_end_flush() -> None:
    origins = tile_origins(1000, 400, 300)
    assert origins[0] == 0
    assert origins[-1] == 600
    assert all(second - first <= 300 for first, second in zip(origins, origins[1:]))


def test_small_images_are_not_tiled(tmp_path: Path) -> None:
    provider = CornerProvider()
    tiled = TiledEvidenceProvider(provider, tile_size=256, min_image_side=1024)
    assert tiled.analyze(_image(640, 480, tmp_path)) == []
    assert provider.seen == []


def test_tiles_are_remapped_into_original_coordinates(tmp_path: Path) -> None:
    provider = CornerProvider()
    tiled = TiledEvidenceProvider(provider, tile_size=512, overlap=0.0, min_image_side=512)
    evidence = tiled.analyze(_image(1024, 1024, tmp_path))

    assert [item.geometry.box_xyxy for item in evidence] == [
        (0.0, 0.0, 4.0, 4.0),
        (512.0, 0.0, 516.0, 4.0),
        (0.0, 512.0, 4.0, 516.0),
        (512.0, 512.0, 516.0, 516.0),
    ]
    assert {item.provider for item in evidence} == {"corner_tiled"}
    assert all("tiled_second_pass" in item.uncertainty_flags for item in evidence)
    for item in evidence:
        assert item.geometry.width == 1024
        assert item.geometry.height == 1024
        mask = decode_binary_mask(item.geometry.mask_rle, 1024, 1024)
        left, top = int(item.geometry.box_xyxy[0]), int(item.geometry.box_xyxy[1])
        assert mask[top : top + 4, left : left + 4].all()
        assert int(mask.sum()) == 16


def test_tile_plan_respects_the_tile_budget(tmp_path: Path) -> None:
    tiled = TiledEvidenceProvider(CornerProvider(), tile_size=256, max_tiles=6, min_image_side=512)
    assert len(tiled.plan(4000, 3000)) <= 6


def test_unavailable_provider_still_raises(tmp_path: Path) -> None:
    tiled = TiledEvidenceProvider(BrokenProvider(), tile_size=256, overlap=0.0, min_image_side=256)
    with pytest.raises(ProviderUnavailableError):
        tiled.analyze(_image(1024, 1024, tmp_path))


def test_mask_rle_roundtrip_handles_edges() -> None:
    assert decode_binary_mask(encode_binary_mask(np.ones((3, 4), dtype=np.uint8)), 3, 4).all()
    assert not decode_binary_mask(encode_binary_mask(np.zeros((3, 4), dtype=np.uint8)), 3, 4).any()
    checker = np.indices((6, 6)).sum(axis=0) % 2
    assert np.array_equal(decode_binary_mask(encode_binary_mask(checker), 6, 6), checker)
