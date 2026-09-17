"""Re-run any evidence provider over overlapping tiles.

The trained detectors are weakest on small regions: validation segmentation mAP
is about 0.075 for small objects, and RPN proposal recall drops from roughly
94% (large) to 62% (small). A distant plate or a caption occupies a handful of
pixels once the full image is resized to the model's short side, so it is never
proposed.

Cutting the image into overlapping tiles and analyzing each tile at the model's
native input size makes those regions large again. This wrapper owns only the
geometry bookkeeping: it crops, delegates, and maps every detection back into
original-image coordinates. Thresholds, fusion, and release decisions stay in
Stage 04 and Stage 05, and a missing dependency still surfaces as
``ProviderUnavailableError`` so the provider can never look like "nothing
found".
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from consentguard.stage_03_specialists.common import stable_evidence_id
from consentguard.stage_04_fusion_calibration.domain import Evidence, EvidenceGeometry
from consentguard.stage_04_fusion_calibration.evidence.base import EvidenceProvider
from consentguard.stage_04_fusion_calibration.evidence.geometry import (
    decode_binary_mask,
    encode_binary_mask,
)
from consentguard.stage_05_review_export.ingest import NormalizedImage


def tile_origins(extent: int, tile: int, step: int) -> tuple[int, ...]:
    """Tile start offsets covering ``extent`` with the final tile flush right."""

    if extent <= tile:
        return (0,)
    origins = list(range(0, max(1, extent - tile + 1), step))
    last = extent - tile
    if origins[-1] != last:
        origins.append(last)
    return tuple(origins)


class TiledEvidenceProvider:
    """Wrap a provider so small regions are analyzed at native resolution."""

    def __init__(
        self,
        provider: EvidenceProvider,
        *,
        provider_name: str | None = None,
        tile_size: int = 768,
        overlap: float = 0.25,
        min_image_side: int = 1024,
        max_tiles: int = 24,
    ) -> None:
        if tile_size < 64:
            raise ValueError("tile_size must be at least 64 pixels")
        if not 0.0 <= overlap < 0.9:
            raise ValueError("overlap must be in [0, 0.9)")
        if max_tiles < 1:
            raise ValueError("max_tiles must be positive")
        self.provider = provider
        self.name = str(provider_name or f"{provider.name}_tiled")
        self.version = f"tiled-{tile_size}px:{provider.version}"
        self.tile_size = int(tile_size)
        self.overlap = float(overlap)
        self.min_image_side = int(min_image_side)
        self.max_tiles = int(max_tiles)

    def plan(self, width: int, height: int) -> tuple[tuple[int, int, int, int], ...]:
        """Tile rectangles, growing the tile until the count fits ``max_tiles``."""

        if max(width, height) < self.min_image_side:
            return ()
        tile = self.tile_size
        while True:
            step = max(1, int(round(tile * (1.0 - self.overlap))))
            xs = tile_origins(width, min(tile, width), step)
            ys = tile_origins(height, min(tile, height), step)
            if len(xs) * len(ys) <= self.max_tiles or tile >= max(width, height):
                break
            tile = int(tile * 1.5)
        if len(xs) * len(ys) <= 1:
            return ()
        return tuple(
            (x, y, min(x + tile, width), min(y + tile, height))
            for y in ys
            for x in xs
        )

    def analyze(self, image: NormalizedImage) -> list[Evidence]:
        evidence: list[Evidence] = []
        for left, top, right, bottom in self.plan(image.width, image.height):
            crop = np.ascontiguousarray(image.pixels_rgb[top:bottom, left:right])
            if crop.size == 0:
                continue
            tile_image = replace(
                image,
                pixels_rgb=crop,
                pixel_sha256=f"{image.pixel_sha256}:tile:{left}:{top}",
            )
            for item in self.provider.analyze(tile_image):
                evidence.append(self._to_original(item, image, left, top))
        return evidence

    def _to_original(
        self,
        item: Evidence,
        image: NormalizedImage,
        left: int,
        top: int,
    ) -> Evidence:
        geometry = item.geometry
        box = geometry.box_xyxy
        if box is not None:
            shifted = (
                min(float(image.width), box[0] + left),
                min(float(image.height), box[1] + top),
                min(float(image.width), box[2] + left),
                min(float(image.height), box[3] + top),
            )
            box = shifted if shifted[2] > shifted[0] and shifted[3] > shifted[1] else None
        polygon = tuple(
            (
                min(float(image.width), point[0] + left),
                min(float(image.height), point[1] + top),
            )
            for point in geometry.polygon_xy
        )
        mask_rle: tuple[int, ...] = ()
        if geometry.mask_rle:
            tile_mask = decode_binary_mask(geometry.mask_rle, geometry.height, geometry.width)
            full = np.zeros((image.height, image.width), dtype=np.uint8)
            full[top : top + geometry.height, left : left + geometry.width] = tile_mask
            mask_rle = encode_binary_mask(full)
        if box is None and not polygon and not mask_rle:
            raise ValueError(f"Tiled evidence {item.evidence_id} lost its geometry during remapping")
        payload = {
            "image": image.pixel_sha256,
            "tile": [left, top],
            "source": item.evidence_id,
            "class": item.privacy_class,
        }
        return Evidence(
            evidence_id=stable_evidence_id(self.name, self.version, payload),
            provider=self.name,
            provider_version=self.version,
            privacy_class=item.privacy_class,
            geometry=EvidenceGeometry(
                width=image.width,
                height=image.height,
                box_xyxy=box,
                polygon_xy=polygon,
                mask_rle=mask_rle,
            ),
            confidence=item.confidence,
            uncertainty_flags=tuple(sorted({*item.uncertainty_flags, "tiled_second_pass"})),
            source_detection_id=item.source_detection_id,
            sensitivity_tier=item.sensitivity_tier,
        )
