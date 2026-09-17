"""Optional zxing-cpp barcode adapter with explicit dependency failure."""

from __future__ import annotations

from consentguard.stage_03_specialists.common import stable_evidence_id
from consentguard.stage_04_fusion_calibration.domain import Evidence, EvidenceGeometry
from consentguard.stage_04_fusion_calibration.evidence.base import ProviderUnavailableError
from consentguard.stage_05_review_export.ingest import NormalizedImage


class ZXingBarcodeProvider:
    name = "zxingcpp"
    version = "runtime"

    def analyze(self, image: NormalizedImage) -> list[Evidence]:
        try:
            import zxingcpp
        except ImportError as error:
            raise ProviderUnavailableError(
                "zxing-cpp is not installed; install the specialists optional dependencies"
            ) from error
        evidence: list[Evidence] = []
        for index, result in enumerate(zxingcpp.read_barcodes(image.pixels_rgb)):
            position = result.position
            points = tuple(
                (float(point.x), float(point.y))
                for point in (position.top_left, position.top_right, position.bottom_right, position.bottom_left)
            )
            payload = {"image": image.pixel_sha256, "index": index, "polygon": points}
            evidence.append(
                Evidence(
                    evidence_id=stable_evidence_id(self.name, self.version, payload),
                    provider=self.name,
                    provider_version=self.version,
                    privacy_class="barcode",
                    confidence=1.0,
                    geometry=EvidenceGeometry(image.width, image.height, polygon_xy=points),
                    source_detection_id=str(index),
                )
            )
        return evidence
