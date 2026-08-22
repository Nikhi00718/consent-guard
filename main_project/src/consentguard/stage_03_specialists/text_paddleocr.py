"""Optional PaddleOCR text-geometry adapter; recognized strings are discarded."""

from __future__ import annotations

from typing import Any

from consentguard.stage_03_specialists.common import stable_evidence_id
from consentguard.stage_04_fusion_calibration.domain import Evidence, EvidenceGeometry
from consentguard.stage_04_fusion_calibration.evidence.base import ProviderUnavailableError
from consentguard.stage_05_review_export.ingest import NormalizedImage


class PaddleOCRTextProvider:
    name = "paddleocr"

    def __init__(self, engine: Any | None = None, *, version: str = "runtime") -> None:
        self.engine = engine
        self.version = version

    def _engine(self):
        if self.engine is not None:
            return self.engine
        try:
            from paddleocr import PaddleOCR
        except ImportError as error:
            raise ProviderUnavailableError(
                "PaddleOCR is not installed; use its isolated optional environment"
            ) from error
        try:
            self.engine = PaddleOCR(
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
        except Exception as error:
            raise ProviderUnavailableError(f"PaddleOCR could not initialize: {error}") from error
        return self.engine

    def analyze(self, image: NormalizedImage) -> list[Evidence]:
        try:
            pages = self._engine().predict(image.pixels_rgb)
        except Exception as error:
            raise ProviderUnavailableError(f"PaddleOCR inference failed: {error}") from error
        evidence: list[Evidence] = []
        index = 0
        for page in pages:
            payload = getattr(page, "json", page)
            if callable(payload):
                payload = payload()
            if isinstance(payload, dict) and "res" in payload:
                payload = payload["res"]
            if not isinstance(payload, dict):
                continue
            polygons = payload.get("dt_polys") or payload.get("rec_polys") or []
            scores = payload.get("dt_scores") or payload.get("rec_scores") or []
            for polygon_index, polygon in enumerate(polygons):
                points = tuple((float(point[0]), float(point[1])) for point in polygon)
                if len(points) < 3:
                    continue
                score = float(scores[polygon_index]) if polygon_index < len(scores) else 0.5
                geometry_payload = {"image": image.pixel_sha256, "index": index, "polygon": points}
                evidence.append(
                    Evidence(
                        evidence_id=stable_evidence_id(self.name, self.version, geometry_payload),
                        provider=self.name,
                        provider_version=self.version,
                        privacy_class="printed_text",
                        confidence=max(0.0, min(1.0, score)),
                        geometry=EvidenceGeometry(image.width, image.height, polygon_xy=points),
                        source_detection_id=str(index),
                        uncertainty_flags=("printed_vs_handwriting_unresolved",),
                    )
                )
                index += 1
        return evidence
