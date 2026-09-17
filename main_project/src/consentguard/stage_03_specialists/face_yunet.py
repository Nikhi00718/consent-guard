"""OpenCV YuNet face localization without recognition or embeddings."""

from __future__ import annotations

from pathlib import Path

import cv2

from consentguard.stage_03_specialists.common import stable_evidence_id
from consentguard.stage_04_fusion_calibration.domain import Evidence, EvidenceGeometry
from consentguard.stage_04_fusion_calibration.evidence.base import ProviderUnavailableError
from consentguard.stage_05_review_export.ingest import NormalizedImage


class YuNetFaceProvider:
    name = "yunet"

    def __init__(self, model_path: str | Path, *, version: str, backend_score_threshold: float = 0.05) -> None:
        self.model_path = Path(model_path)
        self.version = version
        self.backend_score_threshold = backend_score_threshold
        self._detector = None

    def _load(self, width: int, height: int):
        if not self.model_path.is_file():
            raise ProviderUnavailableError(f"YuNet weights are missing: {self.model_path}")
        if not hasattr(cv2, "FaceDetectorYN"):
            raise ProviderUnavailableError("This OpenCV build does not provide FaceDetectorYN")
        if self._detector is None:
            self._detector = cv2.FaceDetectorYN.create(
                str(self.model_path), "", (width, height), self.backend_score_threshold
            )
        self._detector.setInputSize((width, height))
        return self._detector

    def analyze(self, image: NormalizedImage) -> list[Evidence]:
        detector = self._load(image.width, image.height)
        _, detections = detector.detect(cv2.cvtColor(image.pixels_rgb, cv2.COLOR_RGB2BGR))
        evidence: list[Evidence] = []
        if detections is None:
            return evidence
        for index, row in enumerate(detections):
            x, y, width, height = (float(value) for value in row[:4])
            left, top = max(0.0, x), max(0.0, y)
            right, bottom = min(float(image.width), x + width), min(float(image.height), y + height)
            if right <= left or bottom <= top:
                continue
            score = max(0.0, min(1.0, float(row[-1])))
            payload = {"image": image.pixel_sha256, "index": index, "box": [left, top, right, bottom]}
            evidence.append(
                Evidence(
                    evidence_id=stable_evidence_id(self.name, self.version, payload),
                    provider=self.name,
                    provider_version=self.version,
                    privacy_class="face",
                    confidence=score,
                    geometry=EvidenceGeometry(
                        width=image.width,
                        height=image.height,
                        box_xyxy=(left, top, right, bottom),
                    ),
                    source_detection_id=str(index),
                )
            )
        return evidence
