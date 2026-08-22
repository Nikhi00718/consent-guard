"""OpenCV Zoo PP-OCR text-geometry adapter.

This is the PaddleOCR/PP-OCRv3 detector exported to ONNX by OpenCV Zoo. It
returns geometry only; recognized strings are never persisted and handwriting
classification remains explicitly unresolved.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from consentguard.stage_03_specialists.common import stable_evidence_id
from consentguard.stage_04_fusion_calibration.domain import Evidence, EvidenceGeometry
from consentguard.stage_04_fusion_calibration.evidence.base import ProviderUnavailableError
from consentguard.stage_05_review_export.ingest import NormalizedImage


class PPOCRTextGeometryProvider:
    name = "paddleocr_onnx"

    def __init__(
        self,
        model_path: str | Path,
        *,
        version: str,
        input_size: tuple[int, int] = (736, 736),
        binary_threshold: float = 0.3,
        polygon_threshold: float = 0.5,
        unclip_ratio: float = 2.0,
    ) -> None:
        self.model_path = Path(model_path)
        self.version = str(version)
        self.input_size = tuple(int(value) for value in input_size)
        self.binary_threshold = float(binary_threshold)
        self.polygon_threshold = float(polygon_threshold)
        self.unclip_ratio = float(unclip_ratio)
        self._detector = None

    def _load(self):
        if not self.model_path.is_file():
            raise ProviderUnavailableError(f"PP-OCR ONNX weights are missing: {self.model_path}")
        if not hasattr(cv2, "dnn_TextDetectionModel_DB"):
            raise ProviderUnavailableError("This OpenCV build lacks DB text detection")
        if self._detector is None:
            try:
                detector = cv2.dnn_TextDetectionModel_DB(cv2.dnn.readNet(str(self.model_path)))
                detector.setInputSize(self.input_size)
                detector.setInputMean((123.675, 116.28, 103.53))
                detector.setInputScale(1.0 / 255.0 / np.array([0.229, 0.224, 0.225]))
                detector.setBinaryThreshold(self.binary_threshold)
                detector.setPolygonThreshold(self.polygon_threshold)
                detector.setUnclipRatio(self.unclip_ratio)
                self._detector = detector
            except (cv2.error, OSError) as error:
                raise ProviderUnavailableError(f"PP-OCR ONNX could not load: {error}") from error
        return self._detector

    def analyze(self, image: NormalizedImage) -> list[Evidence]:
        detector = self._load()
        bgr = cv2.cvtColor(image.pixels_rgb, cv2.COLOR_RGB2BGR)
        try:
            polygons, confidences = detector.detect(bgr)
        except cv2.error as error:
            raise ProviderUnavailableError(f"PP-OCR ONNX inference failed: {error}") from error
        if polygons is None:
            return []
        polygons_array = np.asarray(polygons)
        scores_array = np.asarray(confidences).reshape(-1) if confidences is not None else np.ones(len(polygons_array))
        evidence: list[Evidence] = []
        for index, polygon in enumerate(polygons_array):
            points = tuple(
                (
                    max(0.0, min(float(image.width), float(point[0]))),
                    max(0.0, min(float(image.height), float(point[1]))),
                )
                for point in np.asarray(polygon).reshape(-1, 2)
            )
            if len(points) < 3:
                continue
            score = float(scores_array[index]) if index < len(scores_array) else 0.5
            payload = {"image": image.pixel_sha256, "index": index, "polygon": points}
            evidence.append(
                Evidence(
                    evidence_id=stable_evidence_id(self.name, self.version, payload),
                    provider=self.name,
                    provider_version=self.version,
                    privacy_class="printed_text",
                    confidence=max(0.0, min(1.0, score)),
                    geometry=EvidenceGeometry(image.width, image.height, polygon_xy=points),
                    source_detection_id=str(index),
                    uncertainty_flags=("handwriting_unresolved",),
                )
            )
        return evidence
