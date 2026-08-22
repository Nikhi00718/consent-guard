"""OpenCV Zoo LPD-YuNet license-plate geometry adapter.

The decoder follows OpenCV Zoo's Apache-2.0 ``lpd_yunet.py`` reference.  The
released model was trained on Chinese plates, so this provider is evidence
only and is not India-target validated.
"""

from __future__ import annotations

from itertools import product
from pathlib import Path

import cv2
import numpy as np

from consentguard.stage_03_specialists.common import stable_evidence_id
from consentguard.stage_04_fusion_calibration.domain import Evidence, EvidenceGeometry
from consentguard.stage_04_fusion_calibration.evidence.base import ProviderUnavailableError
from consentguard.stage_05_review_export.ingest import NormalizedImage


class LPDYuNetPlateProvider:
    name = "plate_yunet"

    def __init__(
        self,
        model_path: str | Path,
        *,
        version: str,
        confidence_threshold: float = 0.8,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
        keep_top_k: int = 750,
    ) -> None:
        self.model_path = Path(model_path)
        self.version = str(version)
        self.confidence_threshold = float(confidence_threshold)
        self.nms_threshold = float(nms_threshold)
        self.top_k = int(top_k)
        self.keep_top_k = int(keep_top_k)
        self._net = None
        self._input_size: tuple[int, int] | None = None
        self._priors: np.ndarray | None = None
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0, 1]")

    def _load(self, width: int, height: int):
        if not self.model_path.is_file():
            raise ProviderUnavailableError(f"LPD-YuNet weights are missing: {self.model_path}")
        if not hasattr(cv2, "dnn"):
            raise ProviderUnavailableError("OpenCV DNN is unavailable")
        if self._net is None:
            try:
                self._net = cv2.dnn.readNet(str(self.model_path))
            except (cv2.error, OSError) as error:
                raise ProviderUnavailableError(f"LPD-YuNet could not load: {error}") from error
        if self._input_size != (width, height):
            self._input_size = (width, height)
            self._priors = self._prior_boxes(width, height)
        return self._net, self._priors

    @staticmethod
    def _prior_boxes(width: int, height: int) -> np.ndarray:
        min_sizes = [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]]
        steps = [8, 16, 32, 64]
        feature_map_2th = [int(int((height + 1) / 2) / 2), int(int((width + 1) / 2) / 2)]
        feature_maps = [
            [int(feature_map_2th[0] / 2), int(feature_map_2th[1] / 2)],
            [int(feature_map_2th[0] / 4), int(feature_map_2th[1] / 4)],
            [int(feature_map_2th[0] / 8), int(feature_map_2th[1] / 8)],
            [int(feature_map_2th[0] / 16), int(feature_map_2th[1] / 16)],
        ]
        priors = []
        for index, feature_map in enumerate(feature_maps):
            for i, j in product(range(feature_map[0]), range(feature_map[1])):
                for min_size in min_sizes[index]:
                    priors.append(
                        [
                            (j + 0.5) * steps[index] / width,
                            (i + 0.5) * steps[index] / height,
                            min_size / width,
                            min_size / height,
                        ]
                    )
        return np.asarray(priors, dtype=np.float32)

    def analyze(self, image: NormalizedImage) -> list[Evidence]:
        net, priors = self._load(image.width, image.height)
        bgr = cv2.cvtColor(image.pixels_rgb, cv2.COLOR_RGB2BGR)
        blob = cv2.dnn.blobFromImage(bgr)
        net.setInput(blob)
        try:
            loc, conf, iou = net.forward(["loc", "conf", "iou"])
        except cv2.error as error:
            raise ProviderUnavailableError(f"LPD-YuNet inference failed: {error}") from error
        cls_scores = conf[:, 1]
        iou_scores = np.clip(iou[:, 0], 0.0, 1.0)
        scores = np.sqrt(np.clip(cls_scores * iou_scores, 0.0, 1.0))
        scale = np.asarray([image.width, image.height], dtype=np.float32)
        corners = np.hstack(
            (
                (priors[:, 0:2] + loc[:, 4:6] * 0.1 * priors[:, 2:4]) * scale,
                (priors[:, 0:2] + loc[:, 6:8] * 0.1 * priors[:, 2:4]) * scale,
                (priors[:, 0:2] + loc[:, 10:12] * 0.1 * priors[:, 2:4]) * scale,
                (priors[:, 0:2] + loc[:, 12:14] * 0.1 * priors[:, 2:4]) * scale,
            )
        )
        boxes = np.column_stack((corners[:, 0], corners[:, 1], corners[:, 2], corners[:, 3]))
        keep = cv2.dnn.NMSBoxes(
            boxes.tolist(),
            scores.tolist(),
            self.confidence_threshold,
            self.nms_threshold,
            top_k=self.top_k,
        )
        indices = np.asarray(keep).reshape(-1).tolist() if len(keep) else []
        evidence: list[Evidence] = []
        for index in indices[: self.keep_top_k]:
            points = corners[index].reshape(4, 2)
            points[:, 0] = np.clip(points[:, 0], 0.0, float(image.width))
            points[:, 1] = np.clip(points[:, 1], 0.0, float(image.height))
            polygon = tuple((float(x), float(y)) for x, y in points)
            if len(polygon) < 4:
                continue
            payload = {"image": image.pixel_sha256, "index": int(index), "polygon": polygon}
            evidence.append(
                Evidence(
                    evidence_id=stable_evidence_id(self.name, self.version, payload),
                    provider=self.name,
                    provider_version=self.version,
                    privacy_class="license_plate",
                    confidence=max(0.0, min(1.0, float(scores[index]))),
                    geometry=EvidenceGeometry(image.width, image.height, polygon_xy=polygon),
                    source_detection_id=str(index),
                    uncertainty_flags=("trained_on_chinese_plates",),
                )
            )
        return evidence
