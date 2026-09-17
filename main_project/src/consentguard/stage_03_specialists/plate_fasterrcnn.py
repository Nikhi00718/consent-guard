"""Adapter for a separately trained TorchVision plate detector."""

from __future__ import annotations

import torch

from consentguard.stage_03_specialists.common import stable_evidence_id
from consentguard.stage_04_fusion_calibration.domain import Evidence, EvidenceGeometry
from consentguard.stage_05_review_export.ingest import NormalizedImage


class PlateFasterRCNNProvider:
    name = "plate_fasterrcnn"

    def __init__(self, model: torch.nn.Module, device: torch.device, *, version: str) -> None:
        self.model = model.to(device).eval()
        self.device = device
        self.version = version

    @torch.inference_mode()
    def analyze(self, image: NormalizedImage) -> list[Evidence]:
        tensor = torch.from_numpy(image.pixels_rgb.transpose(2, 0, 1).copy()).float().div_(255.0)
        prediction = self.model([tensor.to(self.device)])[0]
        evidence: list[Evidence] = []
        for index, (box, score) in enumerate(zip(prediction["boxes"], prediction["scores"])):
            left, top, right, bottom = (float(value) for value in box.detach().cpu().tolist())
            left, top = max(0.0, left), max(0.0, top)
            right, bottom = min(float(image.width), right), min(float(image.height), bottom)
            if right <= left or bottom <= top:
                continue
            confidence = max(0.0, min(1.0, float(score.detach().cpu())))
            payload = {"image": image.pixel_sha256, "index": index, "box": [left, top, right, bottom]}
            evidence.append(
                Evidence(
                    evidence_id=stable_evidence_id(self.name, self.version, payload),
                    provider=self.name,
                    provider_version=self.version,
                    privacy_class="license_plate",
                    confidence=confidence,
                    geometry=EvidenceGeometry(image.width, image.height, (left, top, right, bottom)),
                    source_detection_id=str(index),
                )
            )
        return evidence
