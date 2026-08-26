"""Evidence adapter for separately trained TorchVision box detectors."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from consentguard.stage_03_specialists.common import stable_evidence_id
from consentguard.stage_04_fusion_calibration.domain import Evidence, EvidenceGeometry
from consentguard.stage_05_review_export.ingest import NormalizedImage
from consentguard.stage_05_review_export.redaction.prediction_renderer import resize_for_inference


class BoxDetectorEvidenceProvider:
    """Return raw box evidence while leaving thresholds and review to fusion."""

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        *,
        class_map: Mapping[str, int],
        version: str,
        provider_name: str,
        privacy_class_overrides: Mapping[int, str] | None = None,
        short_side: int = 640,
        max_long_side: int = 1024,
    ) -> None:
        if short_side < 32 or max_long_side < short_side:
            raise ValueError("Require max_long_side >= short_side >= 32")
        id_to_name = {int(class_id): str(name) for name, class_id in class_map.items()}
        if id_to_name.get(0) != "background":
            raise ValueError("class_map must include background class id 0")
        self.model = model.to(device).eval()
        self.device = device
        self.id_to_name = id_to_name
        self.privacy_class_overrides = {
            int(class_id): str(name) for class_id, name in (privacy_class_overrides or {}).items()
        }
        self.name = str(provider_name)
        self.version = str(version)
        self.short_side = int(short_side)
        self.max_long_side = int(max_long_side)

    @torch.inference_mode()
    def analyze(self, image: NormalizedImage) -> list[Evidence]:
        resized, scale = resize_for_inference(image.pixels_rgb, self.short_side, self.max_long_side)
        tensor = (
            torch.from_numpy(resized.transpose(2, 0, 1).copy())
            .float()
            .div_(255.0)
            .to(self.device)
        )
        prediction = self.model([tensor])[0]
        boxes = prediction.get("boxes")
        scores = prediction.get("scores")
        labels = prediction.get("labels")
        if boxes is None or scores is None or labels is None:
            raise ValueError("Box detector prediction must contain boxes, scores, and labels")
        if not (len(boxes) == len(scores) == len(labels)):
            raise ValueError("Box detector prediction fields must have equal lengths")

        inverse_scale = 1.0 / float(scale)
        evidence: list[Evidence] = []
        for index in range(len(scores)):
            class_id = int(labels[index].detach().cpu().item())
            if class_id == 0:
                continue
            privacy_class = self.privacy_class_overrides.get(
                class_id, self.id_to_name.get(class_id, f"class_{class_id}")
            )
            uncertainty_flags = () if class_id in self.id_to_name else ("unknown_model_class",)
            left, top, right, bottom = (
                float(value) * inverse_scale for value in boxes[index].detach().cpu().tolist()
            )
            left = max(0.0, min(float(image.width), left))
            top = max(0.0, min(float(image.height), top))
            right = max(0.0, min(float(image.width), right))
            bottom = max(0.0, min(float(image.height), bottom))
            if right <= left or bottom <= top:
                continue
            payload = {
                "image": image.pixel_sha256,
                "index": index,
                "class_id": class_id,
                "box": [round(left, 4), round(top, 4), round(right, 4), round(bottom, 4)],
            }
            evidence.append(
                Evidence(
                    evidence_id=stable_evidence_id(self.name, self.version, payload),
                    provider=self.name,
                    provider_version=self.version,
                    privacy_class=privacy_class,
                    confidence=max(0.0, min(1.0, float(scores[index].detach().cpu().item()))),
                    geometry=EvidenceGeometry(
                        width=image.width,
                        height=image.height,
                        box_xyxy=(left, top, right, bottom),
                    ),
                    uncertainty_flags=uncertainty_flags,
                    source_detection_id=str(index),
                )
            )
        return evidence
