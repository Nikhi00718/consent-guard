"""Mask R-CNN evidence provider for the broad visual privacy classes."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping

import cv2
import numpy as np
import torch

from consentguard.stage_03_specialists.common import stable_evidence_id
from consentguard.stage_04_fusion_calibration.domain import Evidence, EvidenceGeometry
from consentguard.stage_05_review_export.ingest import NormalizedImage
from consentguard.stage_05_review_export.redaction.prediction_renderer import resize_for_inference


def _binary_mask_rle(mask: np.ndarray) -> tuple[int, ...]:
    """Encode a binary mask as row-major alternating zero/one run lengths."""

    if mask.ndim != 2 or mask.dtype != np.bool_:
        raise ValueError("mask must be a two-dimensional boolean array")
    flat = mask.reshape(-1, order="C").astype(np.uint8, copy=False)
    if flat.size == 0:
        raise ValueError("mask must not be empty")
    runs: list[int] = []
    current = 0
    count = 0
    for value in flat.tolist():
        value = int(value)
        if value != current:
            runs.append(count)
            current = value
            count = 1
        else:
            count += 1
    runs.append(count)
    if sum(runs) != int(flat.size):
        raise RuntimeError("mask RLE does not cover the complete mask")
    return tuple(runs)


class MaskRCNNEvidenceProvider:
    """Adapt a trained instance-segmentation model to the evidence contract.

    The provider deliberately returns raw detections.  Score thresholds,
    geometry expansion, and mandatory-review decisions belong to Stage 04's
    versioned fusion profile, so this adapter does not make release decisions.
    """

    name = "maskrcnn"

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        *,
        class_map: Mapping[str, int],
        version: str,
        provider_name: str = "maskrcnn",
        short_side: int = 640,
        max_long_side: int = 1024,
        mask_threshold: float = 0.5,
    ) -> None:
        if short_side < 32 or max_long_side < short_side:
            raise ValueError("Require max_long_side >= short_side >= 32")
        if not 0.0 <= mask_threshold <= 1.0:
            raise ValueError("mask_threshold must be in [0, 1]")
        id_to_name = {int(class_id): str(name) for name, class_id in class_map.items()}
        if 0 not in id_to_name:
            raise ValueError("class_map must include background class id 0")
        self.model = model.to(device).eval()
        self.device = device
        self.id_to_name = id_to_name
        self.name = str(provider_name)
        self.version = str(version)
        self.short_side = int(short_side)
        self.max_long_side = int(max_long_side)
        self.mask_threshold = float(mask_threshold)

    @torch.inference_mode()
    def analyze(self, image: NormalizedImage) -> list[Evidence]:
        resized, scale = resize_for_inference(
            image.pixels_rgb,
            self.short_side,
            self.max_long_side,
        )
        tensor = (
            torch.from_numpy(np.ascontiguousarray(resized.transpose(2, 0, 1)))
            .float()
            .div_(255.0)
            .to(self.device)
        )
        prediction = self.model([tensor])[0]
        boxes = prediction.get("boxes")
        scores = prediction.get("scores")
        labels = prediction.get("labels")
        masks = prediction.get("masks")
        if boxes is None or scores is None or labels is None or masks is None:
            raise ValueError("Mask R-CNN prediction must contain boxes, scores, labels, and masks")
        if not (len(boxes) == len(scores) == len(labels) == len(masks)):
            raise ValueError("Mask R-CNN prediction fields must have equal lengths")

        evidence: list[Evidence] = []
        inverse_scale = 1.0 / float(scale)
        for index in range(len(scores)):
            class_id = int(labels[index].detach().cpu().item())
            if class_id == 0:
                continue
            privacy_class = self.id_to_name.get(class_id, f"class_{class_id}")
            uncertainty_flags: tuple[str, ...] = ()
            if class_id not in self.id_to_name:
                uncertainty_flags = ("unknown_model_class",)

            box_values = [float(value) * inverse_scale for value in boxes[index].detach().cpu().tolist()]
            left, top, right, bottom = box_values
            left = max(0.0, min(float(image.width), left))
            top = max(0.0, min(float(image.height), top))
            right = max(0.0, min(float(image.width), right))
            bottom = max(0.0, min(float(image.height), bottom))
            if right <= left or bottom <= top:
                continue

            mask = masks[index, 0].detach().cpu().numpy() >= self.mask_threshold
            if mask.shape != resized.shape[:2]:
                mask = cv2.resize(
                    mask.astype(np.uint8),
                    (resized.shape[1], resized.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            if mask.shape != image.pixels_rgb.shape[:2]:
                mask = cv2.resize(
                    mask.astype(np.uint8),
                    (image.width, image.height),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
            if not bool(mask.any()):
                continue
            mask_rle = _binary_mask_rle(mask)
            payload = {
                "image": image.pixel_sha256,
                "index": index,
                "class_id": class_id,
                "box": [round(left, 4), round(top, 4), round(right, 4), round(bottom, 4)],
                "mask_sha256": hashlib.sha256(mask.tobytes(order="C")).hexdigest(),
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
                        mask_rle=mask_rle,
                    ),
                    uncertainty_flags=uncertainty_flags,
                    source_detection_id=str(index),
                )
            )
        return evidence
