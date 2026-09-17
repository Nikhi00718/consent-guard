"""Memory-safe Visual Redactions instance-segmentation dataset.

The same-release Visual Redactions images can be large and contain many instances. This
loader transforms polygon coordinates first and rasterizes masks only at the
model input resolution.  That avoids allocating one full-resolution mask per
instance before an image is resized.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


from consentguard.shared.paths import project_path


def _resolve_project_path(path: str | Path) -> Path:
    return project_path(path)


def _resized_shape(height: int, width: int, short_side: int | None, max_long_side: int | None) -> tuple[int, int]:
    """Return an aspect-preserving output shape."""

    scale = 1.0
    if short_side is not None:
        scale = float(short_side) / float(min(height, width))
    if max_long_side is not None and max(height, width) * scale > max_long_side:
        scale = float(max_long_side) / float(max(height, width))
    return max(1, round(height * scale)), max(1, round(width * scale))


class VisualRedactionsDataset(Dataset):
    """Return TorchVision-compatible images and instance targets.

    Training can use an instance-centred square crop.  Evaluation always uses
    a deterministic full-image resize so metrics remain comparable.
    """

    def __init__(
        self,
        records_path: str | Path,
        *,
        short_side: int | None = 512,
        max_long_side: int | None = 768,
        crop_size: int | None = None,
        crop_probability: float = 0.0,
        crop_context_factor: float = 4.0,
        min_crop_visibility: float = 0.25,
        horizontal_flip_probability: float = 0.0,
        brightness_contrast_probability: float = 0.0,
        training: bool = False,
        limit: int | None = None,
        subset_seed: int = 0,
    ) -> None:
        self.records_path = _resolve_project_path(records_path)
        if not self.records_path.is_file():
            raise FileNotFoundError(f"Processed records file does not exist: {self.records_path}")
        self.records: list[dict[str, Any]] = [
            json.loads(line)
            for line in self.records_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not self.records:
            raise ValueError(f"Processed records file is empty: {self.records_path}")
        if limit is not None:
            if limit < 1:
                raise ValueError("limit must be positive when provided")
            indices = list(range(len(self.records)))
            random.Random(subset_seed).shuffle(indices)
            self.records = [self.records[index] for index in sorted(indices[:limit])]

        if short_side is not None and short_side < 32:
            raise ValueError("short_side must be at least 32 pixels")
        if max_long_side is not None and max_long_side < 32:
            raise ValueError("max_long_side must be at least 32 pixels")
        if crop_size is not None and crop_size < 32:
            raise ValueError("crop_size must be at least 32 pixels")
        for name, value in (
            ("crop_probability", crop_probability),
            ("min_crop_visibility", min_crop_visibility),
            ("horizontal_flip_probability", horizontal_flip_probability),
            ("brightness_contrast_probability", brightness_contrast_probability),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if crop_context_factor <= 0:
            raise ValueError("crop_context_factor must be positive")

        self.short_side = short_side
        self.max_long_side = max_long_side
        self.crop_size = crop_size
        self.crop_probability = crop_probability if training and crop_size else 0.0
        self.crop_context_factor = crop_context_factor
        self.min_crop_visibility = min_crop_visibility
        self.horizontal_flip_probability = horizontal_flip_probability if training else 0.0
        self.brightness_contrast_probability = brightness_contrast_probability if training else 0.0
        self.training = training

    def __len__(self) -> int:
        return len(self.records)

    def get_height_and_width(self, index: int) -> tuple[int, int]:
        """Allow samplers to inspect dimensions without decoding an image."""

        record = self.records[index]
        if self.crop_size is not None and self.crop_probability >= 1.0:
            return self.crop_size, self.crop_size
        return _resized_shape(
            int(record["height"]),
            int(record["width"]),
            self.short_side,
            self.max_long_side,
        )

    @staticmethod
    def _bbox_xyxy(instance: dict[str, Any]) -> tuple[float, float, float, float]:
        x, y, width, height = (float(value) for value in instance["bbox"])
        return x, y, x + width, y + height

    def _sample_crop(self, record: dict[str, Any], height: int, width: int) -> tuple[int, int, int, int] | None:
        if self.crop_size is None or not record["instances"]:
            return None
        selected = record["instances"][int(torch.randint(len(record["instances"]), (1,)).item())]
        left, top, right, bottom = self._bbox_xyxy(selected)
        box_width = max(1.0, right - left)
        box_height = max(1.0, bottom - top)
        minimum_source_crop = min(64.0, float(min(height, width)))
        side = max(minimum_source_crop, max(box_width, box_height) * self.crop_context_factor)
        side = min(side, float(min(height, width)))

        centre_x = (left + right) / 2.0
        centre_y = (top + bottom) / 2.0
        # Mild centre jitter prevents every crop from placing the object at the
        # exact centre while retaining it inside the crop.
        max_jitter = 0.1 * side
        centre_x += float(torch.empty(1).uniform_(-max_jitter, max_jitter).item())
        centre_y += float(torch.empty(1).uniform_(-max_jitter, max_jitter).item())

        x0 = int(round(centre_x - side / 2.0))
        y0 = int(round(centre_y - side / 2.0))
        side_i = max(1, int(round(side)))
        x0 = min(max(0, x0), max(0, width - side_i))
        y0 = min(max(0, y0), max(0, height - side_i))
        return x0, y0, min(width, x0 + side_i), min(height, y0 + side_i)

    def _render_target(
        self,
        record: dict[str, Any],
        image: np.ndarray,
        crop: tuple[int, int, int, int] | None,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        source_height, source_width = image.shape[:2]
        if crop is None:
            x0, y0, x1, y1 = 0, 0, source_width, source_height
            output_height, output_width = _resized_shape(
                source_height, source_width, self.short_side, self.max_long_side
            )
        else:
            x0, y0, x1, y1 = crop
            output_height = output_width = int(self.crop_size)

        cropped = image[y0:y1, x0:x1]
        interpolation = cv2.INTER_AREA if output_height < cropped.shape[0] or output_width < cropped.shape[1] else cv2.INTER_LINEAR
        if cropped.shape[:2] != (output_height, output_width):
            cropped = cv2.resize(cropped, (output_width, output_height), interpolation=interpolation)

        scale_x = output_width / float(x1 - x0)
        scale_y = output_height / float(y1 - y0)
        masks: list[np.ndarray] = []
        labels: list[int] = []
        iscrowd: list[int] = []

        for instance in record["instances"]:
            left, top, right, bottom = self._bbox_xyxy(instance)
            if crop is not None:
                intersection_width = max(0.0, min(right, x1) - max(left, x0))
                intersection_height = max(0.0, min(bottom, y1) - max(top, y0))
                visibility = (intersection_width * intersection_height) / max(1.0, (right - left) * (bottom - top))
                if visibility < self.min_crop_visibility:
                    continue

            mask = np.zeros((output_height, output_width), dtype=np.uint8)
            for polygon in instance["polygons"]:
                points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2).copy()
                points[:, 0] = (points[:, 0] - x0) * scale_x
                points[:, 1] = (points[:, 1] - y0) * scale_y
                cv2.fillPoly(mask, [np.round(points).astype(np.int32)], 1)
            if not np.any(mask):
                continue
            masks.append(mask)
            labels.append(int(instance["class_id"]))
            iscrowd.append(int(bool(instance.get("iscrowd", False))))

        if not masks:
            empty_masks = np.zeros((0, output_height, output_width), dtype=np.uint8)
            return cropped, {
                "boxes": np.zeros((0, 4), dtype=np.float32),
                "labels": np.zeros((0,), dtype=np.int64),
                "masks": empty_masks,
                "iscrowd": np.zeros((0,), dtype=np.int64),
                "area": np.zeros((0,), dtype=np.float32),
            }

        mask_array = np.stack(masks, axis=0)
        boxes: list[list[float]] = []
        for mask in mask_array:
            ys, xs = np.nonzero(mask)
            boxes.append([float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)])
        areas = mask_array.reshape(mask_array.shape[0], -1).sum(axis=1, dtype=np.int64).astype(np.float32)
        return cropped, {
            "boxes": np.asarray(boxes, dtype=np.float32),
            "labels": np.asarray(labels, dtype=np.int64),
            "masks": mask_array,
            "iscrowd": np.asarray(iscrowd, dtype=np.int64),
            "area": areas,
        }

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor | str]]:
        record = self.records[index]
        image_path = _resolve_project_path(record["image_path"])
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"Could not decode image for record {record['image_id']}: {image_path}")
        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        decoded_height, decoded_width = image.shape[:2]
        if (decoded_height, decoded_width) != (int(record["height"]), int(record["width"])):
            raise RuntimeError(
                f"Image dimensions changed for {record['image_id']}: record="
                f"{record['width']}x{record['height']}, decoded={decoded_width}x{decoded_height}"
            )

        use_crop = self.crop_probability > 0 and torch.rand(1).item() < self.crop_probability
        crop = self._sample_crop(record, decoded_height, decoded_width) if use_crop else None
        transformed, target = self._render_target(record, image, crop)
        # A malformed or extremely thin crop must never silently become a
        # background-only training sample. Fall back to the complete image.
        if crop is not None and len(target["labels"]) == 0:
            crop = None
            transformed, target = self._render_target(record, image, None)
        if len(target["labels"]) == 0 and record["instances"]:
            raise RuntimeError(f"No rasterizable instances remain for record {record['image_id']}")

        if self.horizontal_flip_probability > 0 and torch.rand(1).item() < self.horizontal_flip_probability:
            transformed = np.ascontiguousarray(transformed[:, ::-1])
            target["masks"] = np.ascontiguousarray(target["masks"][:, :, ::-1])
            width = transformed.shape[1]
            boxes = target["boxes"].copy()
            target["boxes"][:, 0] = width - boxes[:, 2]
            target["boxes"][:, 2] = width - boxes[:, 0]

        if self.brightness_contrast_probability > 0 and torch.rand(1).item() < self.brightness_contrast_probability:
            alpha = float(torch.empty(1).uniform_(0.85, 1.15).item())
            beta = float(torch.empty(1).uniform_(-12.0, 12.0).item())
            transformed = np.clip(transformed.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

        image_tensor = torch.from_numpy(np.ascontiguousarray(transformed.transpose(2, 0, 1))).float().div_(255.0)
        target_tensor: dict[str, torch.Tensor | str] = {
            "boxes": torch.from_numpy(np.ascontiguousarray(target["boxes"])).float(),
            "labels": torch.from_numpy(np.ascontiguousarray(target["labels"])).long(),
            "masks": torch.from_numpy(np.ascontiguousarray(target["masks"])).to(torch.uint8),
            "iscrowd": torch.from_numpy(np.ascontiguousarray(target["iscrowd"])).long(),
            "area": torch.from_numpy(np.ascontiguousarray(target["area"])).float(),
            "image_id": torch.tensor([index], dtype=torch.int64),
            "image_key": str(record["image_id"]),
            "source_size": torch.tensor([decoded_height, decoded_width], dtype=torch.int64),
            "size": torch.tensor(list(transformed.shape[:2]), dtype=torch.int64),
            "used_instance_crop": torch.tensor(bool(crop is not None)),
        }
        return image_tensor, target_tensor


def detection_collate(batch: Sequence[tuple[torch.Tensor, dict[str, Any]]]):
    """Keep variable-sized detection images and targets as Python lists."""

    images, targets = zip(*batch)
    return list(images), list(targets)


def target_for_model(target: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    """Strip loader metadata and move only TorchVision target fields."""

    fields = ("boxes", "labels", "masks", "image_id", "area", "iscrowd")
    return {name: target[name].to(device, non_blocking=True) for name in fields}
