"""Failure-stage diagnostics for privacy-oriented instance segmentation."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

import torch
import torch.nn.functional as F


def _box_iou(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """Return pairwise IoU for ``xyxy`` boxes without requiring pycocotools."""

    if boxes_a.numel() == 0 or boxes_b.numel() == 0:
        return boxes_a.new_zeros((boxes_a.shape[0], boxes_b.shape[0]))
    top_left = torch.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    bottom_right = torch.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    intersection_wh = (bottom_right - top_left).clamp(min=0)
    intersection = intersection_wh[..., 0] * intersection_wh[..., 1]
    area_a = ((boxes_a[:, 2] - boxes_a[:, 0]).clamp(min=0) * (boxes_a[:, 3] - boxes_a[:, 1]).clamp(min=0))
    area_b = ((boxes_b[:, 2] - boxes_b[:, 0]).clamp(min=0) * (boxes_b[:, 3] - boxes_b[:, 1]).clamp(min=0))
    union = area_a[:, None] + area_b[None, :] - intersection
    return intersection / union.clamp(min=torch.finfo(intersection.dtype).eps)


def _size_bucket(box: torch.Tensor) -> str:
    area = float(((box[2] - box[0]).clamp(min=0) * (box[3] - box[1]).clamp(min=0)).item())
    if area < 32.0**2:
        return "small"
    if area < 96.0**2:
        return "medium"
    return "large"


def _mask_at_image_size(mask: torch.Tensor, height: int, width: int) -> torch.Tensor:
    mask = mask.float()
    if mask.shape[-2:] != (height, width):
        mask = F.interpolate(mask[None, None], size=(height, width), mode="bilinear", align_corners=False)[0, 0]
    return mask


def _dilate(mask: torch.Tensor, pixels: int) -> torch.Tensor:
    if pixels <= 0:
        return mask
    kernel = 2 * int(pixels) + 1
    return F.max_pool2d(mask.float()[None, None], kernel, stride=1, padding=int(pixels))[0, 0] > 0


def _name_for_class(class_id: int, class_map: dict[str, int] | None) -> str:
    for name, mapped_id in (class_map or {}).items():
        if int(mapped_id) == class_id:
            return str(name)
    return str(class_id)


@torch.inference_mode()
def evaluate_privacy_coverage(
    model: torch.nn.Module,
    data_loader: Iterable,
    device: torch.device,
    *,
    score_threshold: float = 0.5,
    mask_threshold: float = 0.5,
    dilation_pixels: int = 0,
    instance_coverage_threshold: float = 0.8,
    max_batches: int | None = None,
    class_map: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Measure union redaction coverage separately from COCO AP."""

    if not 0.0 <= score_threshold <= 1.0:
        raise ValueError("score_threshold must be in [0, 1]")
    if not 0.0 <= mask_threshold <= 1.0:
        raise ValueError("mask_threshold must be in [0, 1]")
    if dilation_pixels < 0:
        raise ValueError("dilation_pixels must be non-negative")
    if not 0.0 <= instance_coverage_threshold <= 1.0:
        raise ValueError("instance_coverage_threshold must be in [0, 1]")
    if max_batches is not None and max_batches < 1:
        raise ValueError("max_batches must be positive when provided")

    totals = defaultdict(int)
    per_class: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    batches = 0
    images_seen = 0
    model.eval()
    for images, targets in data_loader:
        if max_batches is not None and batches >= max_batches:
            break
        predictions = model([image.to(device, non_blocking=True) for image in images])
        if len(predictions) != len(targets):
            raise RuntimeError("Model prediction count does not match target count")
        for prediction, target in zip(predictions, targets):
            target_masks = target["masks"].to(device)
            if target_masks.ndim != 3:
                raise RuntimeError(f"Expected target masks [N,H,W], received {tuple(target_masks.shape)}")
            height, width = target_masks.shape[-2:]
            gt_union = target_masks.bool().any(dim=0)
            predicted_union = torch.zeros((height, width), dtype=torch.bool, device=device)
            scores = prediction["scores"].detach()
            keep = torch.nonzero(scores >= score_threshold, as_tuple=False).flatten()
            for prediction_index in keep.tolist():
                mask = _mask_at_image_size(prediction["masks"][prediction_index, 0], height, width)
                predicted_union |= mask >= mask_threshold
            predicted_union = _dilate(predicted_union, dilation_pixels)

            covered = gt_union & predicted_union
            extra = predicted_union & ~gt_union
            gt_pixels = int(gt_union.sum().item())
            covered_pixels = int(covered.sum().item())
            extra_pixels = int(extra.sum().item())
            totals["gt_pixels"] += gt_pixels
            totals["covered_gt_pixels"] += covered_pixels
            totals["predicted_union_pixels"] += int(predicted_union.sum().item())
            totals["over_redaction_pixels"] += extra_pixels
            totals["image_pixels"] += int(height * width)
            totals["images"] += 1
            for mask, label in zip(target_masks, target["labels"].to(device)):
                label_id = int(label.item())
                class_name = _name_for_class(label_id, class_map)
                class_mask = mask.bool()
                class_pixels = int(class_mask.sum().item())
                class_covered = int((class_mask & predicted_union).sum().item())
                entry = per_class[class_name]
                entry["class_id"] = label_id
                entry["instances"] += 1
                entry["gt_pixels"] += class_pixels
                entry["covered_gt_pixels"] += class_covered
                if class_pixels and class_covered / class_pixels >= instance_coverage_threshold:
                    entry["instances_coverage_at_least_threshold"] += 1
            images_seen += 1
        batches += 1

    if images_seen == 0:
        raise RuntimeError("Evaluation loader produced no images")
    gt_pixels = totals["gt_pixels"]
    covered_pixels = totals["covered_gt_pixels"]
    result_per_class: dict[str, dict[str, float | int]] = {}
    for class_name, entry in sorted(per_class.items()):
        class_gt = entry["gt_pixels"]
        class_covered = entry["covered_gt_pixels"]
        result_per_class[class_name] = {
            **dict(entry),
            "sensitive_pixel_recall": class_covered / class_gt if class_gt else 0.0,
            "leakage_rate": 1.0 - class_covered / class_gt if class_gt else 0.0,
        }
    result = {
        "images": images_seen,
        "batches": batches,
        "score_threshold": score_threshold,
        "mask_threshold": mask_threshold,
        "dilation_pixels": dilation_pixels,
        "instance_coverage_threshold": instance_coverage_threshold,
        "gt_sensitive_pixels": gt_pixels,
        "covered_sensitive_pixels": covered_pixels,
        "sensitive_pixel_recall": covered_pixels / gt_pixels if gt_pixels else 0.0,
        "leakage_rate": 1.0 - covered_pixels / gt_pixels if gt_pixels else 0.0,
        "predicted_union_pixels": totals["predicted_union_pixels"],
        "over_redaction_pixels": totals["over_redaction_pixels"],
        "over_redaction_fraction": (
            totals["over_redaction_pixels"] / totals["image_pixels"] if totals["image_pixels"] else 0.0
        ),
        "per_class": result_per_class,
    }
    return result


@torch.inference_mode()
def evaluate_rpn_proposal_recall(
    model: torch.nn.Module,
    data_loader: Iterable,
    device: torch.device,
    *,
    max_batches: int | None = None,
    class_map: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Measure whether the RPN produces a usable proposal for each GT box."""

    if max_batches is not None and max_batches < 1:
        raise ValueError("max_batches must be positive when provided")
    if not hasattr(model, "transform") or not hasattr(model, "rpn") or not hasattr(model, "backbone"):
        raise TypeError("RPN diagnostics require a TorchVision GeneralizedRCNN-style model")

    overall: list[float] = []
    by_size: dict[str, list[float]] = defaultdict(list)
    by_class: dict[str, list[float]] = defaultdict(list)
    proposal_counts: list[int] = []
    batches = 0
    images_seen = 0
    model.eval()
    for images, targets in data_loader:
        if max_batches is not None and batches >= max_batches:
            break
        images_device = [image.to(device, non_blocking=True) for image in images]
        targets_device = [
            {key: value.to(device) if torch.is_tensor(value) else value for key, value in target.items()}
            for target in targets
        ]
        transformed_images, transformed_targets = model.transform(images_device, targets_device)
        features = model.backbone(transformed_images.tensors)
        if isinstance(features, torch.Tensor):
            features = {"0": features}
        proposals, _ = model.rpn(transformed_images, features, None)
        for proposal_boxes, transformed_target in zip(proposals, transformed_targets or []):
            proposal_boxes = proposal_boxes.detach()
            proposal_counts.append(int(proposal_boxes.shape[0]))
            gt_boxes = transformed_target["boxes"].detach()
            gt_labels = transformed_target["labels"].detach()
            best_ious = _box_iou(gt_boxes, proposal_boxes).max(dim=1).values if proposal_boxes.numel() else gt_boxes.new_zeros((gt_boxes.shape[0],))
            for box, label, best_iou in zip(gt_boxes, gt_labels, best_ious):
                score = float(best_iou.item())
                class_name = _name_for_class(int(label.item()), class_map)
                bucket = _size_bucket(box)
                overall.append(score)
                by_size[bucket].append(score)
                by_class[class_name].append(score)
            images_seen += 1
        batches += 1

    if images_seen == 0:
        raise RuntimeError("Evaluation loader produced no images")

    def summarize(values: list[float]) -> dict[str, float | int]:
        tensor = torch.tensor(values, dtype=torch.float32)
        return {
            "gt_instances": len(values),
            "recall_at_iou_0.50": float((tensor >= 0.50).float().mean().item()) if values else 0.0,
            "recall_at_iou_0.75": float((tensor >= 0.75).float().mean().item()) if values else 0.0,
            "mean_best_iou": float(tensor.mean().item()) if values else 0.0,
        }

    return {
        "images": images_seen,
        "batches": batches,
        "mean_proposals_per_image": sum(proposal_counts) / len(proposal_counts) if proposal_counts else 0.0,
        "overall": summarize(overall),
        "by_size": {bucket: summarize(values) for bucket, values in sorted(by_size.items())},
        "per_class": {name: summarize(values) for name, values in sorted(by_class.items())},
    }
