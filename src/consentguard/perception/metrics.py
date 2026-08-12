"""Memory-bounded COCO bounding-box and instance-mask evaluation.

Dense masks are converted to COCO's compressed run-length encoding as each
batch is processed.  Only compressed annotations remain resident for the final
official pycocotools evaluation, which keeps full-validation runs practical.
"""

from __future__ import annotations

import contextlib
import io
from typing import Any, Iterable

import numpy as np
import torch


COCO_STAT_NAMES = (
    "map",
    "map_50",
    "map_75",
    "map_small",
    "map_medium",
    "map_large",
    "mar_1",
    "mar_10",
    "mar_100",
    "mar_small",
    "mar_medium",
    "mar_large",
)


def _compressed_rle(mask_utils: Any, mask: np.ndarray) -> dict[str, Any]:
    encoded = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8, copy=False)))
    counts = encoded["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")
    return {"size": [int(value) for value in encoded["size"]], "counts": counts}


def _xyxy_to_xywh(box: np.ndarray) -> list[float]:
    left, top, right, bottom = (float(value) for value in box)
    return [left, top, max(0.0, right - left), max(0.0, bottom - top)]


def _create_coco(COCO: Any, dataset: dict[str, Any]) -> Any:
    coco = COCO()
    coco.dataset = dataset
    with contextlib.redirect_stdout(io.StringIO()):
        coco.createIndex()
    return coco


def _run_coco_eval(
    COCO: Any,
    COCOeval: Any,
    ground_truth_dataset: dict[str, Any],
    detections: list[dict[str, Any]],
    iou_type: str,
) -> tuple[dict[str, float], Any]:
    coco_gt = _create_coco(COCO, ground_truth_dataset)
    if detections:
        with contextlib.redirect_stdout(io.StringIO()):
            coco_dt = coco_gt.loadRes(detections)
    else:
        coco_dt = _create_coco(
            COCO,
            {
                "images": ground_truth_dataset["images"],
                "categories": ground_truth_dataset["categories"],
                "annotations": [],
            },
        )
    evaluator = COCOeval(coco_gt, coco_dt, iouType=iou_type)
    with contextlib.redirect_stdout(io.StringIO()):
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    return {
        f"{iou_type}_{name}": float(evaluator.stats[index])
        for index, name in enumerate(COCO_STAT_NAMES)
    }, evaluator


def _mean_valid(values: np.ndarray) -> float:
    valid = values[values > -1]
    return float(valid.mean()) if valid.size else -1.0


def _segmentation_per_class(
    evaluator: Any,
    class_map: dict[str, int] | None,
) -> tuple[list[int], dict[str, dict[str, float | int]]]:
    category_ids = [int(value) for value in evaluator.params.catIds]
    precision = evaluator.eval["precision"]  # IoU, recall, class, area, max detections
    recall = evaluator.eval["recall"]  # IoU, class, area, max detections
    id_to_name = {int(value): str(key) for key, value in (class_map or {}).items()}
    per_class: dict[str, dict[str, float | int]] = {}
    for class_index, class_id in enumerate(category_ids):
        name = id_to_name.get(class_id, str(class_id))
        per_class[name] = {
            "class_id": class_id,
            "map": _mean_valid(precision[:, :, class_index, 0, -1]),
            "mar_100": _mean_valid(recall[:, class_index, 0, -1]),
        }
    return category_ids, per_class


@torch.inference_mode()
def evaluate_instance_segmentation(
    model: torch.nn.Module,
    data_loader: Iterable,
    device: torch.device,
    *,
    score_threshold: float = 0.0,
    class_metrics: bool = True,
    max_batches: int | None = None,
    class_map: dict[str, int] | None = None,
    max_detections_per_image: int = 100,
) -> dict[str, Any]:
    """Evaluate with the official COCO API while retaining only compressed masks."""

    if not 0.0 <= score_threshold <= 1.0:
        raise ValueError("score_threshold must be in [0, 1]")
    if max_batches is not None and max_batches < 1:
        raise ValueError("max_batches must be positive when provided")
    if max_detections_per_image < 1:
        raise ValueError("max_detections_per_image must be positive")
    try:
        from pycocotools import mask as mask_utils
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "COCO metrics are unavailable. Install requirements/base.txt, including pycocotools."
        ) from error

    if class_map:
        categories = [
            {"id": int(class_id), "name": str(name)}
            for name, class_id in sorted(class_map.items(), key=lambda item: int(item[1]))
            if int(class_id) != 0
        ]
    else:
        categories = []

    images_coco: list[dict[str, int]] = []
    annotations_coco: list[dict[str, Any]] = []
    bbox_detections_coco: list[dict[str, Any]] = []
    segmentation_detections_coco: list[dict[str, Any]] = []
    observed_categories: set[int] = set()
    annotation_id = 1
    batches = 0
    images_seen = 0
    used_image_ids: set[int] = set()
    model.eval()

    for images, targets in data_loader:
        if max_batches is not None and batches >= max_batches:
            break
        images_device = [image.to(device, non_blocking=True) for image in images]
        predictions = model(images_device)
        if len(predictions) != len(targets):
            raise RuntimeError("Model prediction count does not match target count")

        for prediction, target in zip(predictions, targets):
            if "image_id" in target:
                image_id = int(target["image_id"].detach().cpu().reshape(-1)[0].item())
            else:
                image_id = images_seen
            if image_id in used_image_ids:
                raise RuntimeError(f"Duplicate image_id encountered during evaluation: {image_id}")
            used_image_ids.add(image_id)

            target_masks = target["masks"].detach().cpu().numpy().astype(np.uint8, copy=False)
            target_boxes = target["boxes"].detach().cpu().numpy()
            target_labels = target["labels"].detach().cpu().numpy()
            target_crowds = target["iscrowd"].detach().cpu().numpy()
            if target_masks.ndim != 3:
                raise RuntimeError(f"Expected target masks [N,H,W], received {target_masks.shape}")
            height, width = target_masks.shape[-2:]
            images_coco.append({"id": image_id, "height": int(height), "width": int(width)})

            for mask, box, label, iscrowd in zip(
                target_masks,
                target_boxes,
                target_labels,
                target_crowds,
            ):
                class_id = int(label)
                observed_categories.add(class_id)
                rle = _compressed_rle(mask_utils, mask)
                annotations_coco.append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": class_id,
                        "bbox": _xyxy_to_xywh(box),
                        "segmentation": rle,
                        "area": float(mask_utils.area(rle)),
                        "iscrowd": int(iscrowd),
                    }
                )
                annotation_id += 1

            scores = prediction["scores"].detach()
            keep = torch.nonzero(scores >= score_threshold, as_tuple=False).flatten()
            # TorchVision orders detections by confidence. COCO's standard
            # summary uses at most 100 detections per image.
            keep = keep[:max_detections_per_image]
            for prediction_index in keep.tolist():
                class_id = int(prediction["labels"][prediction_index].item())
                if class_id == 0:
                    continue
                observed_categories.add(class_id)
                mask = (
                    prediction["masks"][prediction_index, 0].detach().cpu().numpy()
                    >= 0.5
                )
                rle = _compressed_rle(mask_utils, mask)
                score = float(scores[prediction_index].item())
                bbox_detections_coco.append(
                    {
                        "image_id": image_id,
                        "category_id": class_id,
                        "bbox": _xyxy_to_xywh(
                            prediction["boxes"][prediction_index].detach().cpu().numpy()
                        ),
                        "score": score,
                    }
                )
                # Keep segmentation detections separate from bounding-box
                # detections. COCO loadRes otherwise assigns bbox area to a
                # joint result, which makes segmentation area buckets wrong.
                segmentation_detections_coco.append(
                    {
                        "image_id": image_id,
                        "category_id": class_id,
                        "segmentation": rle,
                        "score": score,
                    }
                )
            images_seen += 1
        batches += 1

    if images_seen == 0:
        raise RuntimeError("Evaluation loader produced no images")
    if not categories:
        categories = [
            {"id": class_id, "name": str(class_id)}
            for class_id in sorted(observed_categories)
            if class_id != 0
        ]
    category_ids = {int(category["id"]) for category in categories}
    unexpected = observed_categories - category_ids - {0}
    if unexpected:
        raise RuntimeError(f"Predictions or targets contain unknown class IDs: {sorted(unexpected)}")

    ground_truth_dataset = {
        "images": images_coco,
        "annotations": annotations_coco,
        "categories": categories,
    }
    bbox_metrics, _ = _run_coco_eval(
        COCO,
        COCOeval,
        ground_truth_dataset,
        bbox_detections_coco,
        "bbox",
    )
    segmentation_metrics, segmentation_evaluator = _run_coco_eval(
        COCO,
        COCOeval,
        ground_truth_dataset,
        segmentation_detections_coco,
        "segm",
    )
    result: dict[str, Any] = {
        **bbox_metrics,
        **segmentation_metrics,
        "primary_map": segmentation_metrics["segm_map"],
        "primary_metric": "segm_map",
        "evaluated_images": images_seen,
        "evaluated_batches": batches,
        "ground_truth_instances": len(annotations_coco),
        "retained_predictions": len(bbox_detections_coco),
        "score_threshold": score_threshold,
        "max_detections_per_image": max_detections_per_image,
        "mask_storage": "coco_compressed_rle",
    }
    if class_metrics:
        classes, per_class = _segmentation_per_class(segmentation_evaluator, class_map)
        result["classes"] = classes
        result["per_class"] = per_class
    return result
