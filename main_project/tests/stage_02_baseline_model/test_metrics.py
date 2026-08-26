from __future__ import annotations

import pytest
import torch

from consentguard.stage_02_baseline_model.metrics import evaluate_instance_segmentation


class PerfectPredictionModel(torch.nn.Module):
    def forward(self, images):
        outputs = []
        for image in images:
            height, width = image.shape[-2:]
            mask = torch.zeros((1, 1, height, width), device=image.device)
            mask[:, :, 2:10, 3:12] = 1.0
            outputs.append(
                {
                    "boxes": torch.tensor([[3.0, 2.0, 12.0, 10.0]], device=image.device),
                    "scores": torch.tensor([0.99], device=image.device),
                    "labels": torch.tensor([1], device=image.device),
                    "masks": mask,
                }
            )
        return outputs


class PerfectBoxPredictionModel(torch.nn.Module):
    def forward(self, images):
        return [
            {
                "boxes": torch.tensor([[3.0, 2.0, 12.0, 10.0]], device=image.device),
                "scores": torch.tensor([0.99], device=image.device),
                "labels": torch.tensor([1], device=image.device),
            }
            for image in images
        ]


def test_joint_bbox_and_mask_map_uses_segmentation_as_primary() -> None:
    image = torch.zeros((3, 16, 16))
    mask = torch.zeros((1, 16, 16), dtype=torch.uint8)
    mask[:, 2:10, 3:12] = 1
    target = {
        "boxes": torch.tensor([[3.0, 2.0, 12.0, 10.0]]),
        "labels": torch.tensor([1]),
        "masks": mask,
        "area": torch.tensor([72.0]),
        "iscrowd": torch.tensor([0]),
    }
    metrics = evaluate_instance_segmentation(
        PerfectPredictionModel(),
        [([image], [target])],
        torch.device("cpu"),
        class_metrics=True,
        class_map={"background": 0, "private": 1},
    )
    assert metrics["primary_metric"] == "segm_map"
    assert metrics["bbox_map"] == pytest.approx(1.0)
    assert metrics["segm_map"] == pytest.approx(1.0)
    assert metrics["per_class"]["private"]["map"] == pytest.approx(1.0)


def test_box_detector_uses_bbox_map_as_primary_without_fake_masks() -> None:
    image = torch.zeros((3, 16, 16))
    mask = torch.zeros((1, 16, 16), dtype=torch.uint8)
    mask[:, 2:10, 3:12] = 1
    target = {
        "boxes": torch.tensor([[3.0, 2.0, 12.0, 10.0]]),
        "labels": torch.tensor([1]),
        "masks": mask,
        "area": torch.tensor([72.0]),
        "iscrowd": torch.tensor([0]),
    }
    metrics = evaluate_instance_segmentation(
        PerfectBoxPredictionModel(),
        [([image], [target])],
        torch.device("cpu"),
        class_metrics=True,
        class_map={"background": 0, "private": 1},
    )
    assert metrics["primary_metric"] == "bbox_map"
    assert metrics["bbox_map"] == pytest.approx(1.0)
    assert "segm_map" not in metrics
    assert metrics["mask_storage"] == "not_applicable_box_detector"
