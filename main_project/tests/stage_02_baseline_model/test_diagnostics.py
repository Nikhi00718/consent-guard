from __future__ import annotations

import torch

from consentguard.stage_02_baseline_model.diagnostics import evaluate_privacy_coverage


class FixedPredictionModel(torch.nn.Module):
    def forward(self, images):
        outputs = []
        for image in images:
            height, width = image.shape[-2:]
            mask = torch.zeros((1, 1, height, width), device=image.device)
            mask[0, 0, 4:12, 5:15] = 1.0
            outputs.append(
                {
                    "scores": torch.tensor([0.99], device=image.device),
                    "labels": torch.tensor([1], device=image.device),
                    "masks": mask,
                    "boxes": torch.tensor([[5.0, 4.0, 15.0, 12.0]], device=image.device),
                }
            )
        return outputs


def test_privacy_coverage_reports_union_recall_and_over_redaction() -> None:
    image = torch.zeros((3, 16, 16))
    target_mask = torch.zeros((1, 16, 16), dtype=torch.uint8)
    target_mask[0, 4:12, 5:15] = 1
    target = {
        "masks": target_mask,
        "labels": torch.tensor([1]),
    }

    metrics = evaluate_privacy_coverage(
        FixedPredictionModel(),
        [([image], [target])],
        torch.device("cpu"),
        class_map={"background": 0, "face": 1},
    )

    assert metrics["sensitive_pixel_recall"] == 1.0
    assert metrics["leakage_rate"] == 0.0
    assert metrics["over_redaction_pixels"] == 0
    assert metrics["per_class"]["face"]["instances_coverage_at_least_threshold"] == 1
