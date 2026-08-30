from __future__ import annotations

import sys
from pathlib import Path


SCRIPT_ROOT = Path(__file__).resolve().parents[2] / "scripts" / "stage_03_specialists"
sys.path.insert(0, str(SCRIPT_ROOT))

import evaluate_plate_detection_challenge as module  # noqa: E402


def test_iou_and_greedy_matching_count_duplicates_as_false_positives() -> None:
    truth = [(10.0, 10.0, 30.0, 30.0)]
    predictions = [
        (0.9, (10.0, 10.0, 30.0, 30.0)),
        (0.8, (11.0, 11.0, 29.0, 29.0)),
        (0.4, (50.0, 50.0, 60.0, 60.0)),
    ]

    assert module._iou(truth[0], truth[0]) == 1.0
    assert module._match(predictions, truth, score_threshold=0.5, iou_threshold=0.5) == (1, 1, 0)
    assert module._match(predictions, truth, score_threshold=0.95, iou_threshold=0.5) == (0, 0, 1)


def test_metrics_handle_empty_predictions() -> None:
    metrics = module._metrics(0, 0, 3, images=2)
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0
    assert metrics["false_positives_per_image"] == 0.0
