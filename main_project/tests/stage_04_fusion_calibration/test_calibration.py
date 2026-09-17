import numpy as np

from consentguard.stage_04_fusion_calibration.calibration import match_class_masks


def test_match_class_masks_prefers_high_score_and_reports_false_positives() -> None:
    ground_truth = [np.array([[1, 0], [0, 0]], dtype=bool)]
    predictions = [
        np.array([[1, 0], [0, 0]], dtype=bool),
        np.array([[0, 1], [0, 0]], dtype=bool),
    ]
    result = match_class_masks(predictions, [0.9, 0.8], ground_truth, score_threshold=0.5)
    assert (result.true_positive, result.false_positive, result.false_negative) == (1, 1, 0)
    assert result.precision == 0.5
    assert result.recall == 1.0
