"""Validation-only score-threshold calibration helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MatchCounts:
    true_positive: int
    false_positive: int
    false_negative: int

    @property
    def precision(self) -> float:
        denominator = self.true_positive + self.false_positive
        return self.true_positive / denominator if denominator else 0.0

    @property
    def recall(self) -> float:
        denominator = self.true_positive + self.false_negative
        return self.true_positive / denominator if denominator else 0.0

    @property
    def f2(self) -> float:
        precision, recall = self.precision, self.recall
        denominator = 4.0 * precision + recall
        return 5.0 * precision * recall / denominator if denominator else 0.0


def binary_mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        raise ValueError("mask shapes must match")
    intersection = int(np.logical_and(left, right).sum())
    union = int(np.logical_or(left, right).sum())
    return intersection / union if union else 0.0


def match_class_masks(
    predicted_masks: list[np.ndarray],
    predicted_scores: list[float],
    ground_truth_masks: list[np.ndarray],
    *,
    score_threshold: float,
    iou_threshold: float = 0.5,
) -> MatchCounts:
    """Greedily match score-sorted predictions to same-class GT instances."""

    if not 0.0 <= score_threshold <= 1.0:
        raise ValueError("score_threshold must be in [0, 1]")
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in [0, 1]")
    if len(predicted_masks) != len(predicted_scores):
        raise ValueError("predicted mask and score lengths must match")
    order = sorted(
        (index for index, score in enumerate(predicted_scores) if float(score) >= score_threshold),
        key=lambda index: float(predicted_scores[index]),
        reverse=True,
    )
    unmatched = set(range(len(ground_truth_masks)))
    true_positive = 0
    false_positive = 0
    for index in order:
        best_gt = None
        best_iou = 0.0
        for gt_index in unmatched:
            value = binary_mask_iou(predicted_masks[index], ground_truth_masks[gt_index])
            if value > best_iou:
                best_iou, best_gt = value, gt_index
        if best_gt is not None and best_iou >= iou_threshold:
            unmatched.remove(best_gt)
            true_positive += 1
        else:
            false_positive += 1
    return MatchCounts(true_positive, false_positive, len(unmatched))
