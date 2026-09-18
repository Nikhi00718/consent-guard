"""Regression test for the negative-image false-positive denominator.

The evaluator once appended a flag for every image, so a sample of 120 images
holding 7 negatives, all of which were flagged, reported 7/120 = 0.058 instead
of 7/7 = 1.0. The README repeated the diluted figure as a false-alarm rate.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

EVALUATOR = (
    Path(__file__).resolve().parents[3]
    / "main_project"
    / "scripts"
    / "stage_06_evaluation_release"
    / "evaluate_fused_validation.py"
)


def _evaluator_module():
    spec = importlib.util.spec_from_file_location("consentguard_fused_validation_fpr", EVALUATOR)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rate_uses_only_negative_images() -> None:
    evaluator = _evaluator_module()
    assert evaluator.negative_false_positive_rate([1.0] * 7) == pytest.approx(1.0)
    assert evaluator.negative_false_positive_rate([1.0, 0.0, 0.0, 0.0]) == pytest.approx(0.25)


def test_no_negatives_means_unknown_not_zero() -> None:
    assert _evaluator_module().negative_false_positive_rate([]) is None


def test_positive_images_are_not_counted() -> None:
    source = EVALUATOR.read_text(encoding="utf-8")
    assert "if not instances:" in source
    assert "1.0 if not instances and len(fused.candidates) else 0.0" not in source
