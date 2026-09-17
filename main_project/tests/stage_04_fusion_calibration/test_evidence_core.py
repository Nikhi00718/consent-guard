from __future__ import annotations

from pathlib import Path

import numpy as np

from consentguard.stage_04_fusion_calibration.domain import Evidence, EvidenceGeometry
from consentguard.stage_04_fusion_calibration.evidence import EvidenceFusion, ThresholdRegistry
from consentguard.stage_04_fusion_calibration.evidence.geometry import decode_binary_mask, encode_binary_mask


ROOT = Path(__file__).resolve().parents[3]


def test_binary_mask_rle_round_trip() -> None:
    mask = np.zeros((8, 10), dtype=np.uint8)
    mask[2:6, 3:9] = 1
    assert np.array_equal(decode_binary_mask(encode_binary_mask(mask), 8, 10), mask)


def test_threshold_registry_uses_exact_then_fallback_rules() -> None:
    registry = ThresholdRegistry.load(ROOT / "main_project" / "configs" / "stage_04_fusion_calibration" / "threshold_profile_candidate_v1.yaml")
    assert registry.get("maskrcnn", "a108_license_plate_all").score_threshold == 0.20
    assert registry.get("future_provider", "future_class").experimental is True
    assert registry.profile.release_ready is False


def test_fusion_merges_overlapping_evidence_and_retains_provenance() -> None:
    registry = ThresholdRegistry.load(ROOT / "main_project" / "configs" / "stage_04_fusion_calibration" / "threshold_profile_candidate_v1.yaml")
    evidence = [
        Evidence(
            evidence_id="mask-plate",
            provider="maskrcnn",
            provider_version="baseline-v0.1",
            privacy_class="a108_license_plate_all",
            confidence=0.45,
            geometry=EvidenceGeometry(width=100, height=80, box_xyxy=(20, 20, 50, 40)),
        ),
        Evidence(
            evidence_id="specialist-plate",
            provider="plate_fasterrcnn",
            provider_version="candidate",
            privacy_class="license_plate",
            confidence=0.55,
            geometry=EvidenceGeometry(width=100, height=80, box_xyxy=(40, 25, 70, 45)),
        ),
    ]
    result = EvidenceFusion(registry).combine(evidence, width=100, height=80)
    assert len(result.candidates) == 1
    assert result.candidates[0].evidence_ids == ("mask-plate", "specialist-plate")
    assert result.candidates[0].providers == ("maskrcnn", "plate_fasterrcnn")
    assert result.requires_review is True


def test_fusion_reports_unavailable_provider_and_rejected_low_score() -> None:
    registry = ThresholdRegistry.load(ROOT / "main_project" / "configs" / "stage_04_fusion_calibration" / "threshold_profile_candidate_v1.yaml")
    evidence = Evidence(
        evidence_id="weak-face",
        provider="yunet",
        provider_version="test",
        privacy_class="face",
        confidence=0.1,
        geometry=EvidenceGeometry(width=32, height=32, box_xyxy=(2, 2, 10, 10)),
    )
    result = EvidenceFusion(registry).combine(
        [evidence], width=32, height=32, unavailable_providers=("paddleocr",)
    )
    assert result.candidates == ()
    assert result.rejected_evidence_ids == ("weak-face",)
    assert result.unavailable_providers == ("paddleocr",)
