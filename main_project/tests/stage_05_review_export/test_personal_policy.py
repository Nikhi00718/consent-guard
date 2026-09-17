from __future__ import annotations

import pytest

from consentguard.stage_04_fusion_calibration.domain import (
    AssuranceCheck,
    AssuranceReport,
    AssuranceStatus,
    ConsentState,
    ReleaseAction,
    ReviewCandidate,
    ReviewCandidateSet,
)
from consentguard.stage_05_review_export.policy import PERSONAL_MODE, ReleasePolicy


def _candidate() -> ReviewCandidate:
    return ReviewCandidate(
        candidate_id="candidate-1",
        width=4,
        height=4,
        mask_rle=(0, 16),
        evidence_ids=("face-1",),
        privacy_classes=("face",),
        providers=("maskrcnn",),
        mandatory_review=True,
    )


def _candidate_set(
    *,
    release_ready: bool = False,
    unavailable: tuple[str, ...] = (),
    with_candidate: bool = True,
) -> ReviewCandidateSet:
    return ReviewCandidateSet(
        width=4,
        height=4,
        candidates=(_candidate(),) if with_candidate else (),
        threshold_profile_id="personal-test",
        threshold_profile_release_ready=release_ready,
        unavailable_providers=unavailable,
    )


def _assurance(**statuses: AssuranceStatus) -> AssuranceReport:
    defaults = {
        "pixel_decode": AssuranceStatus.PASS,
        "output_hash": AssuranceStatus.PASS,
        "metadata": AssuranceStatus.PASS,
        "attack_ocr": AssuranceStatus.PASS,
        "attack_barcode": AssuranceStatus.PASS,
        "attack_face": AssuranceStatus.PASS,
        "attack_plate": AssuranceStatus.PASS,
    }
    defaults.update(statuses)
    return AssuranceReport(
        tuple(
            AssuranceCheck(name=name, status=status, reason_code=f"{name.upper()}_{status.value}")
            for name, status in defaults.items()
        )
    )


def _decide(policy: ReleasePolicy, **overrides):
    arguments = {
        "candidates": _candidate_set(),
        "consent_state": ConsentState.UNKNOWN,
        "assurance": _assurance(),
        "review_completed": True,
        "redaction_applied": True,
    }
    arguments.update(overrides)
    return policy.decide(
        arguments["candidates"],
        arguments["consent_state"],
        arguments["assurance"],
        review_completed=arguments["review_completed"],
        redaction_applied=arguments["redaction_applied"],
    )


@pytest.fixture
def policy() -> ReleasePolicy:
    return ReleasePolicy(mode=PERSONAL_MODE)


def test_unknown_consent_and_experimental_profile_no_longer_block_export(policy: ReleasePolicy) -> None:
    """The exact combination that kept the research-mode download button off."""

    decision = _decide(policy)
    assert decision.action is ReleaseAction.ALLOW_REDACTED
    assert decision.export_allowed is True
    assert "WARNING_EXPERIMENTAL_DETECTION_PROFILE" in decision.reason_codes


def test_missing_provider_and_unverified_attacks_downgrade_to_warnings(policy: ReleasePolicy) -> None:
    decision = _decide(
        policy,
        candidates=_candidate_set(unavailable=("zxingcpp",)),
        assurance=_assurance(
            attack_barcode=AssuranceStatus.UNCERTAIN,
            attack_ocr=AssuranceStatus.FAIL,
        ),
    )
    assert decision.export_allowed is True
    assert "WARNING_PROVIDER_UNAVAILABLE_ZXINGCPP" in decision.reason_codes
    assert "WARNING_BARCODE_NOT_VERIFIED" in decision.reason_codes
    assert "WARNING_RESIDUAL_OCR_DETECTED" in decision.reason_codes


@pytest.mark.parametrize("check", ["pixel_decode", "output_hash", "metadata"])
def test_a_broken_or_leaky_output_file_still_blocks_export(policy: ReleasePolicy, check: str) -> None:
    decision = _decide(policy, assurance=_assurance(**{check: AssuranceStatus.FAIL}))
    assert decision.action is ReleaseAction.REJECT_EXPORT
    assert decision.export_allowed is False
    assert f"OUTPUT_INTEGRITY_FAILED_{check.upper()}" in decision.reason_codes


@pytest.mark.parametrize("state", [ConsentState.DENIED, ConsentState.REVOKED])
def test_explicit_refusal_still_wins(policy: ReleasePolicy, state: ConsentState) -> None:
    decision = _decide(policy, consent_state=state)
    assert decision.action is ReleaseAction.REJECT_EXPORT
    assert decision.export_allowed is False


def test_export_waits_until_the_user_accepts(policy: ReleasePolicy) -> None:
    decision = _decide(policy, review_completed=False)
    assert decision.action is ReleaseAction.HOLD_FOR_REVIEW
    assert decision.reason_codes == ("AWAITING_USER_ACCEPTANCE",)


def test_leaving_detected_regions_visible_is_allowed_but_flagged(policy: ReleasePolicy) -> None:
    decision = _decide(policy, redaction_applied=False)
    assert decision.action is ReleaseAction.ALLOW_PIXELS_UNCHANGED
    assert decision.export_allowed is True
    assert "WARNING_DETECTED_REGIONS_LEFT_VISIBLE" in decision.reason_codes
    assert "METADATA_STRIPPED_ONLY" in decision.reason_codes


def test_clean_image_reports_no_detections(policy: ReleasePolicy) -> None:
    decision = _decide(
        policy,
        candidates=_candidate_set(with_candidate=False, release_ready=True),
        redaction_applied=False,
    )
    assert decision.action is ReleaseAction.ALLOW_PIXELS_UNCHANGED
    assert "NO_REGIONS_DETECTED" in decision.reason_codes


def test_research_mode_behaviour_is_unchanged() -> None:
    decision = _decide(ReleasePolicy())
    assert decision.action is ReleaseAction.HOLD_FOR_CONSENT
    assert decision.export_allowed is False


def test_unknown_mode_is_rejected() -> None:
    with pytest.raises(ValueError):
        ReleasePolicy(mode="whatever")
