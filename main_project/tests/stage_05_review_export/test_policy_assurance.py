from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from consentguard.stage_05_review_export.assurance import AssuranceService, RenderedAsset
from consentguard.stage_04_fusion_calibration.domain import (
    AssuranceCheck,
    AssuranceReport,
    AssuranceStatus,
    ConsentState,
    ReleaseAction,
    ReviewCandidateSet,
)
from consentguard.stage_05_review_export.policy import ReleasePolicy
from consentguard.stage_05_review_export.redaction.prediction_renderer import write_metadata_free_redaction


def _candidate_set(*, release_ready: bool = True, unavailable: tuple[str, ...] = ()) -> ReviewCandidateSet:
    return ReviewCandidateSet(
        width=16,
        height=16,
        candidates=(),
        threshold_profile_id="test",
        threshold_profile_release_ready=release_ready,
        unavailable_providers=unavailable,
    )


def _passing_assurance() -> AssuranceReport:
    return AssuranceReport(
        (AssuranceCheck("all", AssuranceStatus.PASS, "ALL_REQUIRED_CHECKS_PASSED"),)
    )


def test_policy_allows_only_reviewed_assured_redacted_output() -> None:
    decision = ReleasePolicy().decide(
        _candidate_set(),
        ConsentState.GRANTED,
        _passing_assurance(),
        review_completed=True,
        redaction_applied=True,
    )
    assert decision.action is ReleaseAction.ALLOW_REDACTED
    assert decision.export_allowed is True


def test_policy_fails_closed_for_unknown_consent_and_uncertain_assurance() -> None:
    policy = ReleasePolicy()
    unknown = policy.decide(
        _candidate_set(),
        ConsentState.UNKNOWN,
        _passing_assurance(),
        review_completed=True,
        redaction_applied=True,
    )
    assert unknown.action is ReleaseAction.HOLD_FOR_CONSENT

    uncertain = policy.decide(
        _candidate_set(),
        ConsentState.GRANTED,
        AssuranceReport(()),
        review_completed=True,
        redaction_applied=True,
    )
    assert uncertain.action is ReleaseAction.HOLD_FOR_REVIEW


def test_policy_rejects_denied_or_revoked_consent_even_after_redaction() -> None:
    policy = ReleasePolicy()
    for state in (ConsentState.DENIED, ConsentState.REVOKED):
        decision = policy.decide(
            _candidate_set(),
            state,
            _passing_assurance(),
            review_completed=True,
            redaction_applied=True,
        )
        assert decision.action is ReleaseAction.REJECT_EXPORT
        assert decision.export_allowed is False


def test_assurance_verifies_redaction_but_is_uncertain_without_attackers(tmp_path: Path) -> None:
    source = tmp_path / "source.jpg"
    output = tmp_path / "output.png"
    bgr = np.full((16, 20, 3), 180, dtype=np.uint8)
    assert cv2.imwrite(str(source), bgr)
    image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    report = write_metadata_free_redaction(
        source, output, image, np.ones((16, 20), dtype=np.uint8)
    )
    assurance = AssuranceService().inspect(
        RenderedAsset(output, expected_width=20, expected_height=16, export_report=report)
    )
    assert assurance.status is AssuranceStatus.NOT_RUN
    assert {check.name for check in assurance.checks} >= {
        "pixel_decode",
        "output_hash",
        "metadata",
        "attack_ocr",
    }


def test_assurance_passes_when_all_independent_attackers_pass(tmp_path: Path) -> None:
    source = tmp_path / "source.jpg"
    output = tmp_path / "output.png"
    bgr = np.full((12, 18, 3), 90, dtype=np.uint8)
    assert cv2.imwrite(str(source), bgr)
    image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    report = write_metadata_free_redaction(
        source, output, image, np.ones((12, 18), dtype=np.uint8)
    )
    passed = {name: AssuranceStatus.PASS for name in ("ocr", "barcode", "face", "plate")}
    assurance = AssuranceService().inspect(
        RenderedAsset(output, 18, 12, report, attack_checks=passed)
    )
    assert assurance.status is AssuranceStatus.PASS


def test_attack_runner_marks_findings_failed_and_missing_attacks_uncertain(tmp_path: Path) -> None:
    source = tmp_path / "source.jpg"
    assert cv2.imwrite(str(source), np.full((12, 18, 3), 90, dtype=np.uint8))
    output = tmp_path / "output.png"
    image = cv2.cvtColor(cv2.imread(str(source)), cv2.COLOR_BGR2RGB)
    write_metadata_free_redaction(source, output, image, np.ones((12, 18), dtype=np.uint8))

    class FakeProvider:
        def __init__(self, name: str, findings: list[object]) -> None:
            self.name = name
            self.findings = findings

        def analyze(self, _image):
            return self.findings

    results = AssuranceService().run_attack_checks(
        output,
        (FakeProvider("yunet", []), FakeProvider("zxingcpp", [object()])),
    )
    assert results["face"] is AssuranceStatus.PASS
    assert results["barcode"] is AssuranceStatus.FAIL
    assert results["ocr"] is AssuranceStatus.UNCERTAIN
    assert results["plate"] is AssuranceStatus.UNCERTAIN
