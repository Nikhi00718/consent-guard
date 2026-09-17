from datetime import datetime, timedelta, timezone

import pytest

from consentguard.stage_04_fusion_calibration.domain import ConsentState
from consentguard.stage_05_review_export.policy import (
    ConsentLedger,
    ConsentRecord,
    ConsentRequest,
    ReleasePolicy,
    digest_resolution,
)
from consentguard.stage_04_fusion_calibration.domain import (
    AssuranceCheck,
    AssuranceReport,
    AssuranceStatus,
    ReleaseAction,
    ReviewCandidateSet,
)


UTC = timezone.utc


def _request() -> ConsentRequest:
    return ConsentRequest("media-1", "context-1", "publish", "public", "portfolio")


def _record(state: ConsentState, *, issued: datetime | None = None, **kwargs) -> ConsentRecord:
    return ConsentRecord.create(
        subject_ref="subject-1",
        bound_region_refs=("candidate-1",),
        request=_request(),
        state=state,
        issued_at=issued or datetime(2026, 1, 1, tzinfo=UTC),
        **kwargs,
    )


def test_consent_scope_resolution_is_fail_closed_and_denial_wins() -> None:
    ledger = ConsentLedger((_record(ConsentState.GRANTED), _record(ConsentState.DENIED)))
    # Separate IDs are required even when two assertions share a scope.
    resolution = ledger.resolve(_request(), at=datetime(2026, 1, 2, tzinfo=UTC))
    assert resolution.state is ConsentState.DENIED
    assert "CONSENT_SCOPE_CONFLICT" in resolution.reason_codes
    assert digest_resolution(resolution) == digest_resolution(resolution)


def test_revocation_and_expiry_are_time_sensitive() -> None:
    issued = datetime(2026, 1, 1, tzinfo=UTC)
    record = _record(
        ConsentState.GRANTED,
        issued=issued,
        expires_at=issued + timedelta(days=1),
    )
    ledger = ConsentLedger((record,))
    assert ledger.resolve(_request(), at=issued + timedelta(hours=1)).state is ConsentState.GRANTED
    assert ledger.resolve(_request(), at=issued + timedelta(days=1)).state is ConsentState.EXPIRED


def test_invalid_lifecycle_transition_is_rejected() -> None:
    with pytest.raises(ValueError, match="invalid consent transition"):
        ConsentLedger.validate_transition(ConsentState.REVOKED, ConsentState.GRANTED)


def test_consent_record_requires_timezone_aware_timestamps() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        _record(ConsentState.GRANTED, issued=datetime(2026, 1, 1))


def test_policy_can_explicitly_allow_unchanged_pixels_with_granted_scope() -> None:
    candidates = ReviewCandidateSet(8, 8, (), "release", True)
    assurance = AssuranceReport((AssuranceCheck("all", AssuranceStatus.PASS, "OK"),))
    decision = ReleasePolicy().decide(
        candidates,
        ConsentState.GRANTED,
        assurance,
        review_completed=True,
        redaction_applied=False,
        allow_unchanged=True,
    )
    assert decision.action is ReleaseAction.ALLOW_PIXELS_UNCHANGED
    assert decision.export_allowed
    assert len(decision.decision_digest) == 64


def test_policy_does_not_allow_unchanged_pixels_by_default() -> None:
    candidates = ReviewCandidateSet(8, 8, (), "release", True)
    assurance = AssuranceReport((AssuranceCheck("all", AssuranceStatus.PASS, "OK"),))
    decision = ReleasePolicy().decide(
        candidates,
        ConsentState.GRANTED,
        assurance,
        review_completed=True,
        redaction_applied=False,
    )
    assert decision.action is ReleaseAction.HOLD_FOR_REVIEW
