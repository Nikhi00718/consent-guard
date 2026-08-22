"""Policy decisions are deterministic and never call perception models."""

from __future__ import annotations

from consentguard.stage_04_fusion_calibration.domain import (
    AssuranceReport,
    AssuranceStatus,
    ConsentState,
    ReleaseAction,
    ReleaseDecision,
    ReviewCandidateSet,
)


class ReleasePolicy:
    version = "release-policy-v1"

    def decide(
        self,
        candidates: ReviewCandidateSet,
        consent_state: ConsentState,
        assurance: AssuranceReport,
        *,
        review_completed: bool,
        redaction_applied: bool,
    ) -> ReleaseDecision:
        reasons: list[str] = []
        if assurance.status is AssuranceStatus.FAIL:
            reasons.append("ASSURANCE_FAILED")
            return self._decision(ReleaseAction.REJECT_EXPORT, reasons)
        if assurance.status is not AssuranceStatus.PASS:
            reasons.append(f"ASSURANCE_{assurance.status.value}")
            return self._decision(ReleaseAction.HOLD_FOR_REVIEW, reasons)
        if consent_state in {ConsentState.UNKNOWN, ConsentState.PENDING, ConsentState.EXPIRED}:
            reasons.append(f"CONSENT_{consent_state.value}")
            return self._decision(ReleaseAction.HOLD_FOR_CONSENT, reasons)
        if candidates.unavailable_providers:
            reasons.append("REQUIRED_PROVIDER_UNAVAILABLE")
        if candidates.requires_review and not review_completed:
            reasons.append("MANDATORY_REVIEW_INCOMPLETE")
        if candidates.candidates and not redaction_applied:
            reasons.append("CANDIDATES_NOT_REDACTED")
        if consent_state in {ConsentState.DENIED, ConsentState.REVOKED} and not redaction_applied:
            reasons.append(f"CONSENT_{consent_state.value}_REQUIRES_REDACTION")
        if reasons:
            return self._decision(ReleaseAction.HOLD_FOR_REVIEW, reasons)
        if not review_completed:
            return self._decision(ReleaseAction.HOLD_FOR_REVIEW, ["REVIEW_INCOMPLETE"])
        if not redaction_applied:
            return self._decision(ReleaseAction.HOLD_FOR_REVIEW, ["REDACTION_NOT_APPLIED"])
        return self._decision(ReleaseAction.ALLOW_REDACTED, ["REVIEWED_REDACTION_ASSURED"])

    def _decision(self, action: ReleaseAction, reasons: list[str]) -> ReleaseDecision:
        return ReleaseDecision(
            action=action,
            reason_codes=tuple(sorted(set(reasons))),
            policy_version=self.version,
            review_required=action is not ReleaseAction.ALLOW_REDACTED,
            export_allowed=action is ReleaseAction.ALLOW_REDACTED,
        )
