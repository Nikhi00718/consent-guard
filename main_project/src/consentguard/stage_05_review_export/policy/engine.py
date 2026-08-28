"""Policy decisions are deterministic and never call perception models."""

from __future__ import annotations

import hashlib
import json

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
        allow_unchanged: bool = False,
    ) -> ReleaseDecision:
        reasons: list[str] = []
        if consent_state in {ConsentState.DENIED, ConsentState.REVOKED}:
            reasons.append(f"CONSENT_{consent_state.value}")
            return self._decision(ReleaseAction.REJECT_EXPORT, reasons)
        if consent_state in {ConsentState.UNKNOWN, ConsentState.PENDING, ConsentState.EXPIRED}:
            reasons.append(f"CONSENT_{consent_state.value}")
            return self._decision(ReleaseAction.HOLD_FOR_CONSENT, reasons)
        if assurance.status is AssuranceStatus.FAIL:
            reasons.append("ASSURANCE_FAILED")
            return self._decision(ReleaseAction.REJECT_EXPORT, reasons)
        if assurance.status is not AssuranceStatus.PASS:
            reasons.append(f"ASSURANCE_{assurance.status.value}")
            if not candidates.threshold_profile_release_ready:
                reasons.append("THRESHOLD_PROFILE_NOT_RELEASE_READY")
            if candidates.unavailable_providers:
                reasons.append("REQUIRED_PROVIDER_UNAVAILABLE")
            if candidates.requires_review and not review_completed:
                reasons.append("MANDATORY_REVIEW_INCOMPLETE")
            return self._decision(ReleaseAction.HOLD_FOR_REVIEW, reasons)
        if not candidates.threshold_profile_release_ready:
            reasons.append("THRESHOLD_PROFILE_NOT_RELEASE_READY")
        if candidates.unavailable_providers:
            reasons.append("REQUIRED_PROVIDER_UNAVAILABLE")
        if candidates.requires_review and not review_completed:
            reasons.append("MANDATORY_REVIEW_INCOMPLETE")
        if candidates.candidates and not redaction_applied:
            reasons.append("CANDIDATES_NOT_REDACTED")
        if reasons:
            return self._decision(ReleaseAction.HOLD_FOR_REVIEW, reasons)
        if allow_unchanged and not candidates.candidates and review_completed and not redaction_applied:
            return self._decision(ReleaseAction.ALLOW_PIXELS_UNCHANGED, ["CONSENTED_PIXELS_UNCHANGED"])
        if not review_completed:
            return self._decision(ReleaseAction.HOLD_FOR_REVIEW, ["REVIEW_INCOMPLETE"])
        if not redaction_applied:
            return self._decision(ReleaseAction.HOLD_FOR_REVIEW, ["REDACTION_NOT_APPLIED"])
        return self._decision(ReleaseAction.ALLOW_REDACTED, ["REVIEWED_REDACTION_ASSURED"])

    def _decision(self, action: ReleaseAction, reasons: list[str]) -> ReleaseDecision:
        normalized_reasons = tuple(sorted(set(reasons)))
        payload = {
            "action": action.value,
            "reason_codes": list(normalized_reasons),
            "policy_version": self.version,
            "review_required": action not in {ReleaseAction.ALLOW_REDACTED, ReleaseAction.ALLOW_PIXELS_UNCHANGED},
            "export_allowed": action in {ReleaseAction.ALLOW_REDACTED, ReleaseAction.ALLOW_PIXELS_UNCHANGED},
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return ReleaseDecision(
            action=action,
            reason_codes=normalized_reasons,
            policy_version=self.version,
            review_required=payload["review_required"],
            export_allowed=payload["export_allowed"],
            decision_digest=digest,
        )
