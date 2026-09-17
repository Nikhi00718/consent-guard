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


RESEARCH_MODE = "research"
PERSONAL_MODE = "personal"

#: Assurance checks that describe the exported file itself. A failure here means
#: the written file is unreadable, mis-hashed, or still carries metadata, so it
#: blocks export in every mode. Residual-content attack findings are advisory in
#: personal mode because the reviewer may have deliberately kept a region.
INTEGRITY_CHECKS = ("render", "pixel_decode", "output_hash", "metadata")


class ReleasePolicy:
    version = "release-policy-v1"

    def __init__(self, mode: str = RESEARCH_MODE) -> None:
        if mode not in {RESEARCH_MODE, PERSONAL_MODE}:
            raise ValueError(f"Unknown release policy mode: {mode!r}")
        self.mode = mode
        self.version = "release-policy-v1" if mode == RESEARCH_MODE else "release-policy-personal-v1"

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
        if self.mode == PERSONAL_MODE:
            return self._decide_personal(
                candidates,
                consent_state,
                assurance,
                review_completed=review_completed,
                redaction_applied=redaction_applied,
                allow_unchanged=allow_unchanged,
            )
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

    def _decide_personal(
        self,
        candidates: ReviewCandidateSet,
        consent_state: ConsentState,
        assurance: AssuranceReport,
        *,
        review_completed: bool,
        redaction_applied: bool,
        allow_unchanged: bool,
    ) -> ReleaseDecision:
        """Single-user mode: the person at the keyboard is the release gate.

        Research mode withholds export until every calibration and assurance
        gate passes, which is correct for a published claim and unusable for a
        personal tool. Personal mode blocks only on an explicit refusal or a
        broken output file, and surfaces everything else as a warning reason
        code the interface must display.
        """

        if consent_state in {ConsentState.DENIED, ConsentState.REVOKED}:
            return self._decision(ReleaseAction.REJECT_EXPORT, [f"CONSENT_{consent_state.value}"])

        failed_integrity = sorted(
            check.name
            for check in assurance.checks
            if check.status is AssuranceStatus.FAIL and check.name in INTEGRITY_CHECKS
        )
        if failed_integrity:
            return self._decision(
                ReleaseAction.REJECT_EXPORT,
                [f"OUTPUT_INTEGRITY_FAILED_{name.upper()}" for name in failed_integrity],
            )
        if not review_completed:
            return self._decision(ReleaseAction.HOLD_FOR_REVIEW, ["AWAITING_USER_ACCEPTANCE"])

        warnings: list[str] = []
        for check in assurance.checks:
            if not check.name.startswith("attack_"):
                continue
            category = check.name[len("attack_") :].upper()
            if check.status is AssuranceStatus.FAIL:
                warnings.append(f"WARNING_RESIDUAL_{category}_DETECTED")
            elif check.status in {AssuranceStatus.UNCERTAIN, AssuranceStatus.NOT_RUN}:
                warnings.append(f"WARNING_{category}_NOT_VERIFIED")
        if not candidates.threshold_profile_release_ready:
            warnings.append("WARNING_EXPERIMENTAL_DETECTION_PROFILE")
        warnings.extend(
            f"WARNING_PROVIDER_UNAVAILABLE_{provider.upper()}"
            for provider in candidates.unavailable_providers
        )
        if not redaction_applied:
            if candidates.candidates:
                warnings.append("WARNING_DETECTED_REGIONS_LEFT_VISIBLE")
            else:
                warnings.append("NO_REGIONS_DETECTED")
            warnings.append("METADATA_STRIPPED_ONLY")
            return self._decision(ReleaseAction.ALLOW_PIXELS_UNCHANGED, warnings)
        return self._decision(ReleaseAction.ALLOW_REDACTED, ["USER_APPROVED_REDACTION", *warnings])

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
