"""Typed, scope-bound consent records for deterministic policy decisions.

ConsentGuard never infers consent from pixels.  A record is an external,
explicit assertion bound to an exact media version and share context.  This
module only resolves those assertions; it does not call perception models.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import uuid4

from consentguard.stage_04_fusion_calibration.domain import ConsentState


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamps must be timezone-aware")
    return value.astimezone(timezone.utc)


@dataclass(frozen=True)
class ConsentRequest:
    media_version_digest: str
    share_context_digest: str
    operation: str
    audience: str
    purpose: str

    def __post_init__(self) -> None:
        if not all(
            isinstance(value, str) and value.strip()
            for value in (
                self.media_version_digest,
                self.share_context_digest,
                self.operation,
                self.audience,
                self.purpose,
            )
        ):
            raise ValueError("consent scope fields must be non-empty strings")


@dataclass(frozen=True)
class ConsentRecord:
    record_id: str
    subject_ref: str
    bound_region_refs: tuple[str, ...]
    request: ConsentRequest
    state: ConsentState
    issued_at: datetime
    expires_at: datetime | None = None
    revoked_at: datetime | None = None
    assertion_source: str = "user"
    assurance_level: str = "explicit"
    policy_version: str = "consent-schema-v1"
    notes: str | None = None

    def __post_init__(self) -> None:
        if not self.record_id or not self.subject_ref:
            raise ValueError("record_id and subject_ref must be non-empty")
        if not self.bound_region_refs or any(not ref for ref in self.bound_region_refs):
            raise ValueError("at least one bound region reference is required")
        issued = _utc(self.issued_at)
        expiry = _utc(self.expires_at) if self.expires_at is not None else None
        revoked = _utc(self.revoked_at) if self.revoked_at is not None else None
        if expiry is not None and expiry <= issued:
            raise ValueError("expires_at must be after issued_at")
        if self.state is ConsentState.REVOKED and revoked is None:
            raise ValueError("revoked consent requires revoked_at")
        if self.state is not ConsentState.REVOKED and revoked is not None:
            raise ValueError("revoked_at is only valid for REVOKED consent")
        if not self.assertion_source or not self.assurance_level:
            raise ValueError("assertion_source and assurance_level must be non-empty")

    @classmethod
    def create(
        cls,
        *,
        subject_ref: str,
        bound_region_refs: tuple[str, ...],
        request: ConsentRequest,
        state: ConsentState,
        issued_at: datetime,
        expires_at: datetime | None = None,
        revoked_at: datetime | None = None,
        assertion_source: str = "user",
        assurance_level: str = "explicit",
        policy_version: str = "consent-schema-v1",
        notes: str | None = None,
    ) -> "ConsentRecord":
        return cls(
            record_id=f"consent-{uuid4().hex}",
            subject_ref=subject_ref,
            bound_region_refs=tuple(sorted(set(bound_region_refs))),
            request=request,
            state=state,
            issued_at=issued_at,
            expires_at=expires_at,
            revoked_at=revoked_at,
            assertion_source=assertion_source,
            assurance_level=assurance_level,
            policy_version=policy_version,
            notes=notes,
        )

    def effective_state(self, at: datetime) -> ConsentState:
        moment = _utc(at)
        if self.state is ConsentState.GRANTED and self.expires_at is not None:
            if moment >= _utc(self.expires_at):
                return ConsentState.EXPIRED
        return self.state

    def matches(self, request: ConsentRequest) -> bool:
        return self.request == request

    def canonical_dict(self) -> dict[str, object]:
        return {
            "record_id": self.record_id,
            "subject_ref": self.subject_ref,
            "bound_region_refs": list(self.bound_region_refs),
            "media_version_digest": self.request.media_version_digest,
            "share_context_digest": self.request.share_context_digest,
            "operation": self.request.operation,
            "audience": self.request.audience,
            "purpose": self.request.purpose,
            "state": self.state.value,
            "issued_at": _utc(self.issued_at).isoformat(),
            "expires_at": _utc(self.expires_at).isoformat() if self.expires_at else None,
            "revoked_at": _utc(self.revoked_at).isoformat() if self.revoked_at else None,
            "assertion_source": self.assertion_source,
            "assurance_level": self.assurance_level,
            "policy_version": self.policy_version,
        }


@dataclass(frozen=True)
class ConsentResolution:
    state: ConsentState
    matched_record_ids: tuple[str, ...]
    reason_codes: tuple[str, ...]
    conflict: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "state": self.state.value,
            "matched_record_ids": list(self.matched_record_ids),
            "reason_codes": list(self.reason_codes),
            "conflict": self.conflict,
        }


class ConsentLedger:
    """Resolve exact-scope records with conservative precedence."""

    _transitions = {
        ConsentState.UNKNOWN: {ConsentState.UNKNOWN, ConsentState.PENDING, ConsentState.GRANTED, ConsentState.DENIED},
        ConsentState.PENDING: {ConsentState.PENDING, ConsentState.GRANTED, ConsentState.DENIED, ConsentState.EXPIRED},
        ConsentState.GRANTED: {ConsentState.GRANTED, ConsentState.REVOKED, ConsentState.EXPIRED},
        ConsentState.DENIED: {ConsentState.DENIED},
        ConsentState.REVOKED: {ConsentState.REVOKED},
        ConsentState.EXPIRED: {ConsentState.EXPIRED},
    }

    def __init__(self, records: tuple[ConsentRecord, ...] = ()) -> None:
        self._records: dict[str, ConsentRecord] = {}
        for record in records:
            self.add(record)

    def add(self, record: ConsentRecord) -> None:
        if record.record_id in self._records:
            raise ValueError(f"duplicate consent record: {record.record_id}")
        self._records[record.record_id] = record

    @classmethod
    def validate_transition(cls, before: ConsentState, after: ConsentState) -> None:
        if after not in cls._transitions[before]:
            raise ValueError(f"invalid consent transition: {before.value} -> {after.value}")

    def resolve(self, request: ConsentRequest, *, at: datetime) -> ConsentResolution:
        matched = [record for record in self._records.values() if record.matches(request)]
        matched.sort(key=lambda record: (record.issued_at, record.record_id), reverse=True)
        if not matched:
            return ConsentResolution(ConsentState.UNKNOWN, (), ("CONSENT_NO_MATCH",))
        states = {record.effective_state(at) for record in matched}
        ids = tuple(record.record_id for record in matched)
        if ConsentState.REVOKED in states:
            state, reason = ConsentState.REVOKED, "CONSENT_REVOKED"
        elif ConsentState.DENIED in states:
            state, reason = ConsentState.DENIED, "CONSENT_DENIED"
        elif ConsentState.EXPIRED in states:
            state, reason = ConsentState.EXPIRED, "CONSENT_EXPIRED"
        elif ConsentState.PENDING in states:
            state, reason = ConsentState.PENDING, "CONSENT_PENDING"
        elif ConsentState.GRANTED in states:
            state, reason = ConsentState.GRANTED, "CONSENT_GRANTED"
        else:
            state, reason = ConsentState.UNKNOWN, "CONSENT_UNKNOWN"
        conflict = len(states) > 1
        reasons = [reason]
        if conflict:
            reasons.append("CONSENT_SCOPE_CONFLICT")
        return ConsentResolution(state, ids, tuple(sorted(set(reasons))), conflict)


def digest_resolution(resolution: ConsentResolution) -> str:
    payload = json.dumps(resolution.to_dict(), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()
