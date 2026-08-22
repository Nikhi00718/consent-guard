"""Human-readable, framework-independent contracts for ConsentGuard.

The perception modules produce evidence.  These models deliberately do not
contain identity recognition or consent inference fields.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class PrivacyClass(str, Enum):
    FACE = "face"
    LICENSE_PLATE = "license_plate"
    PERSON_BODY = "person_body"
    NUDITY = "nudity"
    HANDWRITING = "handwriting"
    PRINTED_TEXT = "printed_text"
    DISABILITY = "disability"
    MEDICINE = "medicine"
    FINGERPRINT = "fingerprint"
    SIGNATURE = "signature"
    BARCODE = "barcode"
    METADATA = "metadata"


class ConsentState(str, Enum):
    UNKNOWN = "UNKNOWN"
    PENDING = "PENDING"
    GRANTED = "GRANTED"
    DENIED = "DENIED"
    REVOKED = "REVOKED"
    EXPIRED = "EXPIRED"


class ReleaseAction(str, Enum):
    ALLOW_PIXELS_UNCHANGED = "ALLOW_PIXELS_UNCHANGED"
    ALLOW_REDACTED = "ALLOW_REDACTED"
    HOLD_FOR_REVIEW = "HOLD_FOR_REVIEW"
    HOLD_FOR_CONSENT = "HOLD_FOR_CONSENT"
    REJECT_EXPORT = "REJECT_EXPORT"


class AssuranceStatus(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    UNCERTAIN = "UNCERTAIN"
    NOT_RUN = "NOT_RUN"


@dataclass(frozen=True)
class EvidenceGeometry:
    """Original-image geometry.

    A provider supplies at least one of `box_xyxy`, `polygon_xy`, or
    `mask_rle`.  `mask_rle` uses row-major alternating zero/one run lengths.
    """

    width: int
    height: int
    box_xyxy: tuple[float, float, float, float] | None = None
    polygon_xy: tuple[tuple[float, float], ...] = ()
    mask_rle: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.width < 1 or self.height < 1:
            raise ValueError("Geometry width and height must be positive")
        if not self.box_xyxy and not self.polygon_xy and not self.mask_rle:
            raise ValueError("Evidence geometry requires a box, polygon, or mask RLE")
        if self.box_xyxy is not None:
            left, top, right, bottom = self.box_xyxy
            if not (0 <= left < right <= self.width and 0 <= top < bottom <= self.height):
                raise ValueError("box_xyxy must be positive-area and inside the image")
        if self.polygon_xy and len(self.polygon_xy) < 3:
            raise ValueError("polygon_xy must contain at least three points")
        if self.mask_rle and sum(self.mask_rle) != self.width * self.height:
            raise ValueError("mask_rle length does not match image dimensions")


@dataclass(frozen=True)
class Evidence:
    evidence_id: str
    provider: str
    provider_version: str
    privacy_class: str
    geometry: EvidenceGeometry
    confidence: float
    uncertainty_flags: tuple[str, ...] = ()
    source_detection_id: str | None = None
    sensitivity_tier: str = "standard"

    def __post_init__(self) -> None:
        if not self.evidence_id or not self.provider or not self.provider_version:
            raise ValueError("Evidence IDs and provider fields must be non-empty")
        if not self.privacy_class:
            raise ValueError("privacy_class must be non-empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ThresholdRule:
    provider: str
    privacy_class: str
    score_threshold: float
    min_area_pixels: int = 1
    expansion_fraction: float = 0.0
    dilation_pixels: int = 0
    mandatory_review: bool = True
    experimental: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.score_threshold <= 1.0:
            raise ValueError("score_threshold must be in [0, 1]")
        if self.min_area_pixels < 1:
            raise ValueError("min_area_pixels must be positive")
        if not 0.0 <= self.expansion_fraction <= 1.0:
            raise ValueError("expansion_fraction must be in [0, 1]")
        if self.dilation_pixels < 0:
            raise ValueError("dilation_pixels must be non-negative")


@dataclass(frozen=True)
class ReviewCandidate:
    candidate_id: str
    width: int
    height: int
    mask_rle: tuple[int, ...]
    evidence_ids: tuple[str, ...]
    privacy_classes: tuple[str, ...]
    providers: tuple[str, ...]
    mandatory_review: bool
    uncertainty_flags: tuple[str, ...] = ()
    user_correction_state: str = "UNREVIEWED"


@dataclass(frozen=True)
class ReviewCandidateSet:
    width: int
    height: int
    candidates: tuple[ReviewCandidate, ...]
    threshold_profile_id: str
    threshold_profile_release_ready: bool
    unavailable_providers: tuple[str, ...] = ()
    rejected_evidence_ids: tuple[str, ...] = ()

    @property
    def requires_review(self) -> bool:
        return (
            not self.threshold_profile_release_ready
            or bool(self.unavailable_providers)
            or any(candidate.mandatory_review for candidate in self.candidates)
        )


@dataclass(frozen=True)
class AssuranceCheck:
    name: str
    status: AssuranceStatus
    reason_code: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AssuranceReport:
    checks: tuple[AssuranceCheck, ...]

    @property
    def status(self) -> AssuranceStatus:
        statuses = {check.status for check in self.checks}
        if AssuranceStatus.FAIL in statuses:
            return AssuranceStatus.FAIL
        if AssuranceStatus.UNCERTAIN in statuses:
            return AssuranceStatus.UNCERTAIN
        if AssuranceStatus.NOT_RUN in statuses or not statuses:
            return AssuranceStatus.NOT_RUN
        return AssuranceStatus.PASS


@dataclass(frozen=True)
class ReleaseDecision:
    action: ReleaseAction
    reason_codes: tuple[str, ...]
    policy_version: str
    review_required: bool
    export_allowed: bool
    decision_digest: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action.value,
            "reason_codes": list(self.reason_codes),
            "policy_version": self.policy_version,
            "review_required": self.review_required,
            "export_allowed": self.export_allowed,
            "decision_digest": self.decision_digest,
        }
