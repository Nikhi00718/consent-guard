"""Typed public contracts for the local reviewer API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ProviderOption(BaseModel):
    key: str
    label: str
    available: bool


class AppConfigResponse(BaseModel):
    providers: list[ProviderOption]
    privacy_groups: list[str]
    upload_max_bytes: int
    upload_max_pixels: int
    session_ttl_seconds: int


class SessionResponse(BaseModel):
    session_id: str
    created_at: float
    expires_at: float


class AssetResponse(BaseModel):
    width: int
    height: int
    source_format: str
    source_sha256: str
    pixel_sha256: str
    metadata_categories: list[str]
    orientation_applied: bool
    normalized_url: str


class AnalyzeRequest(BaseModel):
    provider_keys: list[str] | None = None
    privacy_groups: list[str] | None = None


class CandidateSummary(BaseModel):
    candidate_id: str
    privacy_classes: list[str]
    providers: list[str]
    uncertainty_flags: list[str]
    mandatory_review: bool
    mask_pixels: int


class AnalysisResponse(BaseModel):
    width: int
    height: int
    raw_evidence_count: int
    candidates: list[CandidateSummary]
    selected_provider_keys: list[str]
    selected_privacy_groups: list[str]
    unavailable_providers: list[str]
    provider_errors: dict[str, str]
    threshold_profile_id: str
    threshold_profile_release_ready: bool
    normalized_url: str
    initial_mask_url: str
    mask_overlay_url: str
    overlay_url: str


class AssuranceCheckResponse(BaseModel):
    name: str
    status: str
    reason_code: str
    details: dict[str, object] = Field(default_factory=dict)


class ReleaseDecisionResponse(BaseModel):
    action: str
    reason_codes: list[str]
    policy_version: str
    review_required: bool
    export_allowed: bool
    decision_digest: str


class RenderResponse(BaseModel):
    assurance_status: str
    assurance_checks: list[AssuranceCheckResponse]
    decision: ReleaseDecisionResponse
    export_report: dict[str, object]
    rendered_url: str
    export_available: bool


class HealthResponse(BaseModel):
    status: str
    configured_providers: int
