"""Session-scoped reviewer orchestration without exposing local paths."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

from consentguard.stage_04_fusion_calibration.domain import ConsentState
from consentguard.stage_04_fusion_calibration.evidence.geometry import decode_binary_mask
from consentguard.stage_05_review_export.ingest import IngestLimits, NormalizedImage, SessionHandle, SessionStore, normalize_image
from consentguard.stage_05_review_export.pipeline import AnalysisSnapshot, ReviewExportService
from consentguard.stage_05_review_export.policy import ConsentRecord, ConsentRequest

from consentguard.stage_05_review_export.api.models import (
    AnalysisResponse,
    AppConfigResponse,
    AssetResponse,
    AssuranceCheckResponse,
    CandidateSummary,
    ProviderOption,
    ReleaseDecisionResponse,
    RenderResponse,
    SessionResponse,
)


_COLORS = (
    (48, 183, 166),
    (222, 126, 96),
    (111, 151, 201),
    (215, 172, 83),
    (129, 170, 143),
)


@dataclass
class ReviewSession:
    handle: SessionHandle
    staged_path: Path | None = None
    normalized: NormalizedImage | None = None
    selected_provider_keys: tuple[str, ...] = ()
    selected_privacy_groups: tuple[str, ...] = ()
    service: ReviewExportService | None = None
    analysis: AnalysisSnapshot | None = None
    export_path: Path | None = None
    export_available: bool = False


class ReviewSessionManager:
    """Own local assets and expose only typed, session-relative results."""

    def __init__(
        self,
        root: str | Path,
        providers: dict[str, object],
        thresholds,
        *,
        provider_labels: dict[str, str] | None = None,
        privacy_groups: dict[str, set[str]] | None = None,
        ttl_seconds: int = 3600,
        ingest_limits: IngestLimits = IngestLimits(),
    ) -> None:
        self.store = SessionStore(root, ttl_seconds=ttl_seconds)
        self.providers = dict(providers)
        self.thresholds = thresholds
        self.provider_labels = provider_labels or {key: key.replace("-", " ").title() for key in providers}
        self.privacy_groups = privacy_groups or {}
        self.ingest_limits = ingest_limits
        self._sessions: dict[str, ReviewSession] = {}
        self._lock = threading.RLock()
        self._inference_lock = threading.Lock()

    @property
    def ttl_seconds(self) -> int:
        return self.store.ttl_seconds

    def config(self) -> AppConfigResponse:
        keys = list(dict.fromkeys([*self.provider_labels, *self.providers]))
        return AppConfigResponse(
            providers=[
                ProviderOption(
                    key=key,
                    label=self.provider_labels.get(key, key.replace("-", " ").title()),
                    available=key in self.providers,
                )
                for key in keys
            ],
            privacy_groups=list(self.privacy_groups),
            upload_max_bytes=self.ingest_limits.max_bytes,
            upload_max_pixels=self.ingest_limits.max_pixels,
            session_ttl_seconds=self.store.ttl_seconds,
        )

    def create(self) -> SessionResponse:
        with self._lock:
            expired = set(self.store.cleanup_expired())
            for session_id in expired:
                self._sessions.pop(session_id, None)
            handle = self.store.create()
            self._sessions[handle.session_id] = ReviewSession(handle)
            return SessionResponse(**handle.public_dict())

    def stage_asset(self, session_id: str, payload: bytes, suffix: str) -> AssetResponse:
        session = self._get(session_id)
        if len(payload) > self.ingest_limits.max_bytes:
            raise ValueError(f"Input image exceeds {self.ingest_limits.max_bytes} bytes")
        staged = self.store.stage_bytes(session.handle, payload, suffix=suffix)
        try:
            normalized = normalize_image(staged, self.ingest_limits)
        except Exception:
            staged.unlink(missing_ok=True)
            raise
        normalized_path = session.handle.root / "normalized.png"
        Image.fromarray(normalized.pixels_rgb).save(normalized_path, format="PNG")
        with self._lock:
            session.staged_path = staged
            session.normalized = normalized
            session.analysis = None
            session.service = None
            session.export_path = None
            session.export_available = False
        return AssetResponse(
            width=normalized.width,
            height=normalized.height,
            source_format=normalized.source_format,
            source_sha256=normalized.source_sha256,
            pixel_sha256=normalized.pixel_sha256,
            metadata_categories=list(normalized.metadata_categories),
            orientation_applied=normalized.orientation_applied,
            normalized_url=f"/v1/sessions/{session_id}/assets/normalized",
        )

    def analyze(
        self,
        session_id: str,
        provider_keys: list[str] | None,
        privacy_groups: list[str] | None,
    ) -> AnalysisResponse:
        session = self._get(session_id)
        if session.staged_path is None or session.normalized is None:
            raise ValueError("Upload an image before analysis")
        resolved_provider_keys = tuple(provider_keys or self.providers.keys())
        unknown_providers = sorted(set(resolved_provider_keys) - set(self.providers))
        if unknown_providers:
            raise ValueError(f"Unknown or unavailable providers: {', '.join(unknown_providers)}")
        if not resolved_provider_keys:
            raise ValueError("Select at least one available provider")
        resolved_groups = tuple(privacy_groups if privacy_groups is not None else self.privacy_groups.keys())
        unknown_groups = sorted(set(resolved_groups) - set(self.privacy_groups))
        if unknown_groups:
            raise ValueError(f"Unknown privacy groups: {', '.join(unknown_groups)}")

        selected = tuple(self.providers[key] for key in resolved_provider_keys)
        service = ReviewExportService(
            selected,
            self.thresholds,
            attack_providers=tuple(self.providers.values()),
        )
        with self._inference_lock:
            analysis = service.analyze(session.staged_path)

        selected_classes = {
            privacy_class
            for group in resolved_groups
            for privacy_class in self.privacy_groups.get(group, set())
        }
        chosen: list[tuple[object, np.ndarray]] = []
        for candidate in analysis.candidates.candidates:
            if selected_classes and not selected_classes.intersection(candidate.privacy_classes):
                continue
            mask = decode_binary_mask(candidate.mask_rle, candidate.height, candidate.width).astype(bool)
            chosen.append((candidate, mask))

        union = np.zeros((analysis.image.height, analysis.image.width), dtype=np.uint8)
        overlay = analysis.image.pixels_rgb.copy()
        summaries: list[CandidateSummary] = []
        for index, (candidate, mask) in enumerate(chosen):
            union[mask] = 255
            color = np.asarray(_COLORS[index % len(_COLORS)], dtype=np.float32)
            overlay[mask] = np.clip(overlay[mask] * 0.48 + color * 0.52, 0, 255).astype(np.uint8)
            ys, xs = np.nonzero(mask)
            if xs.size:
                cv2.rectangle(
                    overlay,
                    (int(xs.min()), int(ys.min())),
                    (int(xs.max()), int(ys.max())),
                    tuple(int(value) for value in color),
                    2,
                )
            summaries.append(
                CandidateSummary(
                    candidate_id=candidate.candidate_id,
                    privacy_classes=list(candidate.privacy_classes),
                    providers=list(candidate.providers),
                    uncertainty_flags=list(candidate.uncertainty_flags),
                    mandatory_review=candidate.mandatory_review,
                    mask_pixels=int(mask.sum()),
                )
            )

        Image.fromarray(union, mode="L").save(session.handle.root / "initial-mask.png", format="PNG")
        mask_overlay = np.zeros((analysis.image.height, analysis.image.width, 4), dtype=np.uint8)
        mask_overlay[:, :, :3] = np.asarray((49, 183, 166), dtype=np.uint8)
        mask_overlay[:, :, 3] = np.where(union > 0, 168, 0).astype(np.uint8)
        Image.fromarray(mask_overlay, mode="RGBA").save(session.handle.root / "mask-overlay.png", format="PNG")
        Image.fromarray(overlay).save(session.handle.root / "overlay.png", format="PNG")
        with self._lock:
            session.selected_provider_keys = resolved_provider_keys
            session.selected_privacy_groups = resolved_groups
            session.service = service
            session.analysis = analysis
            session.export_path = None
            session.export_available = False

        return AnalysisResponse(
            width=analysis.image.width,
            height=analysis.image.height,
            raw_evidence_count=len(analysis.evidence.evidence),
            candidates=summaries,
            selected_provider_keys=list(resolved_provider_keys),
            selected_privacy_groups=list(resolved_groups),
            unavailable_providers=list(analysis.evidence.unavailable_providers),
            provider_errors=analysis.provider_errors,
            threshold_profile_id=analysis.candidates.threshold_profile_id,
            threshold_profile_release_ready=analysis.candidates.threshold_profile_release_ready,
            normalized_url=f"/v1/sessions/{session_id}/assets/normalized",
            initial_mask_url=f"/v1/sessions/{session_id}/masks/initial",
            mask_overlay_url=f"/v1/sessions/{session_id}/assets/mask-overlay",
            overlay_url=f"/v1/sessions/{session_id}/assets/overlay",
        )

    def render(
        self,
        session_id: str,
        mask_payload: bytes,
        *,
        consent_state: ConsentState,
        subject_ref: str,
        operation: str,
        audience: str,
        purpose: str,
        review_completed: bool,
    ) -> RenderResponse:
        session = self._get(session_id)
        if session.analysis is None or session.service is None:
            raise ValueError("Analyze the image before rendering")
        for name, value in {
            "subject_ref": subject_ref,
            "operation": operation,
            "audience": audience,
            "purpose": purpose,
        }.items():
            if not value.strip():
                raise ValueError(f"{name} is required")
        mask = self._decode_mask(mask_payload, session.analysis.image.width, session.analysis.image.height)
        output = session.handle.root / "reviewed-redaction.png"
        with self._inference_lock:
            result = session.service.render_review(
                session.analysis,
                consent_state=consent_state,
                review_completed=review_completed,
                output_path=output,
                approved_mask=mask,
            )

        context_payload = json.dumps(
            {"operation": operation.strip(), "audience": audience.strip(), "purpose": purpose.strip()},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        request = ConsentRequest(
            media_version_digest=session.analysis.image.pixel_sha256,
            share_context_digest=hashlib.sha256(context_payload).hexdigest(),
            operation=operation.strip(),
            audience=audience.strip(),
            purpose=purpose.strip(),
        )
        bound_refs = tuple(candidate.candidate_id for candidate in session.analysis.candidates.candidates) + ("review-mask",)
        consent_record = ConsentRecord.create(
            subject_ref=subject_ref.strip(),
            bound_region_refs=bound_refs,
            request=request,
            state=consent_state,
            issued_at=datetime.now(timezone.utc),
        )
        export_available = bool(result.decision.export_allowed and consent_state is ConsentState.GRANTED)
        audit = {
            "session_id": session_id,
            "consent": consent_record.canonical_dict(),
            "review_completed": review_completed,
            "assurance_status": result.assurance.status.value,
            "assurance_checks": [
                {
                    "name": check.name,
                    "status": check.status.value,
                    "reason_code": check.reason_code,
                    "details": check.details,
                }
                for check in result.assurance.checks
            ],
            "decision": result.decision.to_dict(),
            "export_report": result.export_report or {},
        }
        (session.handle.root / "review-audit.json").write_text(
            json.dumps(audit, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with self._lock:
            session.export_path = output
            session.export_available = export_available

        return RenderResponse(
            assurance_status=result.assurance.status.value,
            assurance_checks=[
                AssuranceCheckResponse(
                    name=check.name,
                    status=check.status.value,
                    reason_code=check.reason_code,
                    details=check.details,
                )
                for check in result.assurance.checks
            ],
            decision=ReleaseDecisionResponse(**result.decision.to_dict()),
            export_report=result.export_report or {},
            rendered_url=f"/v1/sessions/{session_id}/assets/rendered",
            export_available=export_available,
        )

    def asset_path(self, session_id: str, asset: str) -> Path:
        session = self._get(session_id)
        mapping = {
            "normalized": session.handle.root / "normalized.png",
            "initial-mask": session.handle.root / "initial-mask.png",
            "mask-overlay": session.handle.root / "mask-overlay.png",
            "overlay": session.handle.root / "overlay.png",
            "rendered": session.handle.root / "reviewed-redaction.png",
        }
        if asset not in mapping or not mapping[asset].is_file():
            raise FileNotFoundError(asset)
        return mapping[asset]

    def export_path(self, session_id: str) -> Path:
        session = self._get(session_id)
        if not session.export_available or session.export_path is None or not session.export_path.is_file():
            raise PermissionError("No verified export capability exists for this session")
        return session.export_path

    def delete(self, session_id: str) -> None:
        session = self._get(session_id)
        with self._lock:
            self.store.delete(session.handle)
            self._sessions.pop(session_id, None)

    def _get(self, session_id: str) -> ReviewSession:
        with self._lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(session_id)
        if time.time() >= session.handle.expires_at:
            with self._lock:
                self._sessions.pop(session_id, None)
            raise TimeoutError(session_id)
        return session

    @staticmethod
    def _decode_mask(payload: bytes, width: int, height: int) -> np.ndarray:
        if not payload:
            raise ValueError("Approved mask is empty")
        try:
            with Image.open(BytesIO(payload)) as image:
                image.load()
                if image.size != (width, height):
                    raise ValueError("Approved mask dimensions must match the normalized image")
                mask = np.asarray(image.convert("L"), dtype=np.uint8)
        except (UnidentifiedImageError, OSError) as error:
            raise ValueError("Approved mask must be a decodable image") from error
        return (mask > 0).astype(np.uint8)
