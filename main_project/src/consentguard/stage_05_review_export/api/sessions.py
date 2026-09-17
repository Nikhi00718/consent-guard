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
from consentguard.stage_05_review_export.policy import (
    PERSONAL_MODE,
    RESEARCH_MODE,
    ConsentRecord,
    ConsentRequest,
    ReleasePolicy,
)

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


#: Suffix used for the newly encoded output, keyed by decoded source format, so
#: a JPEG photo comes back as a JPEG and a PNG screenshot stays lossless.
_OUTPUT_SUFFIX = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}


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
    auto_mask: np.ndarray | None = None
    export_filename: str = "consentguard-redacted.png"


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
        policy_mode: str = RESEARCH_MODE,
        default_privacy_groups: tuple[str, ...] | None = None,
        group_reliability: dict[str, str] | None = None,
    ) -> None:
        self.store = SessionStore(root, ttl_seconds=ttl_seconds)
        self.providers = dict(providers)
        self.thresholds = thresholds
        self.provider_labels = provider_labels or {key: key.replace("-", " ").title() for key in providers}
        self.privacy_groups = privacy_groups or {}
        self.ingest_limits = ingest_limits
        self.policy_mode = policy_mode
        self.default_privacy_groups = tuple(default_privacy_groups or self.privacy_groups.keys())
        self.group_reliability = dict(group_reliability or {})
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
            policy_mode=self.policy_mode,
            default_privacy_groups=[
                group for group in self.default_privacy_groups if group in self.privacy_groups
            ],
            group_reliability=dict(self.group_reliability),
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
        resolved_groups = tuple(privacy_groups if privacy_groups is not None else self.default_privacy_groups)
        unknown_groups = sorted(set(resolved_groups) - set(self.privacy_groups))
        if unknown_groups:
            raise ValueError(f"Unknown privacy groups: {', '.join(unknown_groups)}")

        selected = tuple(self.providers[key] for key in resolved_provider_keys)
        service = ReviewExportService(
            selected,
            self.thresholds,
            policy=ReleasePolicy(mode=self.policy_mode),
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
            session.auto_mask = union
            session.export_path = None
            session.export_available = False
            session.export_filename = f"consentguard-redacted{self._output_suffix(session)}"

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
            auto_mask_pixels=int(np.count_nonzero(union)),
            unreliable_groups=[
                group
                for group in resolved_groups
                if self.group_reliability.get(group, "reliable") != "reliable"
            ],
        )

    def auto_redact(self, session_id: str) -> RenderResponse:
        """One-click path: erase every detected region without a review pass.

        The automatic mask is the same union the review screen starts from, so
        accepting it here and accepting it after editing follow one code path.
        Only available in personal policy mode; research mode requires the
        explicit consent and review inputs.
        """

        session = self._get(session_id)
        if self.policy_mode != PERSONAL_MODE:
            raise PermissionError("One-click redaction requires personal policy mode")
        if session.analysis is None or session.auto_mask is None:
            raise ValueError("Analyze the image before rendering")
        return self._render_mask(session, session.auto_mask, review_completed=True)

    def render(
        self,
        session_id: str,
        mask_payload: bytes,
        *,
        consent_state: ConsentState = ConsentState.UNKNOWN,
        subject_ref: str | None = None,
        operation: str | None = None,
        audience: str | None = None,
        purpose: str | None = None,
        review_completed: bool = True,
    ) -> RenderResponse:
        session = self._get(session_id)
        if session.analysis is None or session.service is None:
            raise ValueError("Analyze the image before rendering")
        context = {
            "subject_ref": subject_ref,
            "operation": operation,
            "audience": audience,
            "purpose": purpose,
        }
        if self.policy_mode == RESEARCH_MODE:
            for name, value in context.items():
                if not (value or "").strip():
                    raise ValueError(f"{name} is required")
        mask = self._decode_mask(mask_payload, session.analysis.image.width, session.analysis.image.height)
        return self._render_mask(
            session,
            mask,
            consent_state=consent_state,
            review_completed=review_completed,
            **{name: value for name, value in context.items()},
        )

    def _render_mask(
        self,
        session: ReviewSession,
        mask: np.ndarray,
        *,
        consent_state: ConsentState = ConsentState.UNKNOWN,
        subject_ref: str | None = None,
        operation: str | None = None,
        audience: str | None = None,
        purpose: str | None = None,
        review_completed: bool = True,
    ) -> RenderResponse:
        if session.analysis is None or session.service is None:
            raise ValueError("Analyze the image before rendering")
        session_id = session.handle.session_id
        output = session.handle.root / f"reviewed-redaction{self._output_suffix(session)}"
        with self._inference_lock:
            result = session.service.render_review(
                session.analysis,
                consent_state=consent_state,
                review_completed=review_completed,
                output_path=output,
                approved_mask=mask,
            )

        consent_record = self._consent_record(
            session,
            consent_state=consent_state,
            subject_ref=subject_ref,
            operation=operation,
            audience=audience,
            purpose=purpose,
        )
        if self.policy_mode == PERSONAL_MODE:
            export_available = bool(result.decision.export_allowed)
        else:
            export_available = bool(result.decision.export_allowed and consent_state is ConsentState.GRANTED)
        audit = {
            "session_id": session_id,
            "consent": consent_record.canonical_dict() if consent_record is not None else None,
            "policy_mode": self.policy_mode,
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
            export_filename=session.export_filename,
            warnings=[reason for reason in result.decision.reason_codes if reason.startswith("WARNING_")],
        )

    def asset_path(self, session_id: str, asset: str) -> Path:
        session = self._get(session_id)
        mapping = {
            "normalized": session.handle.root / "normalized.png",
            "initial-mask": session.handle.root / "initial-mask.png",
            "mask-overlay": session.handle.root / "mask-overlay.png",
            "overlay": session.handle.root / "overlay.png",
            "rendered": session.handle.root / f"reviewed-redaction{self._output_suffix(session)}",
        }
        if asset not in mapping or not mapping[asset].is_file():
            raise FileNotFoundError(asset)
        return mapping[asset]

    def export_path(self, session_id: str) -> Path:
        session = self._get(session_id)
        if not session.export_available or session.export_path is None or not session.export_path.is_file():
            raise PermissionError("No verified export capability exists for this session")
        return session.export_path

    def export_filename(self, session_id: str) -> str:
        return self._get(session_id).export_filename

    @staticmethod
    def _output_suffix(session: ReviewSession) -> str:
        source_format = session.normalized.source_format if session.normalized is not None else "PNG"
        return _OUTPUT_SUFFIX.get(source_format.upper(), ".png")

    def _consent_record(
        self,
        session: ReviewSession,
        *,
        consent_state: ConsentState,
        subject_ref: str | None,
        operation: str | None,
        audience: str | None,
        purpose: str | None,
    ) -> ConsentRecord | None:
        """Record a scoped consent assertion only when the caller supplied one.

        Personal mode does not collect consent fields, so there is nothing to
        assert and the audit file stores ``null`` rather than an invented
        record.
        """

        fields = {
            "subject_ref": (subject_ref or "").strip(),
            "operation": (operation or "").strip(),
            "audience": (audience or "").strip(),
            "purpose": (purpose or "").strip(),
        }
        if not all(fields.values()):
            return None
        context_payload = json.dumps(
            {key: fields[key] for key in ("operation", "audience", "purpose")},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        request = ConsentRequest(
            media_version_digest=session.analysis.image.pixel_sha256,
            share_context_digest=hashlib.sha256(context_payload).hexdigest(),
            operation=fields["operation"],
            audience=fields["audience"],
            purpose=fields["purpose"],
        )
        bound_refs = tuple(
            candidate.candidate_id for candidate in session.analysis.candidates.candidates
        ) + ("review-mask",)
        return ConsentRecord.create(
            subject_ref=fields["subject_ref"],
            bound_region_refs=bound_refs,
            request=request,
            state=consent_state,
            issued_at=datetime.now(timezone.utc),
        )

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
