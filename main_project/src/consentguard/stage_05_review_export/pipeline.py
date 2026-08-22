"""Framework-independent analysis, review, render, and release service."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from consentguard.stage_03_specialists.orchestrator import AnalysisOrchestrator
from consentguard.stage_04_fusion_calibration.domain import (
    AssuranceCheck,
    AssuranceReport,
    AssuranceStatus,
    ConsentState,
    ReviewCandidateSet,
    ReleaseDecision,
)
from consentguard.stage_04_fusion_calibration.evidence import EvidenceFusion, ThresholdRegistry
from consentguard.stage_05_review_export.assurance import AssuranceService, RenderedAsset
from consentguard.stage_05_review_export.evidence_registry import EvidenceRegistry, EvidenceSnapshot
from consentguard.stage_05_review_export.ingest import NormalizedImage, normalize_image
from consentguard.stage_05_review_export.policy import ReleasePolicy
from consentguard.stage_05_review_export.redaction.prediction_renderer import write_metadata_free_redaction


@dataclass(frozen=True)
class PipelineResult:
    image: NormalizedImage
    evidence: EvidenceSnapshot
    candidates: ReviewCandidateSet
    assurance: AssuranceReport
    decision: ReleaseDecision
    output_path: Path | None
    export_report: dict[str, object] | None
    provider_errors: dict[str, str]


class ReviewExportService:
    """Compose existing stages without letting providers make policy decisions."""

    def __init__(
        self,
        providers: tuple[object, ...],
        thresholds: ThresholdRegistry,
        *,
        assurance: AssuranceService | None = None,
        policy: ReleasePolicy | None = None,
        attack_providers: tuple[object, ...] = (),
    ) -> None:
        self.orchestrator = AnalysisOrchestrator(providers)  # type: ignore[arg-type]
        self.fusion = EvidenceFusion(thresholds)
        self.assurance = assurance or AssuranceService()
        self.policy = policy or ReleasePolicy()
        self.attack_providers = tuple(attack_providers)

    def run(
        self,
        source_path: str | Path,
        *,
        consent_state: ConsentState,
        review_completed: bool,
        output_path: str | Path | None = None,
        approved_mask: np.ndarray | None = None,
        allow_unchanged: bool = False,
        attack_checks: dict[str, AssuranceStatus] | None = None,
    ) -> PipelineResult:
        image = normalize_image(source_path)
        analysis = self.orchestrator.analyze(image)
        registry = EvidenceRegistry()
        for item in analysis.evidence:
            registry.add(item)
        for provider in analysis.unavailable_providers:
            registry.mark_unavailable(provider)
        snapshot = registry.snapshot()
        candidates = self.fusion.combine(
            list(snapshot.evidence),
            width=image.width,
            height=image.height,
            unavailable_providers=snapshot.unavailable_providers,
        )

        rendered: dict[str, object] | None = None
        destination: Path | None = Path(output_path).resolve() if output_path is not None else None
        redaction_applied = False
        assurance_report = AssuranceReport(
            (AssuranceCheck("render", AssuranceStatus.NOT_RUN, "RENDER_NOT_REQUESTED"),)
        )
        if destination is not None:
            if approved_mask is None:
                if allow_unchanged and not candidates.candidates:
                    approved_mask = np.zeros((image.height, image.width), dtype=np.uint8)
                else:
                    raise ValueError("approved_mask is required for a redacted export")
            if approved_mask.shape != (image.height, image.width):
                raise ValueError("approved_mask dimensions must match normalized image")
            approved_mask = (approved_mask > 0).astype(np.uint8)
            redaction_applied = bool(approved_mask.any())
            rendered = write_metadata_free_redaction(
                image.source_path,
                destination,
                image.pixels_rgb,
                approved_mask,
            )
            resolved_attack_checks = attack_checks
            if resolved_attack_checks is None and self.attack_providers:
                resolved_attack_checks = self.assurance.run_attack_checks(destination, self.attack_providers)
            assurance_report = self.assurance.inspect(
                RenderedAsset(
                    destination,
                    image.width,
                    image.height,
                    rendered,
                    attack_checks=resolved_attack_checks or {},
                )
            )

        decision = self.policy.decide(
            candidates,
            consent_state,
            assurance_report,
            review_completed=review_completed,
            redaction_applied=redaction_applied,
            allow_unchanged=allow_unchanged,
        )
        return PipelineResult(
            image=image,
            evidence=snapshot,
            candidates=candidates,
            assurance=assurance_report,
            decision=decision,
            output_path=destination,
            export_report=rendered,
            provider_errors=analysis.provider_errors,
        )
