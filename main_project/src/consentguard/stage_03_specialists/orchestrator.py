"""Run evidence providers sequentially and preserve unavailable states."""

from __future__ import annotations

from dataclasses import dataclass

from consentguard.stage_04_fusion_calibration.domain import Evidence
from consentguard.stage_04_fusion_calibration.evidence.base import (
    EvidenceProvider,
    ProviderUnavailableError,
)
from consentguard.stage_05_review_export.ingest import NormalizedImage


@dataclass(frozen=True)
class AnalysisResult:
    evidence: tuple[Evidence, ...]
    unavailable_providers: tuple[str, ...]
    provider_errors: dict[str, str]


class AnalysisOrchestrator:
    """GPU-safe default: providers run one at a time in configured order."""

    def __init__(self, providers: tuple[EvidenceProvider, ...]) -> None:
        names = [provider.name for provider in providers]
        if len(names) != len(set(names)):
            raise ValueError("Provider names must be unique")
        self.providers = providers

    def analyze(self, image: NormalizedImage) -> AnalysisResult:
        evidence: list[Evidence] = []
        unavailable: list[str] = []
        errors: dict[str, str] = {}
        for provider in self.providers:
            try:
                evidence.extend(provider.analyze(image))
            except ProviderUnavailableError as error:
                unavailable.append(provider.name)
                errors[provider.name] = str(error)
        return AnalysisResult(
            evidence=tuple(evidence),
            unavailable_providers=tuple(sorted(unavailable)),
            provider_errors=dict(sorted(errors.items())),
        )
