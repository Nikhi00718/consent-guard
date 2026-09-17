"""Shared interface for optional and built-in evidence providers."""

from __future__ import annotations

from typing import Protocol

from consentguard.stage_04_fusion_calibration.domain import Evidence
from consentguard.stage_05_review_export.ingest import NormalizedImage


class ProviderUnavailableError(RuntimeError):
    """Raised when weights, dependencies, or runtime support are unavailable."""


class EvidenceProvider(Protocol):
    name: str
    version: str

    def analyze(self, image: NormalizedImage) -> list[Evidence]:
        """Return original-image evidence or raise ProviderUnavailableError."""
