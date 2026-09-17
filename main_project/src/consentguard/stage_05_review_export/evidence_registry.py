"""Small in-memory evidence registry with provenance and deduplication."""

from __future__ import annotations

from dataclasses import dataclass

from consentguard.stage_04_fusion_calibration.domain import Evidence


@dataclass(frozen=True)
class EvidenceSnapshot:
    evidence: tuple[Evidence, ...]
    unavailable_providers: tuple[str, ...]


class EvidenceRegistry:
    """Track provider evidence without storing OCR plaintext or source paths."""

    def __init__(self) -> None:
        self._evidence: dict[str, Evidence] = {}
        self._unavailable: set[str] = set()

    def add(self, item: Evidence) -> None:
        existing = self._evidence.get(item.evidence_id)
        if existing is not None and existing != item:
            raise ValueError(f"evidence ID collision: {item.evidence_id}")
        self._evidence[item.evidence_id] = item

    def mark_unavailable(self, provider: str) -> None:
        if not provider or "/" in provider or "\\" in provider:
            raise ValueError("provider name must be a simple non-empty identifier")
        self._unavailable.add(provider)

    def snapshot(self) -> EvidenceSnapshot:
        return EvidenceSnapshot(
            evidence=tuple(self._evidence[key] for key in sorted(self._evidence)),
            unavailable_providers=tuple(sorted(self._unavailable)),
        )

    def clear(self) -> None:
        self._evidence.clear()
        self._unavailable.clear()
