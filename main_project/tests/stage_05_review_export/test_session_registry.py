import os
from pathlib import Path

import pytest

from consentguard.stage_04_fusion_calibration.domain import Evidence, EvidenceGeometry
from consentguard.stage_05_review_export.evidence_registry import EvidenceRegistry
from consentguard.stage_05_review_export.ingest import SessionStore


def _evidence(evidence_id: str) -> Evidence:
    return Evidence(
        evidence_id=evidence_id,
        provider="test-provider",
        provider_version="v1",
        privacy_class="face",
        geometry=EvidenceGeometry(4, 4, box_xyxy=(0, 0, 2, 2)),
        confidence=0.8,
    )


def test_session_store_uses_random_root_and_never_user_filename(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions", ttl_seconds=10)
    handle = store.create()
    staged = store.stage_bytes(handle, b"payload", suffix=".JPG")
    assert handle.session_id.startswith("session-")
    assert staged.parent == handle.root
    assert staged.name.startswith("upload-")
    assert "payload" not in staged.name
    assert handle.public_dict().keys() == {"session_id", "created_at", "expires_at"}


def test_session_store_rejects_path_escape_and_cleans_expired(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions", ttl_seconds=5)
    handle = store.create()
    forged = type(handle)("session-../escape", tmp_path, handle.created_at, handle.expires_at)
    with pytest.raises(ValueError):
        store.stage_bytes(forged, b"bad")
    # The real session directory is old enough to be removed.
    os.utime(handle.root, (0, 0))
    assert store.cleanup_expired(now=handle.created_at + 100.0) == (handle.session_id,)


def test_evidence_registry_deduplicates_and_preserves_unavailable_state() -> None:
    registry = EvidenceRegistry()
    item = _evidence("e-2")
    registry.add(item)
    registry.add(item)
    registry.mark_unavailable("ocr")
    snapshot = registry.snapshot()
    assert snapshot.evidence == (item,)
    assert snapshot.unavailable_providers == ("ocr",)
    with pytest.raises(ValueError, match="collision"):
        registry.add(Evidence(
            evidence_id="e-2", provider="other", provider_version="v1",
            privacy_class="face", geometry=EvidenceGeometry(4, 4, box_xyxy=(0, 0, 2, 2)), confidence=0.8
        ))
