from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from consentguard.stage_04_fusion_calibration.domain import Evidence, EvidenceGeometry
from consentguard.stage_04_fusion_calibration.evidence import ThresholdRegistry
from consentguard.stage_05_review_export.api import create_app


class DetectionProvider:
    name = "fake"
    version = "fake-v1"

    def analyze(self, image):
        return [
            Evidence(
                evidence_id="face-1",
                provider=self.name,
                provider_version=self.version,
                privacy_class="face",
                geometry=EvidenceGeometry(image.width, image.height, box_xyxy=(2, 2, 8, 8)),
                confidence=0.96,
            )
        ]


class EmptyProvider:
    version = "test-v1"

    def __init__(self, name: str) -> None:
        self.name = name

    def analyze(self, _image):
        return []


def _thresholds(tmp_path: Path, *, release_ready: bool = True) -> ThresholdRegistry:
    tmp_path.mkdir(parents=True, exist_ok=True)
    profile = tmp_path / "thresholds.yaml"
    profile.write_text(
        f"profile_id: api-test\nrelease_ready: {str(release_ready).lower()}\nrules:\n"
        "  - {provider: fake, privacy_class: face, score_threshold: 0.5, min_area_pixels: 1, mandatory_review: true}\n",
        encoding="utf-8",
    )
    return ThresholdRegistry.load(profile)


def _png(*, mask: bool = False) -> bytes:
    pixels = np.zeros((12, 16), dtype=np.uint8) if mask else np.full((12, 16, 3), 132, dtype=np.uint8)
    if mask:
        pixels[1:10, 1:12] = 255
    buffer = BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG")
    return buffer.getvalue()


def _client(tmp_path: Path, *, release_ready: bool = True) -> TestClient:
    providers = {
        "detect": DetectionProvider(),
        "ocr": EmptyProvider("paddleocr"),
        "barcode": EmptyProvider("zxingcpp"),
        "face": EmptyProvider("yunet"),
        "plate": EmptyProvider("plate_yunet"),
    }
    app = create_app(
        providers,
        _thresholds(tmp_path, release_ready=release_ready),
        provider_labels={key: key for key in providers},
        privacy_groups={"Face": {"face"}},
        session_root=tmp_path / "sessions",
    )
    return TestClient(app)


def _analyzed_session(client: TestClient) -> str:
    session_id = client.post("/v1/sessions").json()["session_id"]
    uploaded = client.post(
        f"/v1/sessions/{session_id}/assets",
        files={"asset": ("private-source.png", _png(), "image/png")},
    )
    assert uploaded.status_code == 201
    analyzed = client.post(
        f"/v1/sessions/{session_id}/analyze",
        json={"provider_keys": ["detect"], "privacy_groups": ["Face"]},
    )
    assert analyzed.status_code == 200
    assert analyzed.json()["candidates"][0]["privacy_classes"] == ["face"]
    return session_id


def _render(client: TestClient, session_id: str, consent_state: str = "GRANTED"):
    return client.post(
        f"/v1/sessions/{session_id}/render",
        files={"mask": ("approved-mask.png", _png(mask=True), "image/png")},
        data={
            "consent_state": consent_state,
            "subject_ref": "subject-01",
            "operation": "share",
            "audience": "research-team",
            "purpose": "review fixture",
            "review_completed": "true",
        },
    )


def test_reviewer_api_completes_gated_export_flow(tmp_path: Path) -> None:
    client = _client(tmp_path)
    session_id = _analyzed_session(client)
    normalized = client.get(f"/v1/sessions/{session_id}/assets/normalized")
    assert normalized.status_code == 200
    assert normalized.headers["cache-control"].startswith("no-store")

    rendered = _render(client, session_id)
    assert rendered.status_code == 200
    assert rendered.json()["export_available"] is True
    exported = client.get(f"/v1/sessions/{session_id}/export")
    assert exported.status_code == 200
    assert exported.headers["content-type"] == "image/png"


def test_reviewer_api_blocks_denied_consent_and_unready_profile(tmp_path: Path) -> None:
    denied_client = _client(tmp_path / "denied")
    denied_session = _analyzed_session(denied_client)
    denied = _render(denied_client, denied_session, "DENIED")
    assert denied.json()["decision"]["action"] == "REJECT_EXPORT"
    assert denied_client.get(f"/v1/sessions/{denied_session}/export").status_code == 403

    unready_client = _client(tmp_path / "unready", release_ready=False)
    unready_session = _analyzed_session(unready_client)
    unready = _render(unready_client, unready_session)
    assert unready.json()["export_available"] is False
    assert "THRESHOLD_PROFILE_NOT_RELEASE_READY" in unready.json()["decision"]["reason_codes"]


def test_reviewer_api_deletion_revokes_all_assets(tmp_path: Path) -> None:
    client = _client(tmp_path)
    session_id = _analyzed_session(client)
    assert client.delete(f"/v1/sessions/{session_id}").status_code == 204
    assert client.get(f"/v1/sessions/{session_id}/assets/normalized").status_code == 404
