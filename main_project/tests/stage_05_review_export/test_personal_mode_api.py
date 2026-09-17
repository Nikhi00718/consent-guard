"""The one-click flow a person actually uses: upload, erase, download."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from consentguard.stage_04_fusion_calibration.domain import Evidence, EvidenceGeometry
from consentguard.stage_04_fusion_calibration.evidence import ThresholdRegistry
from consentguard.stage_05_review_export.api import create_app
from consentguard.stage_05_review_export.policy import PERSONAL_MODE


class FaceProvider:
    name = "maskrcnn"
    version = "fixture-v1"

    def analyze(self, image):
        return [
            Evidence(
                evidence_id="face-1",
                provider=self.name,
                provider_version=self.version,
                privacy_class="face",
                geometry=EvidenceGeometry(image.width, image.height, box_xyxy=(2, 2, 10, 8)),
                confidence=0.42,
            )
        ]


def _thresholds(tmp_path: Path) -> ThresholdRegistry:
    tmp_path.mkdir(parents=True, exist_ok=True)
    profile = tmp_path / "personal.yaml"
    profile.write_text(
        "profile_id: personal-test\nrelease_ready: false\nrules:\n"
        "  - {provider: '*', privacy_class: face, score_threshold: 0.3, min_area_pixels: 1,"
        " dilation_pixels: 1, mandatory_review: true}\n",
        encoding="utf-8",
    )
    return ThresholdRegistry.load(profile)


def _image_bytes(image_format: str) -> bytes:
    buffer = BytesIO()
    Image.fromarray(np.full((24, 32, 3), 140, dtype=np.uint8)).save(buffer, format=image_format)
    return buffer.getvalue()


def _client(tmp_path: Path, *, policy_mode: str = PERSONAL_MODE) -> TestClient:
    app = create_app(
        {"global": FaceProvider()},
        _thresholds(tmp_path / "profile"),
        provider_labels={"global": "Everything detector"},
        privacy_groups={"Faces": {"face"}, "Bodies": {"person_body"}},
        session_root=tmp_path / "sessions",
        policy_mode=policy_mode,
        default_privacy_groups=("Faces",),
        group_reliability={"Faces": "reliable", "Bodies": "check_manually"},
    )
    return TestClient(app)


def _uploaded(client: TestClient, image_format: str = "PNG") -> str:
    session_id = client.post("/v1/sessions").json()["session_id"]
    suffix = {"PNG": "png", "JPEG": "jpg", "WEBP": "webp"}[image_format]
    response = client.post(
        f"/v1/sessions/{session_id}/assets",
        files={"asset": (f"holiday.{suffix}", _image_bytes(image_format), f"image/{suffix}")},
    )
    assert response.status_code == 201
    return session_id


def test_config_advertises_personal_defaults(tmp_path: Path) -> None:
    config = _client(tmp_path).get("/v1/config").json()
    assert config["policy_mode"] == PERSONAL_MODE
    assert config["default_privacy_groups"] == ["Faces"]
    assert config["group_reliability"]["Bodies"] == "check_manually"


def test_one_click_flow_produces_a_downloadable_file(tmp_path: Path) -> None:
    client = _client(tmp_path)
    session_id = _uploaded(client)

    analysis = client.post(f"/v1/sessions/{session_id}/analyze", json={}).json()
    assert analysis["selected_privacy_groups"] == ["Faces"]
    assert analysis["auto_mask_pixels"] > 0

    redacted = client.post(f"/v1/sessions/{session_id}/auto-redact")
    assert redacted.status_code == 200
    body = redacted.json()
    assert body["decision"]["action"] == "ALLOW_REDACTED"
    assert body["export_available"] is True
    assert "WARNING_EXPERIMENTAL_DETECTION_PROFILE" in body["warnings"]

    exported = client.get(f"/v1/sessions/{session_id}/export")
    assert exported.status_code == 200
    assert exported.headers["content-disposition"].endswith('filename="consentguard-redacted.png"')

    with Image.open(BytesIO(exported.content)) as output:
        pixels = np.asarray(output.convert("RGB"))
    assert pixels.shape == (24, 32, 3)
    assert (pixels[2:8, 2:10] == 0).all(), "the detected region must be solid black"
    assert (pixels[20:, 24:] == 140).all(), "untouched pixels must survive"


def test_output_format_matches_the_input(tmp_path: Path) -> None:
    client = _client(tmp_path)
    session_id = _uploaded(client, "JPEG")
    client.post(f"/v1/sessions/{session_id}/analyze", json={})
    body = client.post(f"/v1/sessions/{session_id}/auto-redact").json()
    assert body["export_filename"] == "consentguard-redacted.jpg"
    assert client.get(f"/v1/sessions/{session_id}/export").headers["content-type"] == "image/jpeg"


def test_manual_review_needs_no_consent_paperwork(tmp_path: Path) -> None:
    client = _client(tmp_path)
    session_id = _uploaded(client)
    client.post(f"/v1/sessions/{session_id}/analyze", json={})
    mask = BytesIO()
    painted = np.zeros((24, 32), dtype=np.uint8)
    painted[4:12, 4:16] = 255
    Image.fromarray(painted).save(mask, format="PNG")

    response = client.post(
        f"/v1/sessions/{session_id}/render",
        files={"mask": ("mask.png", mask.getvalue(), "image/png")},
    )
    assert response.status_code == 200
    assert response.json()["export_available"] is True


def test_research_mode_keeps_its_gate_and_rejects_one_click(tmp_path: Path) -> None:
    client = _client(tmp_path, policy_mode="research")
    session_id = _uploaded(client)
    client.post(f"/v1/sessions/{session_id}/analyze", json={})
    assert client.post(f"/v1/sessions/{session_id}/auto-redact").status_code == 403
    assert client.get(f"/v1/sessions/{session_id}/export").status_code == 403
