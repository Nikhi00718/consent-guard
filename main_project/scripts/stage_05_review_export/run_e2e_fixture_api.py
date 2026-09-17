"""Run a deterministic, model-free reviewer API for browser end-to-end tests."""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import uvicorn

from consentguard.stage_04_fusion_calibration.domain import Evidence, EvidenceGeometry
from consentguard.stage_04_fusion_calibration.evidence import ThresholdRegistry
from consentguard.stage_05_review_export.api import create_app


class _DetectionProvider:
    name = "e2e_fixture"
    version = "e2e-fixture-v1"

    def analyze(self, image):
        right = max(2, min(image.width - 1, image.width // 2))
        bottom = max(2, min(image.height - 1, image.height // 2))
        return [
            Evidence(
                evidence_id="e2e-face-1",
                provider=self.name,
                provider_version=self.version,
                privacy_class="face",
                geometry=EvidenceGeometry(
                    image.width,
                    image.height,
                    box_xyxy=(1, 1, right, bottom),
                ),
                confidence=0.99,
            )
        ]


class _EmptyProvider:
    version = "e2e-fixture-v1"

    def __init__(self, name: str) -> None:
        self.name = name

    def analyze(self, _image):
        return []


def _thresholds(root: Path) -> ThresholdRegistry:
    profile = root / "thresholds.yaml"
    profile.write_text(
        "profile_id: e2e-fixture\n"
        "release_ready: false\n"
        "rules:\n"
        "  - {provider: e2e_fixture, privacy_class: face, score_threshold: 0.5, "
        "min_area_pixels: 1, mandatory_review: true}\n",
        encoding="utf-8",
    )
    return ThresholdRegistry.load(profile)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7861)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    with tempfile.TemporaryDirectory(prefix="consentguard-e2e-") as temporary:
        root = Path(temporary)
        providers = {
            "fixture": _DetectionProvider(),
            "ocr": _EmptyProvider("paddleocr"),
            "barcode": _EmptyProvider("zxingcpp"),
            "face-attacker": _EmptyProvider("yunet"),
            "plate-attacker": _EmptyProvider("plate_yunet"),
        }
        app = create_app(
            providers,
            _thresholds(root),
            provider_labels={key: key.replace("-", " ").title() for key in providers},
            privacy_groups={"Faces": {"face"}},
            session_root=root / "sessions",
        )
        uvicorn.run(app, host=args.host, port=args.port, access_log=False)


if __name__ == "__main__":
    main()
