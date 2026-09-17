from pathlib import Path

import cv2
import numpy as np

from consentguard.stage_04_fusion_calibration.domain import (
    AssuranceStatus,
    ConsentState,
    Evidence,
    EvidenceGeometry,
)
from consentguard.stage_04_fusion_calibration.evidence import ThresholdRegistry
from consentguard.stage_04_fusion_calibration.evidence.base import ProviderUnavailableError
from consentguard.stage_05_review_export.pipeline import ReviewExportService


class FakeProvider:
    name = "fake"
    version = "fake-v1"

    def analyze(self, image):
        return [
            Evidence(
                evidence_id="fake-e1",
                provider=self.name,
                provider_version=self.version,
                privacy_class="face",
                geometry=EvidenceGeometry(
                    image.width, image.height, box_xyxy=(1, 1, min(5, image.width), min(5, image.height))
                ),
                confidence=0.99,
            )
        ]


class MissingProvider:
    name = "missing"
    version = "missing-v1"

    def analyze(self, image):
        raise ProviderUnavailableError("weights not installed")


def _thresholds(tmp_path: Path) -> ThresholdRegistry:
    profile = tmp_path / "thresholds.yaml"
    profile.write_text(
        "profile_id: test\nrelease_ready: true\nrules:\n"
        "  - {provider: fake, privacy_class: face, score_threshold: 0.5, min_area_pixels: 1, mandatory_review: true}\n",
        encoding="utf-8",
    )
    return ThresholdRegistry.load(profile)


def test_pipeline_preserves_unavailable_provider_and_blocks_without_review(tmp_path: Path) -> None:
    source = tmp_path / "source.jpg"
    assert cv2.imwrite(str(source), np.full((12, 16, 3), 120, dtype=np.uint8))
    result = ReviewExportService((FakeProvider(), MissingProvider()), _thresholds(tmp_path)).run(
        source,
        consent_state=ConsentState.GRANTED,
        review_completed=False,
    )
    assert result.evidence.unavailable_providers == ("missing",)
    assert result.candidates.candidates
    assert result.decision.export_allowed is False
    assert "REQUIRED_PROVIDER_UNAVAILABLE" in result.decision.reason_codes


def test_pipeline_renders_and_requires_independent_attacks(tmp_path: Path) -> None:
    source = tmp_path / "source.jpg"
    output = tmp_path / "redacted.png"
    assert cv2.imwrite(str(source), np.full((12, 16, 3), 120, dtype=np.uint8))
    mask = np.zeros((12, 16), dtype=np.uint8)
    mask[1:5, 1:5] = 1
    result = ReviewExportService((FakeProvider(),), _thresholds(tmp_path)).run(
        source,
        consent_state=ConsentState.GRANTED,
        review_completed=True,
        approved_mask=mask,
        output_path=output,
        attack_checks={"ocr": AssuranceStatus.PASS, "barcode": AssuranceStatus.PASS, "face": AssuranceStatus.PASS, "plate": AssuranceStatus.PASS},
    )
    assert output.is_file()
    assert result.assurance.status is AssuranceStatus.PASS
    assert result.decision.export_allowed
