"""Optional NudeNet adapter for intimate-content localization.

The trained nudity head covers 93% of labelled nudity pixels but fires on only
a third of individual regions (reports/PERSONAL_PROFILE_SCORECARD_2026-09-18.md),
and the Visual Redactions split holds just 27 nudity instances — too few to
train against. NudeNet is a maintained, pretrained body-part detector, so this
branch is replaced rather than retrained.

Detection only: this returns boxes for exposed and covered intimate regions and
never identifies a person. Faces reported by NudeNet are ignored; the face
providers own that class.

NudeNet is AGPL-3.0, which suits local personal use. Shipping ConsentGuard as a
hosted service with this provider enabled would place the AGPL's source-offer
obligation on that service, so the provider stays optional and off unless it is
explicitly configured.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from consentguard.stage_03_specialists.common import stable_evidence_id
from consentguard.stage_04_fusion_calibration.domain import Evidence, EvidenceGeometry, PrivacyClass
from consentguard.stage_04_fusion_calibration.evidence.base import ProviderUnavailableError
from consentguard.stage_05_review_export.ingest import NormalizedImage


#: NudeNet labels that describe intimate content. The "COVERED" variants are
#: kept deliberately: a redactor is asked to cover what a viewer would read as
#: intimate, and the reviewer can always un-erase. Faces, feet, armpits and
#: belly are excluded — either another provider owns the class, or the region is
#: not privacy-sensitive on its own.
INTIMATE_LABELS = frozenset(
    {
        "FEMALE_BREAST_EXPOSED",
        "FEMALE_BREAST_COVERED",
        "FEMALE_GENITALIA_EXPOSED",
        "FEMALE_GENITALIA_COVERED",
        "MALE_GENITALIA_EXPOSED",
        "MALE_BREAST_EXPOSED",
        "BUTTOCKS_EXPOSED",
        "BUTTOCKS_COVERED",
        "ANUS_EXPOSED",
        "ANUS_COVERED",
    }
)


class NudeNetIntimateContentProvider:
    """Wrap a NudeNet detector behind the ConsentGuard evidence contract."""

    name = "nudenet"

    def __init__(
        self,
        *,
        version: str = "runtime",
        model_path: str | Path | None = None,
        detector_factory: Callable[[], Any] | None = None,
        labels: frozenset[str] = INTIMATE_LABELS,
    ) -> None:
        self.version = str(version)
        self.model_path = Path(model_path) if model_path is not None else None
        self.labels = labels
        self._detector_factory = detector_factory
        self._detector: Any | None = None

    def _load(self) -> Any:
        if self._detector is not None:
            return self._detector
        if self._detector_factory is not None:
            self._detector = self._detector_factory()
            return self._detector
        if self.model_path is not None and not self.model_path.is_file():
            raise ProviderUnavailableError(f"NudeNet weights are missing: {self.model_path}")
        try:
            from nudenet import NudeDetector
        except ImportError as error:
            raise ProviderUnavailableError(
                "nudenet is not installed; install it or leave this provider off"
            ) from error
        try:
            self._detector = (
                NudeDetector(model_path=str(self.model_path)) if self.model_path else NudeDetector()
            )
        except Exception as error:  # noqa: BLE001 - any init failure is unavailability
            raise ProviderUnavailableError(f"NudeNet could not initialize: {error}") from error
        return self._detector

    def analyze(self, image: NormalizedImage) -> list[Evidence]:
        detector = self._load()
        try:
            detections = detector.detect(image.pixels_rgb)
        except Exception as error:  # noqa: BLE001 - a runtime failure is never "nothing found"
            raise ProviderUnavailableError(f"NudeNet inference failed: {error}") from error
        evidence: list[Evidence] = []
        for index, detection in enumerate(detections or []):
            label = str(detection.get("class", ""))
            if label not in self.labels:
                continue
            box = detection.get("box") or []
            if len(box) != 4:
                continue
            # NudeNet reports [x, y, width, height].
            x, y, width, height = (float(value) for value in box)
            left = max(0.0, min(float(image.width), x))
            top = max(0.0, min(float(image.height), y))
            right = max(0.0, min(float(image.width), x + width))
            bottom = max(0.0, min(float(image.height), y + height))
            if right <= left or bottom <= top:
                continue
            score = max(0.0, min(1.0, float(detection.get("score", 0.0))))
            payload = {
                "image": image.pixel_sha256,
                "index": index,
                "label": label,
                "box": [round(left, 3), round(top, 3), round(right, 3), round(bottom, 3)],
            }
            evidence.append(
                Evidence(
                    evidence_id=stable_evidence_id(self.name, self.version, payload),
                    provider=self.name,
                    provider_version=self.version,
                    privacy_class=PrivacyClass.NUDITY.value,
                    confidence=score,
                    geometry=EvidenceGeometry(
                        width=image.width,
                        height=image.height,
                        box_xyxy=(left, top, right, bottom),
                    ),
                    uncertainty_flags=("covered_region",) if label.endswith("_COVERED") else (),
                    source_detection_id=str(index),
                    sensitivity_tier="high",
                )
            )
        return evidence
