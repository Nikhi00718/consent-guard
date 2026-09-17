"""Verify output decoding, hashes, metadata, and configured attack checks."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from consentguard.stage_04_fusion_calibration.domain import AssuranceCheck, AssuranceReport, AssuranceStatus
from consentguard.stage_04_fusion_calibration.evidence.base import ProviderUnavailableError
from consentguard.stage_05_review_export.ingest import normalize_image


@dataclass(frozen=True)
class RenderedAsset:
    path: Path
    expected_width: int
    expected_height: int
    export_report: dict
    attack_checks: dict[str, AssuranceStatus] = field(default_factory=dict)


class AssuranceService:
    """Fail uncertain when a required independent attacker has not run."""

    def __init__(
        self,
        required_attack_checks: tuple[str, ...] = ("ocr", "barcode", "face", "plate"),
    ) -> None:
        self.required_attack_checks = required_attack_checks

    def run_attack_checks(
        self,
        asset_path: str | Path,
        providers: tuple[object, ...],
        *,
        thresholds: object | None = None,
    ) -> dict[str, AssuranceStatus]:
        """Run configured residual-content detectors on the freshly encoded asset.

        A detector finding is a failed residual-content attack.  A missing
        provider or runtime error is uncertain, never a pass.  Provider names
        are mapped to the four assurance categories without storing OCR text.
        This is a practical local attack harness; independent attack models
        should still be supplied before a production release.

        ``thresholds`` is the same versioned profile the detection path uses.
        Without it, every check fails on every image: these detectors always
        emit low-confidence noise (LPD-YuNet needs 0.8 to be believed), so an
        unfiltered finding count says nothing about residual content and the
        warning becomes something the reviewer learns to ignore.
        """

        image = normalize_image(asset_path)
        aliases = {
            "ocr": {"paddleocr", "paddleocr_onnx"},
            "barcode": {"zxingcpp"},
            "face": {"face_maskrcnn", "yunet"},
            "plate": {"plate_maskrcnn", "plate_yunet"},
        }
        by_name = {str(getattr(provider, "name", "")): provider for provider in providers}
        results: dict[str, AssuranceStatus] = {}
        for attack_name in self.required_attack_checks:
            provider = next((by_name[name] for name in aliases.get(attack_name, set()) if name in by_name), None)
            if provider is None:
                results[attack_name] = AssuranceStatus.UNCERTAIN
                continue
            try:
                findings = provider.analyze(image)
            except ProviderUnavailableError:
                results[attack_name] = AssuranceStatus.UNCERTAIN
            except Exception:
                results[attack_name] = AssuranceStatus.FAIL
            else:
                credible = self._credible(findings, thresholds)
                results[attack_name] = AssuranceStatus.FAIL if credible else AssuranceStatus.PASS
        return results

    @staticmethod
    def _credible(findings: list, thresholds: object | None) -> list:
        """Keep only findings the detection profile would have acted on."""

        if thresholds is None:
            return list(findings)
        credible = []
        for finding in findings:
            try:
                rule = thresholds.get(finding.provider, finding.privacy_class)  # type: ignore[attr-defined]
            except (KeyError, AttributeError):
                credible.append(finding)  # No rule: stay conservative.
                continue
            if finding.confidence >= rule.score_threshold:
                credible.append(finding)
        return credible

    def inspect(self, asset: RenderedAsset) -> AssuranceReport:
        checks = [self._decode_check(asset), self._hash_check(asset), self._metadata_check(asset)]
        for name in self.required_attack_checks:
            status = asset.attack_checks.get(name, AssuranceStatus.NOT_RUN)
            checks.append(
                AssuranceCheck(
                    name=f"attack_{name}",
                    status=status,
                    reason_code=(
                        f"{name.upper()}_ATTACK_PASSED"
                        if status is AssuranceStatus.PASS
                        else f"{name.upper()}_ATTACK_{status.value}"
                    ),
                )
            )
        return AssuranceReport(tuple(checks))

    @staticmethod
    def _decode_check(asset: RenderedAsset) -> AssuranceCheck:
        try:
            with Image.open(asset.path) as image:
                image.load()
                valid = image.size == (asset.expected_width, asset.expected_height)
        except (FileNotFoundError, UnidentifiedImageError, OSError):
            valid = False
        return AssuranceCheck(
            name="pixel_decode",
            status=AssuranceStatus.PASS if valid else AssuranceStatus.FAIL,
            reason_code="PIXEL_DECODE_VERIFIED" if valid else "PIXEL_DECODE_FAILED",
        )

    @staticmethod
    def _hash_check(asset: RenderedAsset) -> AssuranceCheck:
        expected = str(asset.export_report.get("output_sha256", ""))
        actual = hashlib.sha256(asset.path.read_bytes()).hexdigest() if asset.path.is_file() else ""
        newly_encoded = asset.export_report.get("newly_encoded") is True
        valid = bool(expected) and actual == expected and newly_encoded
        return AssuranceCheck(
            name="output_hash",
            status=AssuranceStatus.PASS if valid else AssuranceStatus.FAIL,
            reason_code="OUTPUT_HASH_VERIFIED" if valid else "OUTPUT_HASH_OR_ENCODING_FAILED",
        )

    @staticmethod
    def _metadata_check(asset: RenderedAsset) -> AssuranceCheck:
        metadata_keys: list[str] = []
        try:
            with Image.open(asset.path) as image:
                if image.getexif():
                    metadata_keys.append("exif")
                metadata_keys.extend(
                    key for key in ("xmp", "comment", "icc_profile") if key in image.info
                )
        except (FileNotFoundError, UnidentifiedImageError, OSError):
            return AssuranceCheck(
                name="metadata",
                status=AssuranceStatus.FAIL,
                reason_code="METADATA_INSPECTION_FAILED",
            )
        clean = not metadata_keys
        return AssuranceCheck(
            name="metadata",
            status=AssuranceStatus.PASS if clean else AssuranceStatus.FAIL,
            reason_code="METADATA_ABSENT" if clean else "METADATA_PRESENT",
            details={"categories": sorted(set(metadata_keys))},
        )
