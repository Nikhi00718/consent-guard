"""Run the local ConsentGuard evidence-to-policy pipeline on one still image."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from consentguard.shared.paths import project_path
from consentguard.shared.runtime import atomic_json_dump, select_device
from consentguard.stage_02_baseline_model.config import load_training_config, validate_checkpoint_inference_compatibility
from consentguard.stage_02_baseline_model.models import build_instance_segmentation_model
from consentguard.stage_03_specialists.maskrcnn import MaskRCNNEvidenceProvider
from consentguard.stage_04_fusion_calibration.domain import AssuranceStatus, ConsentState
from consentguard.stage_04_fusion_calibration.evidence import ThresholdRegistry
from consentguard.stage_05_review_export.pipeline import ReviewExportService


def _load_mask(path: Path, *, width: int, height: int) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as image:
        mask = np.asarray(image.convert("L"), dtype=np.uint8)
    if mask.shape != (height, width):
        raise ValueError(f"Approved mask must be {width}x{height}, received {mask.shape[1]}x{mask.shape[0]}")
    return mask


def _build_provider(config, checkpoint_path: Path, device: torch.device) -> MaskRCNNEvidenceProvider:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    validate_checkpoint_inference_compatibility(checkpoint, config)
    model_config = copy.deepcopy(config.section("model"))
    # Loading a checkpoint must never trigger a surprise pretrained-weight download.
    model_config["pretrained"] = False
    model_config["trainable_backbone_layers"] = 5
    data = config.section("data")
    model = build_instance_segmentation_model(
        model_config,
        num_classes=config.num_classes,
        min_size=int(data["short_side"]),
        max_size=int(data["max_long_side"]),
    )
    model.load_state_dict(checkpoint["model_state"])
    version = f"{checkpoint_path.name}:{hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()[:16]}"
    return MaskRCNNEvidenceProvider(
        model,
        device,
        class_map=config.class_map,
        version=version,
        short_side=int(data["short_side"]),
        max_long_side=int(data["max_long_side"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--approved-mask", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--threshold-profile", default="main_project/configs/stage_04_fusion_calibration/threshold_profile_candidate_v1.yaml")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--consent-state", choices=[state.value for state in ConsentState], default=ConsentState.UNKNOWN.value)
    parser.add_argument("--review-completed", action="store_true")
    parser.add_argument("--allow-unchanged", action="store_true")
    args = parser.parse_args()

    config = load_training_config(args.config, require_validation_data=False)
    input_path = project_path(args.input)
    checkpoint_path = project_path(args.checkpoint)
    device = select_device(args.device)
    provider = _build_provider(config, checkpoint_path, device)
    thresholds = ThresholdRegistry.load(project_path(args.threshold_profile))
    # Normalize once to size the optional reviewer mask; the service repeats this
    # bounded decode so its own source digest is authoritative.
    from consentguard.stage_05_review_export.ingest import normalize_image

    normalized = normalize_image(input_path)
    approved_mask = (
        _load_mask(project_path(args.approved_mask), width=normalized.width, height=normalized.height)
        if args.approved_mask
        else None
    )
    service = ReviewExportService((provider,), thresholds)
    result = service.run(
        input_path,
        consent_state=ConsentState(args.consent_state),
        review_completed=args.review_completed,
        output_path=project_path(args.output) if args.output else None,
        approved_mask=approved_mask,
        allow_unchanged=args.allow_unchanged,
    )
    report = {
        "schema_version": "analysis-report-v1",
        "input": {
            "source_sha256": result.image.source_sha256,
            "pixel_sha256": result.image.pixel_sha256,
            "width": result.image.width,
            "height": result.image.height,
            "source_format": result.image.source_format,
            "metadata_categories": list(result.image.metadata_categories),
        },
        "evidence": {
            "count": len(result.evidence.evidence),
            "ids": [item.evidence_id for item in result.evidence.evidence],
            "providers": sorted({item.provider for item in result.evidence.evidence}),
            "unavailable_providers": list(result.evidence.unavailable_providers),
        },
        "candidates": {
            "count": len(result.candidates.candidates),
            "threshold_profile_id": result.candidates.threshold_profile_id,
            "threshold_profile_release_ready": result.candidates.threshold_profile_release_ready,
            "rejected_evidence_ids": list(result.candidates.rejected_evidence_ids),
        },
        "assurance": {
            "status": result.assurance.status.value,
            "checks": [
                {"name": check.name, "status": check.status.value, "reason_code": check.reason_code}
                for check in result.assurance.checks
            ],
        },
        "decision": result.decision.to_dict(),
        "provider_errors": result.provider_errors,
        "export": result.export_report,
    }
    report_path = project_path(args.report) if args.report else (
        project_path(args.output).with_suffix(project_path(args.output).suffix + ".analysis.json")
        if args.output else input_path.with_suffix(input_path.suffix + ".analysis.json")
    )
    atomic_json_dump(report, report_path)
    print(json.dumps({"report": str(report_path), "decision": report["decision"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
