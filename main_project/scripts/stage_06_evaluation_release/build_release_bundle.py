"""Build a reproducible, explicitly non-release ConsentGuard bundle manifest.

The bundle manifest is useful even when gates fail: it records exactly which
code, data, checkpoints, thresholds, and reports were evaluated, plus the
remaining blockers.  Large local checkpoints are hashed but never copied into
Git or silently uploaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from consentguard.shared.paths import project_path
from consentguard.stage_06_evaluation_release.release_gates import evaluate_release_gates


ROOT = project_path(".")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path: str, *, required: bool = True) -> dict[str, Any]:
    resolved = project_path(path)
    if not resolved.is_file():
        return {"path": str(resolved), "required": required, "present": False}
    return {
        "path": str(resolved),
        "required": required,
        "present": True,
        "bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _git(command: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *command], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _blocked_metrics() -> dict[str, Any]:
    """Return an intentionally incomplete gate input; missing evidence must fail closed."""

    return {
        "schema_version": "release-metrics-v1",
        "confidence_level": 0.95,
        "baseline_mask_map": 0.2083981845491433,
        "seed_count": 1,
        "assurance_fail_closed": False,
        "supported_classes": [
            "face",
            "license_plate",
            "person_body",
            "nudity",
            "handwriting",
            "disability",
            "medicine",
            "fingerprint",
            "signature",
        ],
    }


def build(output: Path) -> dict[str, Any]:
    gate_input = _blocked_metrics()
    gate_result = evaluate_release_gates(gate_input)
    gate_input_path = output.parent / "release_metrics_v1_incomplete.json"
    gate_result_path = output.parent / "release_gates_v1_blocked.json"
    gate_input_path.write_text(json.dumps(gate_input, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    gate_result_path.write_text(json.dumps(gate_result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checkpoint_specs = {
        "global_maskrcnn": "artifacts/checkpoints/maskrcnn_moderate_v2_negatives_10ep/last.pt",
        "face_maskrcnn": "artifacts/checkpoints/specialist_face_maskrcnn_5ep/last.pt",
        "plate_fasterrcnn_india": "artifacts/checkpoints/specialist_plate_ccpd2020_india_finetune_5ep/best.pt",
        "handwriting_maskrcnn": "artifacts/checkpoints/specialist_handwriting_maskrcnn_5ep/last.pt",
    }
    artifacts = {
        "generation_base_commit": _git(["rev-parse", "HEAD"]),
        "data": [
            _artifact("reports/visual_redactions_release_validation.json"),
            _artifact("reports/processed_records_v2_negatives_validation.json"),
            _artifact("reports/same_release_split_leakage_audit_clean.json"),
            _artifact("reports/roboflow_nivu_indian_plate_v1_audit.json"),
            _artifact("data/manifests/roboflow_nivu_indian_plate_v1.jsonl"),
            _artifact("data/processed/visual_redactions_verified_visual_v2_negatives/class_map.json"),
            _artifact("data/processed/visual_redactions_verified_visual_v2_negatives/records_train2017.jsonl"),
            _artifact("data/processed/visual_redactions_verified_visual_v2_negatives/records_val2017.jsonl"),
        ],
        "configs": [
            _artifact("main_project/configs/stage_02_baseline_model/train_maskrcnn_moderate_v2_negatives_10ep.yaml"),
            _artifact("main_project/configs/stage_04_fusion_calibration/threshold_profile_v2_validation_calibrated.yaml"),
            _artifact("main_project/configs/stage_03_specialists/train_face_maskrcnn_5ep.yaml"),
            _artifact("main_project/configs/stage_03_specialists/train_plate_ccpd2020_india_finetune_5ep.yaml"),
            _artifact("main_project/configs/stage_03_specialists/train_handwriting_maskrcnn_5ep.yaml"),
        ],
        "reports": [
            _artifact("reports/maskrcnn_moderate_v2_negatives_10ep_last_eval.json"),
            _artifact("reports/maskrcnn_moderate_v2_negatives_10ep_threshold_calibration.json"),
            _artifact("reports/specialist_finetune_summary.json"),
            _artifact("reports/fused_validation_v2_1_baseline.json", required=False),
            _artifact("reports/maskrcnn_moderate_v2_negatives_10ep_finetuned_specialists_smoke.json"),
            _artifact("reports/plate_deepak_current_challenge.json"),
            _artifact("reports/plate_full_scene_current_validation.json"),
            _artifact("reports/PROJECT_AUDIT_AND_EXECUTION_PLAN_2026-08-30.md"),
        ],
        "checkpoints": {
            name: {**_artifact(path), "tracked_in_git": False, "storage_note": "local ignored artifact; hash is recorded for reproducibility"}
            for name, path in checkpoint_specs.items()
        },
        "provider_assets": [_artifact("artifacts/specialists/opencv_zoo/MANIFEST.json")],
    }
    manifest = {
        "schema_version": "consentguard-release-bundle-v1",
        "bundle_status": "VALIDATION_ONLY_BLOCKED",
        "release_candidate": False,
        "repository": {
            "root": str(ROOT),
            "branch": _git(["branch", "--show-current"]),
            "generation_base_commit": _git(["rev-parse", "HEAD"]),
            "commit_relation": (
                "This manifest is generated content, so the commit containing it is expected "
                "to be a descendant of generation_base_commit."
            ),
        },
        "test_split_used": False,
        "artifacts": artifacts,
        "gate_input": gate_input,
        "gate_result": gate_result,
        "gate_input_path": str(gate_input_path),
        "gate_result_path": str(gate_result_path),
        "blockers": [
            "No licensed Target-2K general/India manifest, pixels, and annotations are admitted.",
            "Only one trained seed is present for the current bundle; the release contract requires at least three.",
            "The global V2 validation segmentation mAP is below the preservation gate.",
            "Fused validation does not yet contain general/India domain recall or full-bundle leakage/FPR confidence bounds.",
            "Independent OCR, barcode, face, and plate residual-content attacks are not implemented as passing checks.",
            "The threshold profile is explicitly release_ready: false; plate and handwriting specialists remain experimental.",
        ],
        "next_safe_actions": [
            "Acquire and admit only licensed Target-2K data with image-level rights and a frozen manifest.",
            "Run the fused evaluator on the full V2 validation split and target-domain splits.",
            "Train and evaluate three controlled seeds per release candidate.",
            "Implement independent output attackers and archive their reports before opening the locked test once.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("reports/release_bundle_manifest.json"))
    args = parser.parse_args()
    manifest = build(project_path(args.output))
    print(json.dumps({"output": str(project_path(args.output)), "release_candidate": manifest["release_candidate"], "blockers": len(manifest["blockers"])}, indent=2))


if __name__ == "__main__":
    main()
