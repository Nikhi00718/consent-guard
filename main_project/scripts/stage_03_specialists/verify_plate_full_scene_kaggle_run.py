"""Verify the completed full-scene plate Kaggle run and promotion decision."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUN_ROOT = (
    ROOT
    / "artifacts/kaggle/remote-runs/plate-full-scene-v4/consentguard/artifacts/checkpoints"
    / "specialist_plate_full_scene_research_v1_highres_5ep"
)
DEFAULT_TRANSPORT_ROOT = ROOT / "artifacts/kaggle/consentguard-plate-full-scene-v1"
DEFAULT_DEEPAK_REPORT = ROOT / "reports/plate_full_scene_v4_deepak_challenge.json"
DEFAULT_VALIDATION_REPORT = ROOT / "reports/plate_full_scene_v4_merged_validation.json"
DEFAULT_OUTPUT = ROOT / "reports/plate_full_scene_kaggle_v4_verification.json"
CANDIDATE_LABEL = "kaggle_full_scene_v4"
CURRENT_LABEL = "current_india"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _candidate(report: dict[str, Any], label: str) -> dict[str, Any]:
    matches = [candidate for candidate in report["candidates"] if candidate["label"] == label]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one candidate {label!r}, found {len(matches)}")
    return matches[0]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--transport-root", type=Path, default=DEFAULT_TRANSPORT_ROOT)
    parser.add_argument("--deepak-report", type=Path, default=DEFAULT_DEEPAK_REPORT)
    parser.add_argument("--validation-report", type=Path, default=DEFAULT_VALIDATION_REPORT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    run_root = args.run_root.resolve(strict=True)
    transport_root = args.transport_root.resolve(strict=True)
    checkpoint = run_root / "best.pt"
    environment = _json(run_root / "environment.json")
    initialization = _json(run_root / "initialization.json")
    resolved_config = _json(run_root / "resolved_config.json")
    training_result = _json(run_root / "training_result.json")
    transport = _json(transport_root / "transport_manifest.json")
    deepak = _json(args.deepak_report.resolve(strict=True))
    validation = _json(args.validation_report.resolve(strict=True))

    _require(transport["test_split_used"] is False, "Transport used a test split")
    _require(transport["cross_split_hash_leakage"] == 0, "Transport has cross-split leakage")
    record_paths = {
        split: transport_root
        / "data/processed/external/plate_full_scene_research_v1"
        / f"records_{split}.jsonl"
        for split in ("train", "val")
    }
    actual_record_hashes = {split: _sha256(path) for split, path in record_paths.items()}
    _require(actual_record_hashes == transport["record_sha256"], "Transport record hash mismatch")

    init_checkpoint = transport_root / transport["initialization_checkpoint"]
    init_hash = _sha256(init_checkpoint)
    _require(init_hash == transport["initialization_checkpoint_sha256"], "Transport init hash mismatch")
    _require(
        initialization["initialization_checkpoint_sha256"] == init_hash,
        "Training initialization hash does not match transport",
    )
    _require(
        initialization["initialization_checkpoint_bytes"] == init_checkpoint.stat().st_size,
        "Training initialization size does not match transport",
    )

    checkpoint_hash = _sha256(checkpoint)
    deepak_candidate = _candidate(deepak, CANDIDATE_LABEL)
    validation_candidate = _candidate(validation, CANDIDATE_LABEL)
    _require(
        deepak_candidate["checkpoint_sha256"] == checkpoint_hash,
        "Deepak evaluation used a different candidate checkpoint",
    )
    _require(
        validation_candidate["checkpoint_sha256"] == checkpoint_hash,
        "Merged validation used a different candidate checkpoint",
    )

    deepak_current = _candidate(deepak, CURRENT_LABEL)["metrics"]["0.50"]
    deepak_metrics = deepak_candidate["metrics"]["0.50"]
    validation_current = _candidate(validation, CURRENT_LABEL)["metrics"]["0.50"]
    validation_metrics = validation_candidate["metrics"]["0.50"]
    checks = {
        "kaggle_complete": True,
        "test_split_locked": transport["test_split_used"] is False,
        "cross_split_hash_leakage_zero": transport["cross_split_hash_leakage"] == 0,
        "merged_validation_recall": validation_metrics["recall"] >= 0.6832,
        "merged_validation_fppi": validation_metrics["false_positives_per_image"] <= 0.6557,
        "deepak_recall": deepak_metrics["recall"] >= 0.50,
        "deepak_fppi": deepak_metrics["false_positives_per_image"] <= 1.0233,
    }
    failed_checks = [name for name, passed in checks.items() if not passed]
    summary = {
        "schema_version": "consentguard-plate-kaggle-verification-v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "kaggle": {
            "kernel": "nikhil00718/consentguard-plate-full-scene-training",
            "kernel_version": 4,
            "observed_status": "COMPLETE",
        },
        "transport": {
            "manifest": str((transport_root / "transport_manifest.json").relative_to(ROOT)),
            "manifest_sha256": _sha256(transport_root / "transport_manifest.json"),
            "records": transport["records"],
            "record_sha256": actual_record_hashes,
            "unique_images": transport["unique_images"],
            "image_bytes": transport["image_bytes"],
            "cross_split_hash_leakage": transport["cross_split_hash_leakage"],
            "test_split_used": transport["test_split_used"],
        },
        "training": {
            "checkpoint": str(checkpoint.relative_to(ROOT)),
            "checkpoint_bytes": checkpoint.stat().st_size,
            "checkpoint_sha256": checkpoint_hash,
            "initialization_checkpoint_sha256": init_hash,
            "environment": environment,
            "resolved_experiment": resolved_config["experiment"],
            "resolved_model": resolved_config["model"],
            "resolved_training": resolved_config["training"],
            "best_map": training_result["best_map"],
            "completed_epoch": training_result["completed_epoch"],
            "global_step": training_result["global_step"],
            "last_evaluation": training_result["last_evaluation"],
            "cuda_memory": training_result["cuda_memory"],
        },
        "score_threshold": 0.50,
        "iou_threshold": 0.50,
        "evaluation": {
            "merged_validation": {
                "images": validation["images"],
                "ground_truth_boxes": validation["ground_truth_boxes"],
                "current": validation_current,
                "candidate": validation_metrics,
            },
            "deepak_vid1": {
                "images": deepak["images"],
                "ground_truth_boxes": deepak["ground_truth_boxes"],
                "current": deepak_current,
                "candidate": deepak_metrics,
            },
        },
        "promotion_gate": {
            "requirements": {
                "merged_validation_min_recall": 0.6832,
                "merged_validation_max_fppi": 0.6557,
                "deepak_vid1_min_recall": 0.50,
                "deepak_vid1_max_fppi": 1.0233,
            },
            "checks": checks,
            "failed_checks": failed_checks,
            "website_promotion_allowed": not failed_checks,
            "decision": "retain_current_website_default" if failed_checks else "promote_candidate",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary["promotion_gate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
