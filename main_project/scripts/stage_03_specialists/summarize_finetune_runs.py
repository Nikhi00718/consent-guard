"""Write an auditable summary of the local specialist fine-tune runs.

This report intentionally records only local, reproducible artifacts.  It does
not copy model weights into Git; the checkpoint paths and hashes are recorded so
the ignored weights can be verified on the training machine.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from consentguard.shared.paths import project_path


ROOT = project_path(".")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _artifact(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _specialist(name: str) -> dict[str, Any]:
    checkpoint_dir = ROOT / "artifacts" / "checkpoints" / f"specialist_{name}_maskrcnn_5ep"
    profile_dir = ROOT / "data" / "processed" / "specialists" / name
    result = _read_json(checkpoint_dir / "training_result.json") or {}
    bounded = _read_json(ROOT / "reports" / f"specialist_{name}_maskrcnn_5ep_val300.json") or {}
    full = _read_json(ROOT / "reports" / f"specialist_{name}_maskrcnn_5ep_val_full.json")
    return {
        "checkpoint": _artifact(checkpoint_dir / "last.pt"),
        "training_result": result,
        "profile_manifest": _read_json(profile_dir / "manifest.json"),
        "validation_300_images": bounded,
        "validation_full": full,
    }


def main() -> None:
    report = {
        "schema_version": "specialist-finetune-summary-v1",
        "source_dataset": "Visual Redactions V2 verified records",
        "test_split_used": False,
        "specialists": {
            name: _specialist(name) for name in ("face", "plate", "handwriting")
        },
        "combined_smoke": _read_json(
            ROOT / "reports" / "maskrcnn_moderate_v2_negatives_10ep_finetuned_specialists_smoke.json"
        ),
        "notes": [
            "Weights are local ignored artifacts; checkpoint hashes above provide provenance.",
            "Validation reports use only the V2 validation split; the test split remains locked.",
            "Plate and handwriting specialists remain experimental until more licensed target-domain data is admitted.",
        ],
    }
    output = ROOT / "reports" / "specialist_finetune_summary.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
