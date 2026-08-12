"""Remove confirmed train-side duplicate scenes while preserving official raw splits."""

from __future__ import annotations

import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "visual_redactions_verified_visual"
REPORT = ROOT / "reports" / "cross_split_leakage_quarantine.json"

# Validation/test are preserved as evaluation references. These training records
# duplicate or near-duplicate their pixels and must not be used for learning.
POLICY_PATH = ROOT / "configs" / "cross_split_leakage_quarantine.json"
POLICY = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
QUARANTINED_TRAIN_IDS = {str(key): str(value) for key, value in POLICY["train2017"].items()}


def atomic_write(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    train_path = DATA / "records_train2017.jsonl"
    rows = [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines() if line]
    removed = [row for row in rows if str(row["image_id"]) in QUARANTINED_TRAIN_IDS]
    kept = [row for row in rows if str(row["image_id"]) not in QUARANTINED_TRAIN_IDS]
    found = {str(row["image_id"]) for row in removed}
    missing = sorted(set(QUARANTINED_TRAIN_IDS) - found)
    if missing:
        raise RuntimeError(f"Expected leakage records were not present in train data: {missing}")
    atomic_write(
        train_path,
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in kept),
    )
    report = {
        "policy": "preserve validation/test; remove duplicate train-side records",
        "source_train_records": len(rows),
        "clean_train_records": len(kept),
        "removed_train_records": [
            {"image_id": image_id, "reason": QUARANTINED_TRAIN_IDS[image_id]}
            for image_id in sorted(QUARANTINED_TRAIN_IDS)
        ],
        "raw_release_modified": False,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(REPORT, json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
