"""Audit available Visual Redactions images against their annotations."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifests" / "visual_redactions_master.jsonl"
REPORTS = ROOT / "reports"


def main() -> None:
    rows = [json.loads(line) for line in MANIFEST.read_text(encoding="utf-8").splitlines()]
    available = [row for row in rows if row["image_path"]]
    dimension_pairs: Counter[str] = Counter()
    bad_decode: list[str] = []
    annotation_mismatch: list[str] = []
    attr_counts: Counter[str] = Counter()
    instance_counts: Counter[str] = Counter()

    for row in available:
        image_path = ROOT / row["image_path"]
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            bad_decode.append(row["image_id"])
            continue
        height, width = image.shape[:2]
        dimension_pairs[f"{width}x{height} -> {row['image_width']}x{row['image_height']}"] += 1
        if width <= 0 or height <= 0:
            annotation_mismatch.append(row["image_id"])
        for attr_id in row["attribute_ids"]:
            attr_counts[str(attr_id)] += 1
        instance_counts[str(row["instance_count"])] += 1

    report = {
        "manifest_rows": len(rows),
        "available_rows": len(available),
        "missing_rows": len(rows) - len(available),
        "decoded_images": len(available) - len(bad_decode),
        "decode_failures": bad_decode,
        "invalid_dimensions": annotation_mismatch,
        "dimension_pairs": dict(dimension_pairs.most_common()),
        "attribute_image_counts": dict(attr_counts.most_common()),
        "instance_count_distribution": dict(instance_counts),
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    output = REPORTS / "phase1_dataset_audit.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
