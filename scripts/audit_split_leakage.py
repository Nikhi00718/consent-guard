"""Audit exact and perceptual duplicates across processed data partitions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "processed" / "visual_redactions_verified_visual"
REPORT = ROOT / "reports" / "split_leakage_audit.json"
SPLITS = ("train2017", "val2017", "test2017")
PHASH_PART_WIDTHS = (11, 11, 11, 11, 10, 10)


def perceptual_hash(image: np.ndarray) -> int:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    coefficients = cv2.dct(resized.astype(np.float32))[:8, :8].reshape(-1)
    median = float(np.median(coefficients[1:]))
    bits = coefficients > median
    bits[0] = False
    value = 0
    for index, enabled in enumerate(bits.tolist()):
        if enabled:
            value |= 1 << index
    return value


def inspect_record(record: dict[str, Any]) -> dict[str, Any]:
    path = (ROOT / record["image_path"]).resolve()
    payload = path.read_bytes()
    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not decode image during leakage audit: {path}")
    return {
        "image_id": str(record["image_id"]),
        "split": str(record["redactions_split"]),
        "image_path": str(path.relative_to(ROOT)).replace("\\", "/"),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "phash": perceptual_hash(image),
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
    }


def phash_parts(value: int):
    shift = 0
    for index, width in enumerate(PHASH_PART_WIDTHS):
        yield index, (value >> shift) & ((1 << width) - 1)
        shift += width


def near_duplicate_pairs(rows: list[dict[str, Any]], threshold: int) -> list[dict[str, Any]]:
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    candidates: set[tuple[int, int]] = set()
    for row_index, row in enumerate(rows):
        for part in phash_parts(int(row["phash"])):
            for other_index in buckets[part]:
                if rows[other_index]["split"] != row["split"]:
                    candidates.add((other_index, row_index))
            buckets[part].append(row_index)

    result = []
    for left_index, right_index in sorted(candidates):
        left = rows[left_index]
        right = rows[right_index]
        distance = (int(left["phash"]) ^ int(right["phash"])).bit_count()
        if distance <= threshold and left["sha256"] != right["sha256"]:
            result.append(
                {
                    "left_image_id": left["image_id"],
                    "left_split": left["split"],
                    "right_image_id": right["image_id"],
                    "right_split": right["split"],
                    "phash_hamming_distance": distance,
                }
            )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--phash-distance", type=int, default=5)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--fail-on-near-duplicates", action="store_true")
    parser.add_argument("--output", type=Path, default=REPORT)
    args = parser.parse_args()
    data = args.data if args.data.is_absolute() else ROOT / args.data
    if args.workers < 1:
        parser.error("--workers must be positive")
    if not 0 <= args.phash_distance <= 5:
        parser.error("--phash-distance must be in [0, 5] for the exact multi-index guarantee")

    records: list[dict[str, Any]] = []
    records_by_split: dict[str, int] = {}
    missing_splits: list[str] = []
    for split in SPLITS:
        path = data / f"records_{split}.jsonl"
        if not path.is_file() or path.stat().st_size == 0:
            missing_splits.append(split)
            records_by_split[split] = 0
            continue
        split_records = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        records.extend(split_records)
        records_by_split[split] = len(split_records)
    if missing_splits and not args.allow_incomplete:
        raise RuntimeError(
            f"Cannot certify split leakage with missing/empty records: {missing_splits}"
        )

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        rows = list(executor.map(inspect_record, records))

    exact_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        exact_groups[row["sha256"]].append(row)
    exact_cross_split = []
    for sha256, group in sorted(exact_groups.items()):
        if len({row["split"] for row in group}) > 1:
            exact_cross_split.append(
                {
                    "sha256": sha256,
                    "images": [
                        {"image_id": row["image_id"], "split": row["split"]}
                        for row in group
                    ],
                }
            )
    near_cross_split = near_duplicate_pairs(rows, args.phash_distance)
    blocking_duplicate = bool(exact_cross_split) or (
        args.fail_on_near_duplicates and bool(near_cross_split)
    )
    certified = not missing_splits and not blocking_duplicate
    execution_passed = not blocking_duplicate and (not missing_splits or args.allow_incomplete)
    report = {
        "passed": certified,
        "audit_execution_passed": execution_passed,
        "complete_split_coverage": not missing_splits,
        "missing_splits": missing_splits,
        "records_by_split": records_by_split,
        "audited_images": len(rows),
        "exact_cross_split_duplicate_groups": exact_cross_split,
        "near_cross_split_candidates": near_cross_split,
        "phash_hamming_threshold": args.phash_distance,
        "near_duplicates_require_manual_review": bool(near_cross_split),
        "fail_on_near_duplicates": args.fail_on_near_duplicates,
        "image_pixels_or_paths_exported": False,
    }
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + f".tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(json.dumps(report, indent=2, sort_keys=True))
    if not execution_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
