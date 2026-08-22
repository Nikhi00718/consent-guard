"""Validate provenance, grouping, licensing, and split rules for target-domain data."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
REQUIRED = {"image_id", "image_path", "source_url", "license", "sha256", "domain", "split", "group_id", "privacy_classes", "difficult_conditions"}


def validate_manifest(path: Path, *, require_release_size: bool = False) -> dict:
    errors: list[str] = []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids: set[str] = set()
    hashes: set[str] = set()
    group_splits: defaultdict[str, set[str]] = defaultdict(set)
    domains: Counter[str] = Counter()
    negatives = 0
    difficult = 0
    for number, row in enumerate(rows, start=1):
        missing = REQUIRED - set(row)
        if missing:
            errors.append(f"line {number}: missing {sorted(missing)}")
            continue
        image_id, digest = str(row["image_id"]), str(row["sha256"]).lower()
        if image_id in ids or digest in hashes:
            errors.append(f"line {number}: duplicate image_id or sha256")
        ids.add(image_id); hashes.add(digest)
        if row["domain"] not in {"general", "india"} or row["split"] not in {"train", "val", "test"}:
            errors.append(f"line {number}: invalid domain or split")
        if not row["source_url"] or not row["license"] or row["license"] == "unknown":
            errors.append(f"line {number}: source and verified license are required")
        image_path = Path(row["image_path"])
        image_path = image_path if image_path.is_absolute() else ROOT / image_path
        if not image_path.is_file():
            errors.append(f"line {number}: image file missing")
        elif hashlib.sha256(image_path.read_bytes()).hexdigest() != digest:
            errors.append(f"line {number}: sha256 mismatch")
        group_splits[str(row["group_id"])].add(str(row["split"]))
        domains[str(row["domain"])] += 1
        negatives += not bool(row["privacy_classes"])
        difficult += bool(row["difficult_conditions"])
    for group, splits in group_splits.items():
        if len(splits) > 1:
            errors.append(f"group {group!r} crosses splits: {sorted(splits)}")
    if require_release_size:
        if len(rows) < 2000 or domains["general"] < 1000 or domains["india"] < 1000:
            errors.append("release manifest requires 2,000 images: 1,000 general and 1,000 India")
        if rows and (negatives / len(rows) < 0.25 or difficult / len(rows) < 0.25):
            errors.append("release manifest requires at least 25% negative and 25% difficult images")
    return {"rows": len(rows), "domains": dict(domains), "negative_images": negatives, "difficult_images": difficult, "errors": errors, "passed": bool(rows) and not errors}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--require-release-size", action="store_true")
    args = parser.parse_args()
    result = validate_manifest(args.manifest, require_release_size=args.require_release_size)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
