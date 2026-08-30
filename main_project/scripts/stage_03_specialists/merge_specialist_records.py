"""Merge audited specialist record files without cross-split image leakage."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(paths: list[Path]) -> list[tuple[dict[str, Any], str]]:
    rows = []
    for source in paths:
        resolved = source.resolve()
        for line in resolved.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append((json.loads(line), str(resolved)))
    return rows


def _deduplicate(
    candidates: list[tuple[dict[str, Any], str]],
    split: str,
    excluded_hashes: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, str]], set[str]]:
    selected: dict[str, tuple[dict[str, Any], str]] = {}
    dropped: list[dict[str, str]] = []
    for row, source in candidates:
        image_path = Path(row["image_path"])
        if not image_path.is_file():
            raise FileNotFoundError(f"Record image is missing: {image_path}")
        digest = _sha256(image_path)
        if digest in excluded_hashes:
            dropped.append({"image_path": str(image_path), "reason": "present_in_higher_priority_split", "source": source})
            continue
        existing = selected.get(digest)
        if existing is not None:
            existing_row, existing_source = existing
            if len(row.get("instances", [])) > len(existing_row.get("instances", [])):
                dropped.append({"image_path": existing_row["image_path"], "reason": "duplicate_with_fewer_instances", "source": existing_source})
                selected[digest] = (row, source)
            else:
                dropped.append({"image_path": str(image_path), "reason": "duplicate_image", "source": source})
            continue
        selected[digest] = (row, source)
    output = []
    for digest, (row, _source) in sorted(selected.items(), key=lambda item: (item[1][0]["image_id"], item[0])):
        row = dict(row)
        row["image_split"] = split
        row["redactions_split"] = split
        row["image_sha256"] = digest
        output.append(row)
    return output, dropped, set(selected)


def _write(path: Path, rows: list[dict[str, Any]]) -> str:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    return _sha256(path)


def merge(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    validation, validation_dropped, validation_hashes = _deduplicate(
        _load(args.val_records),
        "val",
        set(),
    )
    train, train_dropped, train_hashes = _deduplicate(
        _load(args.train_records),
        "train",
        validation_hashes,
    )
    overlap = train_hashes & validation_hashes
    if overlap:
        raise RuntimeError(f"Cross-split image leakage remains: {len(overlap)} hashes")
    class_map = {"background": 0, args.class_name: 1}
    class_map_path = output / "class_map.json"
    class_map_path.write_text(json.dumps(class_map, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summaries = {}
    for split, rows in (("train", train), ("val", validation)):
        path = output / f"records_{split}.jsonl"
        summaries[split] = {
            "records": len(rows),
            "positive_images": sum(bool(row.get("instances")) for row in rows),
            "negative_images": sum(not row.get("instances") for row in rows),
            "instances": sum(len(row.get("instances", [])) for row in rows),
            "sha256": _write(path, rows),
        }
    result = {
        "schema_version": "merged-specialist-records-v1",
        "class_name": args.class_name,
        "train_sources": [str(path.resolve()) for path in args.train_records],
        "validation_sources": [str(path.resolve()) for path in args.val_records],
        "splits": summaries,
        "dropped": train_dropped + validation_dropped,
        "cross_split_hash_leakage": 0,
        "class_map_sha256": _sha256(class_map_path),
        "test_split_used": False,
    }
    (output / "manifest.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-records", type=Path, action="append", required=True)
    parser.add_argument("--val-records", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--class-name", default="a108_license_plate_all")
    args = parser.parse_args()
    print(json.dumps(merge(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
