"""Build one-class specialist records from the verified Visual Redactions split.

The source records and the locked test split are never modified.  Each output
profile keeps every source image (including negatives), filters instances to one
privacy class, and remaps the selected class to label 1 for a one-class
Mask R-CNN fine-tune.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from consentguard.shared.paths import project_path


def _read_records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n" for record in records)
    path.write_text(payload, encoding="utf-8", newline="\n")
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _transform(
    records: list[dict[str, Any]],
    *,
    source_class_id: int,
    negative_multiplier: float | None = None,
    seed: int = 1337,
) -> tuple[list[dict[str, Any]], int]:
    transformed: list[dict[str, Any]] = []
    instances = 0
    for source in records:
        record = dict(source)
        selected: list[dict[str, Any]] = []
        for instance in source.get("instances", []):
            if int(instance["class_id"]) != source_class_id:
                continue
            item = dict(instance)
            item["class_id"] = 1
            selected.append(item)
        record["instances"] = selected
        record["specialist_source_class_id"] = source_class_id
        record["specialist_negative"] = not bool(selected)
        transformed.append(record)
        instances += len(selected)
    if negative_multiplier is not None:
        if negative_multiplier < 0:
            raise ValueError("negative_multiplier must be non-negative")
        positives = [record for record in transformed if not record["specialist_negative"]]
        negatives = [record for record in transformed if record["specialist_negative"]]
        keep_negatives = min(len(negatives), int(len(positives) * negative_multiplier))
        rng = random.Random(seed)
        negatives = sorted(negatives, key=lambda record: str(record["image_id"]))
        rng.shuffle(negatives)
        transformed = positives + negatives[:keep_negatives]
        transformed.sort(key=lambda record: str(record["image_id"]))
    return transformed, instances


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-records", required=True, type=Path)
    parser.add_argument("--val-records", required=True, type=Path)
    parser.add_argument("--source-class-map", required=True, type=Path)
    parser.add_argument("--class-name", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--train-negative-multiplier", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    source_map = json.loads(project_path(args.source_class_map).read_text(encoding="utf-8"))
    if args.class_name not in source_map or args.class_name == "background":
        raise ValueError(f"Unknown foreground class: {args.class_name}")
    source_class_id = int(source_map[args.class_name])
    train_source_path = project_path(args.train_records)
    val_source_path = project_path(args.val_records)
    train_source = _read_records(train_source_path)
    val_source = _read_records(val_source_path)
    train_records, train_instances = _transform(
        train_source,
        source_class_id=source_class_id,
        negative_multiplier=args.train_negative_multiplier,
        seed=args.seed,
    )
    val_records, val_instances = _transform(val_source, source_class_id=source_class_id)

    output_dir = project_path(args.output_dir)
    train_path = output_dir / "records_train.jsonl"
    val_path = output_dir / "records_val.jsonl"
    train_hash = _write_jsonl(train_path, train_records)
    val_hash = _write_jsonl(val_path, val_records)
    class_map_path = output_dir / "class_map.json"
    class_map_path.write_text(
        json.dumps({"background": 0, args.class_name: 1}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    source_hashes = {
        "train_records": hashlib.sha256(train_source_path.read_bytes()).hexdigest(),
        "val_records": hashlib.sha256(val_source_path.read_bytes()).hexdigest(),
        "source_class_map": hashlib.sha256(project_path(args.source_class_map).read_bytes()).hexdigest(),
    }
    manifest = {
        "schema_version": "specialist-records-v1",
        "class_name": args.class_name,
        "source_class_id": source_class_id,
        "train_negative_multiplier": args.train_negative_multiplier,
        "seed": args.seed,
        "source_records": source_hashes,
        "train": {
            "records": len(train_records),
            "positive_instances": train_instances,
            "negative_images": sum(1 for record in train_records if record["specialist_negative"]),
            "records_path": str(train_path),
            "sha256": train_hash,
        },
        "validation": {
            "records": len(val_records),
            "positive_instances": val_instances,
            "negative_images": sum(1 for record in val_records if record["specialist_negative"]),
            "records_path": str(val_path),
            "sha256": val_hash,
        },
        "test_split_used": False,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
