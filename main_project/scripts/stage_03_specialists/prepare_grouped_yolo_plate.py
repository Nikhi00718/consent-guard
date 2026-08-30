"""Create leakage-safe ConsentGuard plate records from an audited YOLO manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from prepare_external_specialist import CLASS_NAMES, _instance, _record, _sha256


def _rank(seed: int, group_id: str) -> str:
    return hashlib.sha256(f"{seed}:{group_id}".encode("utf-8")).hexdigest()


def _assign_groups(rows: list[dict[str, Any]], seed: int, validation_fraction: float, test_fraction: float):
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["group_id"])].append(row)
    total = len(rows)
    targets = {
        "test": round(total * test_fraction),
        "val": round(total * validation_fraction),
    }
    targets["train"] = total - targets["test"] - targets["val"]
    assigned = {name: 0 for name in targets}
    assignments: dict[str, str] = {}
    ordered = sorted(grouped.items(), key=lambda item: (-len(item[1]), _rank(seed, item[0])))
    for group_id, group_rows in ordered:
        split = min(
            targets,
            key=lambda name: (
                assigned[name] / max(1, targets[name]),
                _rank(seed, f"{group_id}:{name}"),
            ),
        )
        assignments[group_id] = split
        assigned[split] += len(group_rows)
    return assignments, assigned, targets


def _parse_instances(label_path: Path, width: int, height: int) -> list[dict[str, Any]]:
    instances = []
    for instance_id, line in enumerate(label_path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        class_id, center_x, center_y, normalized_width, normalized_height = (
            float(value) for value in line.split()
        )
        if int(class_id) != 0:
            raise ValueError(f"Unexpected class {class_id} in {label_path}")
        box_width = normalized_width * width
        box_height = normalized_height * height
        x = (center_x - normalized_width / 2) * width
        y = (center_y - normalized_height / 2) * height
        instances.append(
            _instance(
                instance_id=instance_id,
                class_name=CLASS_NAMES["plate"],
                bbox=(x, y, box_width, box_height),
            )
        )
    return instances


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    return _sha256(path)


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    project_root = Path.cwd().resolve()
    manifest_path = args.audit_manifest.resolve()
    rows = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"Audit manifest is empty: {manifest_path}")
    if any(row.get("annotation_qa") != "valid" for row in rows):
        raise ValueError("Audit manifest contains records that did not pass annotation QA")
    if not 0.05 <= args.validation_fraction <= 0.3:
        raise ValueError("validation_fraction must be in [0.05, 0.3]")
    if not 0.05 <= args.test_fraction <= 0.3:
        raise ValueError("test_fraction must be in [0.05, 0.3]")
    if args.validation_fraction + args.test_fraction > 0.5:
        raise ValueError("validation_fraction + test_fraction must not exceed 0.5")

    assignments, assigned, targets = _assign_groups(
        rows,
        args.seed,
        args.validation_fraction,
        args.test_fraction,
    )
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    converted: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    group_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        image_path = project_root / row["image_path"]
        label_path = project_root / row["label_path"]
        if _sha256(image_path) != row["image_sha256"]:
            raise RuntimeError(f"Image hash changed after audit: {image_path}")
        if _sha256(label_path) != row["label_sha256"]:
            raise RuntimeError(f"Label hash changed after audit: {label_path}")
        split = assignments[row["group_id"]]
        group_splits[row["group_id"]].add(split)
        instances = _parse_instances(label_path, int(row["width"]), int(row["height"]))
        converted[split].append(
            _record(
                image_path=image_path,
                image_id=f"roboflow-nivu-{row['image_sha256'][:20]}",
                width=int(row["width"]),
                height=int(row["height"]),
                split=split,
                source=args.source_name,
                instances=instances,
            )
        )
    leakage = {group: sorted(splits) for group, splits in group_splits.items() if len(splits) > 1}
    if leakage:
        raise RuntimeError(f"Generated split leakage: {leakage}")

    (output / "class_map.json").write_text(
        json.dumps({"background": 0, CLASS_NAMES["plate"]: 1}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summaries = {}
    for split in ("train", "val", "test"):
        path = output / f"records_{split}.jsonl"
        digest = _write_jsonl(path, converted[split])
        summaries[split] = {
            "records": len(converted[split]),
            "positive_images": sum(bool(row["instances"]) for row in converted[split]),
            "negative_images": sum(not row["instances"] for row in converted[split]),
            "instances": sum(len(row["instances"]) for row in converted[split]),
            "sha256": digest,
        }
    assignment_path = output / "group_assignments.json"
    assignment_path.write_text(json.dumps(assignments, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    result = {
        "schema_version": "grouped-external-specialist-records-v2",
        "dataset_id": args.dataset_id,
        "source_name": args.source_name,
        "audit_manifest": str(manifest_path),
        "audit_manifest_sha256": _sha256(manifest_path),
        "seed": args.seed,
        "requested_records": targets,
        "assigned_records": assigned,
        "group_count": len(assignments),
        "group_leakage": leakage,
        "test_split_used": False,
        "rectangle_masks_from_boxes": True,
        "splits": summaries,
        "group_assignments_sha256": _sha256(assignment_path),
    }
    (output / "manifest.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--source-name", required=True)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    args = parser.parse_args()
    print(json.dumps(prepare(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
