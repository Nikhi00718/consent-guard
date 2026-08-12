"""Create same-release model-ready Visual Redactions records without altering raw data.

The released images are resized versions of the dimensions stored in the
annotations. This script scales polygons and boxes to the decoded image size,
validates them, and writes compact JSONL records for the model loader.
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifests" / "visual_redactions_master.jsonl"
RAW_ANNOTATIONS = ROOT / "data" / "raw" / "visual_redactions"
OUTPUT = ROOT / "data" / "processed" / "visual_redactions"


def write_json_atomic(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_jsonl_atomic(path: Path, records: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def scaled_polygon(polygon: list[float], sx: float, sy: float, width: int, height: int):
    if len(polygon) < 6 or len(polygon) % 2:
        raise ValueError("polygon must contain at least three x/y pairs")
    values = []
    for index in range(0, len(polygon), 2):
        x = float(polygon[index])
        y = float(polygon[index + 1])
        if not math.isfinite(x) or not math.isfinite(y):
            raise ValueError("polygon contains a non-finite coordinate")
        values.extend(
            [
                round(clamp(x * sx, 0.0, width - 1.0), 3),
                round(clamp(y * sy, 0.0, height - 1.0), 3),
            ]
        )
    return values


def scaled_bbox(bbox: list[float], sx: float, sy: float, width: int, height: int):
    if len(bbox) != 4:
        raise ValueError("bbox must contain x, y, width, height")
    x, y, box_width, box_height = (float(value) for value in bbox)
    if not all(math.isfinite(value) for value in (x, y, box_width, box_height)):
        raise ValueError("bbox contains a non-finite coordinate")
    left = clamp(x * sx, 0.0, width - 1.0)
    top = clamp(y * sy, 0.0, height - 1.0)
    right = clamp((x + box_width) * sx, left, float(width))
    bottom = clamp((y + box_height) * sy, top, float(height))
    return [round(left, 3), round(top, 3), round(right - left, 3), round(bottom - top, 3)]


def polygon_area(polygon: list[float]) -> float:
    points = list(zip(polygon[0::2], polygon[1::2]))
    return abs(
        sum(
            points[index][0] * points[(index + 1) % len(points)][1]
            - points[(index + 1) % len(points)][0] * points[index][1]
            for index in range(len(points))
        )
        / 2.0
    )


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest_rows = [
        json.loads(line) for line in MANIFEST.read_text(encoding="utf-8").splitlines()
    ]
    annotations = {}
    for split in ("train2017", "val2017", "test2017"):
        path = RAW_ANNOTATIONS / f"{split}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        annotations.update(
            {image_id: (split, record) for image_id, record in payload["annotations"].items()}
        )

    class_ids = sorted(
        {
            str(attr_id)
            for _split, record in annotations.values()
            for attr in record.get("attributes", [])
            for attr_id in [attr.get("attr_id")]
            if attr_id is not None
        }
    )
    class_map = {"background": 0}
    class_map.update({attr_id: index for index, attr_id in enumerate(class_ids, start=1)})
    write_json_atomic(OUTPUT / "class_map.json", class_map)

    records_by_split: defaultdict[str, list[dict]] = defaultdict(list)
    pending: list[dict] = []
    errors: list[dict] = []
    instance_counts: Counter[str] = Counter()
    attribute_counts: Counter[str] = Counter()
    resize_pairs: Counter[str] = Counter()
    excluded_instances: Counter[str] = Counter()

    for row in manifest_rows:
        if not row["image_path"]:
            pending.append({"image_id": row["image_id"], "redactions_split": row["redactions_split"]})
            continue

        image_path = ROOT / row["image_path"]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            errors.append({"image_id": row["image_id"], "error": "decode_failed"})
            continue
        height, width = image.shape[:2]
        source_width = int(row["image_width"])
        source_height = int(row["image_height"])
        sx = width / source_width
        sy = height / source_height
        resize_pairs[f"{width}x{height}<-{source_width}x{source_height}"] += 1

        _annotation_split, annotation = annotations[row["image_id"]]
        instances = []
        excluded_for_image = []
        for attr in annotation.get("attributes", []):
            attr_id = str(attr.get("attr_id", "unknown"))
            try:
                attr_id = str(attr["attr_id"])
                polygons = [
                    scaled_polygon(polygon, sx, sy, width, height)
                    for polygon in attr.get("polygons", [])
                ]
                if not polygons:
                    raise ValueError("instance has no polygon")
                if not any(polygon_area(polygon) > 0 for polygon in polygons):
                    raise ValueError("instance has zero-area polygon")
                bbox = scaled_bbox(attr["bbox"], sx, sy, width, height)
                if bbox[2] <= 0 or bbox[3] <= 0:
                    raise ValueError("instance has zero-area bbox")
                instance = {
                    "instance_id": int(attr["instance_id"]),
                    "class_id": class_map[attr_id],
                    "attr_id": attr_id,
                    "bbox": bbox,
                    "polygons": polygons,
                    "iscrowd": bool(attr.get("iscrowd", False)),
                    "area_pixels": round(float(attr.get("area", 0.0)) * sx * sy, 3),
                }
                instances.append(instance)
                attribute_counts[attr_id] += 1
            except (KeyError, TypeError, ValueError) as error:
                reason = str(error)
                excluded_instances[reason] += 1
                excluded_for_image.append(
                    {
                        "instance_id": attr.get("instance_id"),
                        "attr_id": attr_id,
                        "reason": reason,
                    }
                )

        if not instances:
            errors.append({"image_id": row["image_id"], "error": "no_valid_instances"})
            continue

        instance_counts[str(len(instances))] += 1
        records_by_split[row["redactions_split"]].append(
            {
                "image_id": row["image_id"],
                "redactions_split": row["redactions_split"],
                "image_release": row["image_release"],
                "image_split": row["image_split"],
                "image_path": row["image_path"],
                "width": width,
                "height": height,
                "source_annotation_width": source_width,
                "source_annotation_height": source_height,
                "instances": instances,
                "excluded_instances": excluded_for_image,
            }
        )

    output_files = {}
    for split in ("train2017", "val2017", "test2017"):
        records = records_by_split[split]
        path = OUTPUT / f"records_{split}.jsonl"
        write_jsonl_atomic(path, sorted(records, key=lambda item: item["image_id"]))
        output_files[split] = str(path.relative_to(ROOT)).replace("\\", "/")

    pending_path = OUTPUT / "pending_records.jsonl"
    write_jsonl_atomic(pending_path, sorted(pending, key=lambda item: item["image_id"]))

    summary = {
        "manifest_rows": len(manifest_rows),
        "processed_records": sum(len(records) for records in records_by_split.values()),
        "pending_records": len(pending),
        "invalid_records": len(errors),
        "errors": errors,
        "records_by_split": {split: len(records) for split, records in sorted(records_by_split.items())},
        "instances_by_attribute": dict(attribute_counts.most_common()),
        "excluded_instances": dict(excluded_instances),
        "excluded_instance_total": sum(excluded_instances.values()),
        "instance_count_distribution": dict(instance_counts),
        "resize_pairs_count": len(resize_pairs),
        "class_count": len(class_ids),
        "class_map": str((OUTPUT / "class_map.json").relative_to(ROOT)).replace("\\", "/"),
        "output_files": output_files,
        "pending_file": str(pending_path.relative_to(ROOT)).replace("\\", "/"),
        "raw_data_unchanged": True,
    }
    write_json_atomic(OUTPUT / "preprocess_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
