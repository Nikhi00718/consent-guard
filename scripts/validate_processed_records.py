"""Validate compact model-ready records and enforce split/path invariants."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


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


def add_invalid(invalid: list[dict[str, str]], image_id: str, error: str) -> None:
    invalid.append({"image_id": image_id, "error": error})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/processed/visual_redactions"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/processed_records_validation.json"),
    )
    args = parser.parse_args()
    data = args.data if args.data.is_absolute() else ROOT / args.data
    report_path = args.report if args.report.is_absolute() else ROOT / args.report

    class_map_path = data / "class_map.json"
    if not class_map_path.is_file():
        raise FileNotFoundError(f"Missing class map: {class_map_path}")
    class_map = json.loads(class_map_path.read_text(encoding="utf-8"))
    valid_class_ids = {int(value) for value in class_map.values()} - {0}

    checked = 0
    instance_count = 0
    invalid: list[dict[str, str]] = []
    duplicate_ids: list[str] = []
    duplicate_image_paths: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    path_owner: dict[str, str] = {}
    records_by_split: Counter[str] = Counter()
    instances_by_class: Counter[int] = Counter()
    root_resolved = ROOT.resolve()
    release_root = (ROOT / "data" / "raw" / "visual_redactions" / "images").resolve()

    for path in sorted(data.glob("records_*.jsonl")):
        expected_split = path.stem.removeprefix("records_")
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                add_invalid(invalid, f"{path.name}:{line_number}", f"invalid_json:{error.msg}")
                continue
            checked += 1
            image_id = str(record.get("image_id", f"{path.name}:{line_number}"))
            if image_id in seen_ids:
                duplicate_ids.append(image_id)
            seen_ids.add(image_id)
            if record.get("redactions_split") != expected_split:
                add_invalid(invalid, image_id, "record_split_does_not_match_filename")
                continue
            if record.get("image_release") != "visual_redactions_v1":
                add_invalid(invalid, image_id, "record_image_release_is_not_visual_redactions_v1")
                continue
            if record.get("image_split") != expected_split:
                add_invalid(invalid, image_id, "record_image_split_does_not_match_annotation_split")
                continue
            if "geometry_status" in record and record["geometry_status"] != "aligned_resize":
                add_invalid(invalid, image_id, "record_geometry_is_not_verified")
                continue
            records_by_split[expected_split] += 1

            try:
                image_path = (ROOT / record["image_path"]).resolve()
                width = int(record["width"])
                height = int(record["height"])
                instances = record["instances"]
            except (KeyError, TypeError, ValueError):
                add_invalid(invalid, image_id, "missing_or_invalid_required_field")
                continue
            if image_path != root_resolved and root_resolved not in image_path.parents:
                add_invalid(invalid, image_id, "image_path_outside_project")
                continue
            expected_image_root = release_root / expected_split
            if expected_image_root != image_path.parent and expected_image_root not in image_path.parents:
                add_invalid(invalid, image_id, "image_path_outside_same_release_split")
                continue
            if not image_path.is_file():
                add_invalid(invalid, image_id, "image_file_missing")
                continue
            normalized_image_path = str(image_path).casefold()
            previous_owner = path_owner.get(normalized_image_path)
            if previous_owner is not None and previous_owner != image_id:
                duplicate_image_paths.append(
                    {"image_path": str(image_path), "first_image_id": previous_owner, "image_id": image_id}
                )
            path_owner[normalized_image_path] = image_id
            if width < 1 or height < 1 or not isinstance(instances, list) or not instances:
                add_invalid(invalid, image_id, "invalid_dimensions_or_empty_instances")
                continue

            record_error = None
            for instance in instances:
                try:
                    class_id = int(instance["class_id"])
                    x, y, box_width, box_height = (float(value) for value in instance["bbox"])
                    polygons = instance["polygons"]
                except (KeyError, TypeError, ValueError):
                    record_error = "malformed_instance"
                    break
                if class_id not in valid_class_ids:
                    record_error = "unknown_class_id"
                    break
                if not all(math.isfinite(value) for value in (x, y, box_width, box_height)):
                    record_error = "non_finite_bbox"
                    break
                if box_width <= 0 or box_height <= 0:
                    record_error = "invalid_bbox_area"
                    break
                if x < 0 or y < 0 or x + box_width > width + 0.01 or y + box_height > height + 0.01:
                    record_error = "bbox_out_of_bounds"
                    break
                if not isinstance(polygons, list) or not polygons:
                    record_error = "missing_polygons"
                    break
                valid_polygon_area = False
                for polygon in polygons:
                    if not isinstance(polygon, list) or len(polygon) < 6 or len(polygon) % 2:
                        record_error = "invalid_polygon_length"
                        break
                    try:
                        coordinates = [float(value) for value in polygon]
                    except (TypeError, ValueError):
                        record_error = "non_numeric_polygon"
                        break
                    if not all(math.isfinite(value) for value in coordinates):
                        record_error = "non_finite_polygon"
                        break
                    points = list(zip(coordinates[0::2], coordinates[1::2]))
                    if any(
                        x_coord < 0 or y_coord < 0 or x_coord >= width or y_coord >= height
                        for x_coord, y_coord in points
                    ):
                        record_error = "polygon_out_of_bounds"
                        break
                    valid_polygon_area |= polygon_area(coordinates) > 0
                if record_error:
                    break
                if not valid_polygon_area:
                    record_error = "zero_area_polygons"
                    break
                instances_by_class[class_id] += 1
                instance_count += 1
            if record_error:
                add_invalid(invalid, image_id, record_error)

    report = {
        "checked_records": checked,
        "checked_instances": instance_count,
        "records_by_split": dict(sorted(records_by_split.items())),
        "instances_by_class_id": {
            str(class_id): count for class_id, count in sorted(instances_by_class.items())
        },
        "invalid_records": invalid,
        "duplicate_ids": sorted(set(duplicate_ids)),
        "duplicate_image_paths": duplicate_image_paths,
        "passed": checked > 0 and not invalid and not duplicate_ids and not duplicate_image_paths,
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_path.with_suffix(report_path.suffix + f".tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, report_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
