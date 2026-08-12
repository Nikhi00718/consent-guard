"""Build same-release, geometry-verified Visual Redactions records.

This pipeline intentionally writes to a new directory.  Legacy records used
independent X/Y scaling for every image and are retained only for debugging old
checkpoints; they must not be used for new training.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2

from consentguard.data_quality import (
    IGNORED_ATTRIBUTE_IDS,
    OFFICIAL_MIN_PIXELS,
    PROFILE_ATTRIBUTE_IDS,
    geometry_status,
    image_display_size,
)
from preprocess_visual_redactions import (
    polygon_area,
    scaled_bbox,
    scaled_polygon,
    write_json_atomic,
    write_jsonl_atomic,
)


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifests" / "visual_redactions_master.jsonl"
RAW_ANNOTATIONS = ROOT / "data" / "raw" / "visual_redactions"
LEAKAGE_QUARANTINE = ROOT / "configs" / "cross_split_leakage_quarantine.json"


def source_area(attribute: dict) -> float:
    value = attribute.get("area")
    if isinstance(value, list):
        return float(sum(float(item) for item in value))
    if value is not None:
        return float(value)
    return float(sum(polygon_area([float(item) for item in polygon]) for polygon in attribute["polygons"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILE_ATTRIBUTE_IDS), default="visual")
    parser.add_argument("--geometry-tolerance", type=float, default=0.01)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/visual_redactions_verified_visual"),
    )
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)

    selected_attributes = tuple(PROFILE_ATTRIBUTE_IDS[args.profile])
    selected_set = set(selected_attributes)
    class_map = {"background": 0}
    class_map.update(
        {attr_id: class_id for class_id, attr_id in enumerate(selected_attributes, start=1)}
    )
    write_json_atomic(output / "class_map.json", class_map)

    annotations: dict[str, dict] = {}
    for split in ("train2017", "val2017", "test2017"):
        payload = json.loads((RAW_ANNOTATIONS / f"{split}.json").read_text(encoding="utf-8"))
        annotations.update(payload["annotations"])
    manifest_rows = [
        json.loads(line) for line in MANIFEST.read_text(encoding="utf-8").splitlines()
    ]
    leakage_policy = json.loads(LEAKAGE_QUARANTINE.read_text(encoding="utf-8"))
    leakage_train_ids = {
        str(image_id): str(reason)
        for image_id, reason in leakage_policy.get("train2017", {}).items()
    }

    records_by_split: defaultdict[str, list[dict]] = defaultdict(list)
    quarantined: list[dict] = []
    leakage_quarantined: list[dict] = []
    omitted: list[dict] = []
    invalid_instances: list[dict] = []
    geometry_counts: Counter[str] = Counter()
    split_geometry_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    attribute_counts: Counter[str] = Counter()
    tiny_instance_counts: Counter[str] = Counter()

    for row in manifest_rows:
        image_id = str(row["image_id"])
        split = str(row["redactions_split"])
        if split == "train2017" and image_id in leakage_train_ids:
            leakage_quarantined.append(
                {
                    "image_id": image_id,
                    "split": split,
                    "status": "cross_split_duplicate",
                    "reason": leakage_train_ids[image_id],
                }
            )
            continue
        if not row.get("image_path"):
            status = "missing_image"
            geometry_counts[status] += 1
            split_geometry_counts[split][status] += 1
            quarantined.append({"image_id": image_id, "split": split, "status": status})
            continue

        image_path = ROOT / str(row["image_path"])
        try:
            decoded_width, decoded_height = image_display_size(image_path)
        except (OSError, ValueError) as error:
            status = "decode_failed"
            geometry_counts[status] += 1
            split_geometry_counts[split][status] += 1
            quarantined.append(
                {"image_id": image_id, "split": split, "status": status, "error": str(error)}
            )
            continue

        source_width = int(row["image_width"])
        source_height = int(row["image_height"])
        status, direct_error, rotated_error = geometry_status(
            source_width,
            source_height,
            decoded_width,
            decoded_height,
            args.geometry_tolerance,
        )
        geometry_counts[status] += 1
        split_geometry_counts[split][status] += 1
        geometry_record = {
            "image_id": image_id,
            "split": split,
            "status": status,
            "image_path": row["image_path"],
            "decoded_width": decoded_width,
            "decoded_height": decoded_height,
            "annotation_width": source_width,
            "annotation_height": source_height,
            "direct_aspect_log_error": direct_error,
            "rotated_aspect_log_error": rotated_error,
        }
        if status != "aligned_resize":
            quarantined.append(geometry_record)
            continue

        # Confirm the dimensions seen by the actual training decoder.  Header
        # metadata alone is never allowed to define a model target.
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            geometry_record["status"] = "training_decode_failed"
            quarantined.append(geometry_record)
            continue
        cv_height, cv_width = image.shape[:2]
        if (cv_width, cv_height) != (decoded_width, decoded_height):
            actual_status, actual_direct_error, actual_rotated_error = geometry_status(
                source_width,
                source_height,
                cv_width,
                cv_height,
                args.geometry_tolerance,
            )
            if actual_status != "aligned_resize":
                geometry_record.update(
                    {
                        "status": "training_decoder_geometry_mismatch",
                        "training_decoder_width": cv_width,
                        "training_decoder_height": cv_height,
                        "training_direct_aspect_log_error": actual_direct_error,
                        "training_rotated_aspect_log_error": actual_rotated_error,
                    }
                )
                quarantined.append(geometry_record)
                continue
            decoded_width, decoded_height = cv_width, cv_height

        scale_x = decoded_width / source_width
        scale_y = decoded_height / source_height
        annotation = annotations[image_id]
        instances = []
        for attribute in annotation.get("attributes", []):
            attr_id = str(attribute.get("attr_id", "unknown"))
            if attr_id not in selected_set:
                continue
            try:
                polygons = [
                    scaled_polygon(
                        polygon,
                        scale_x,
                        scale_y,
                        decoded_width,
                        decoded_height,
                    )
                    for polygon in attribute.get("polygons", [])
                ]
                if not polygons or not any(polygon_area(polygon) > 0 for polygon in polygons):
                    raise ValueError("instance has no positive-area polygon")
                bbox = scaled_bbox(
                    attribute["bbox"],
                    scale_x,
                    scale_y,
                    decoded_width,
                    decoded_height,
                )
                if bbox[2] <= 0 or bbox[3] <= 0:
                    raise ValueError("instance has zero-area bbox")
                original_area = source_area(attribute)
                instances.append(
                    {
                        "instance_id": int(attribute["instance_id"]),
                        "class_id": class_map[attr_id],
                        "attr_id": attr_id,
                        "bbox": bbox,
                        "polygons": polygons,
                        "iscrowd": bool(attribute.get("iscrowd", False)),
                        "source_area_pixels": round(original_area, 3),
                        "area_pixels": round(original_area * scale_x * scale_y, 3),
                        "official_eval_ignore_small": original_area < OFFICIAL_MIN_PIXELS,
                    }
                )
                attribute_counts[attr_id] += 1
                if original_area < OFFICIAL_MIN_PIXELS:
                    tiny_instance_counts[attr_id] += 1
            except (KeyError, TypeError, ValueError) as error:
                invalid_instances.append(
                    {
                        "image_id": image_id,
                        "instance_id": attribute.get("instance_id"),
                        "attr_id": attr_id,
                        "error": str(error),
                    }
                )

        if not instances:
            omitted.append(
                {
                    "image_id": image_id,
                    "split": split,
                    "reason": "no_valid_profile_instances",
                }
            )
            continue

        records_by_split[split].append(
            {
                "image_id": image_id,
                "redactions_split": split,
                "image_release": row["image_release"],
                "image_split": row["image_split"],
                "image_path": row["image_path"],
                "width": decoded_width,
                "height": decoded_height,
                "source_annotation_width": source_width,
                "source_annotation_height": source_height,
                "geometry_status": "aligned_resize",
                "geometry_tolerance_fraction": args.geometry_tolerance,
                "instances": instances,
                "excluded_instances": [],
            }
        )

    output_files = {}
    for split in ("train2017", "val2017", "test2017"):
        path = output / f"records_{split}.jsonl"
        write_jsonl_atomic(
            path,
            sorted(records_by_split[split], key=lambda record: record["image_id"]),
        )
        output_files[split] = str(path.relative_to(ROOT)).replace("\\", "/")
    write_jsonl_atomic(
        output / "quarantined_geometry.jsonl",
        sorted(quarantined, key=lambda record: record["image_id"]),
    )
    write_jsonl_atomic(
        output / "quarantined_cross_split_leakage.jsonl",
        sorted(leakage_quarantined, key=lambda record: record["image_id"]),
    )
    write_jsonl_atomic(
        output / "omitted_no_profile_instances.jsonl",
        sorted(omitted, key=lambda record: record["image_id"]),
    )
    write_jsonl_atomic(
        output / "invalid_instances.jsonl",
        sorted(invalid_instances, key=lambda record: (record["image_id"], str(record["instance_id"]))),
    )

    summary = {
        "profile": args.profile,
        "profile_attribute_ids": list(selected_attributes),
        "ignored_official_attribute_ids": list(IGNORED_ATTRIBUTE_IDS),
        "official_min_pixels": OFFICIAL_MIN_PIXELS,
        "geometry_tolerance_fraction": args.geometry_tolerance,
        "manifest_rows": len(manifest_rows),
        "geometry_status_counts": dict(geometry_counts),
        "split_geometry_status_counts": {
            split: dict(counts) for split, counts in sorted(split_geometry_counts.items())
        },
        "records_by_split": {
            split: len(records_by_split[split])
            for split in ("train2017", "val2017", "test2017")
        },
        "processed_records": sum(len(records) for records in records_by_split.values()),
        "quarantined_records": len(quarantined),
        "cross_split_leakage_policy": str(LEAKAGE_QUARANTINE.relative_to(ROOT)).replace("\\", "/"),
        "cross_split_leakage_quarantined": len(leakage_quarantined),
        "omitted_no_profile_instances": len(omitted),
        "invalid_instance_count": len(invalid_instances),
        "instances_by_attribute": dict(attribute_counts),
        "small_instances_ignored_by_official_evaluation": dict(tiny_instance_counts),
        "class_map": str((output / "class_map.json").relative_to(ROOT)).replace("\\", "/"),
        "output_files": output_files,
        "raw_data_unchanged": True,
        "legacy_records_safe_for_new_training": False,
    }
    write_json_atomic(output / "preprocess_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not records_by_split["train2017"] or not records_by_split["val2017"]:
        raise RuntimeError("verified preprocessing produced an empty train or validation split")


if __name__ == "__main__":
    main()
