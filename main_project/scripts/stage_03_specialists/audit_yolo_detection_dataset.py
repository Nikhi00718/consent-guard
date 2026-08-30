"""Audit a split YOLO detection export before it enters ConsentGuard training."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
SPLIT_NAMES = ("train", "valid", "val", "test")


class _Groups:
    def __init__(self, count: int) -> None:
        self.parent = list(range(count))

    def find(self, index: int) -> int:
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _perceptual_hash(rgb: np.ndarray) -> int:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    frequencies = cv2.dct(resized)[:8, :8]
    median = float(np.median(frequencies.reshape(-1)[1:]))
    bits = frequencies > median
    value = 0
    for bit in bits.reshape(-1):
        value = (value << 1) | int(bit)
    return value


def _asset_keys(path: Path) -> tuple[str, str]:
    stem = re.sub(r"\.rf\.[0-9a-f]{16,}$", "", path.stem, flags=re.IGNORECASE)
    stem = re.sub(r"^[0-9a-f-]{36}___", "", stem, flags=re.IGNORECASE)
    normalized = re.sub(r"[^a-z0-9]+", "-", stem.lower()).strip("-")
    video_group = re.sub(
        r"(?:-?(?:mp4|mov|avi)-?t-?\d+.*|-?frame-?\d+.*)$",
        "",
        normalized,
        flags=re.IGNORECASE,
    ).strip("-")
    numbered_video = re.match(r"^(video-?\d+)-\d+(?:-jpg)?$", normalized, flags=re.IGNORECASE)
    if numbered_video:
        video_group = numbered_video.group(1)
    if not video_group or video_group == normalized:
        video_group = ""
    return normalized, video_group


def _parse_label(path: Path, width: int, height: int) -> tuple[list[dict[str, Any]], list[str]]:
    boxes: list[dict[str, Any]] = []
    errors: list[str] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"line {line_number}: expected 5 values, found {len(parts)}")
            continue
        try:
            class_value, center_x, center_y, box_width, box_height = (float(value) for value in parts)
        except ValueError:
            errors.append(f"line {line_number}: non-numeric value")
            continue
        values = (class_value, center_x, center_y, box_width, box_height)
        if not all(math.isfinite(value) for value in values):
            errors.append(f"line {line_number}: non-finite value")
            continue
        if not class_value.is_integer() or class_value < 0:
            errors.append(f"line {line_number}: invalid class id {class_value}")
            continue
        if not (0 <= center_x <= 1 and 0 <= center_y <= 1 and 0 < box_width <= 1 and 0 < box_height <= 1):
            errors.append(f"line {line_number}: normalized geometry outside [0, 1]")
            continue
        left = center_x - box_width / 2
        top = center_y - box_height / 2
        right = center_x + box_width / 2
        bottom = center_y + box_height / 2
        if left < 0 or top < 0 or right > 1 or bottom > 1:
            errors.append(f"line {line_number}: box extends outside image")
            continue
        pixel_width = box_width * width
        pixel_height = box_height * height
        if pixel_width <= 1 or pixel_height <= 1:
            errors.append(f"line {line_number}: box is one pixel or thinner")
            continue
        boxes.append(
            {
                "class_id": int(class_value),
                "center_x": center_x,
                "center_y": center_y,
                "width": box_width,
                "height": box_height,
                "pixel_width": pixel_width,
                "pixel_height": pixel_height,
                "pixel_area": pixel_width * pixel_height,
            }
        )
    return boxes, errors


def _collect(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, str]], list[str]]:
    records: list[dict[str, Any]] = []
    label_errors: list[dict[str, str]] = []
    unmatched_labels: list[str] = []
    observed_labels: set[Path] = set()
    for split in SPLIT_NAMES:
        image_root = root / split / "images"
        label_root = root / split / "labels"
        if not image_root.is_dir():
            continue
        for image_path in sorted(path for path in image_root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES):
            relative = image_path.relative_to(image_root)
            label_path = label_root / relative.with_suffix(".txt")
            observed_labels.add(label_path.resolve())
            record: dict[str, Any] = {
                "source_split": split,
                "image_path": image_path,
                "label_path": label_path,
                "image_sha256": _sha256(image_path),
                "label_sha256": _sha256(label_path) if label_path.is_file() else None,
                "decode_error": None,
                "boxes": [],
            }
            asset_key, video_group = _asset_keys(image_path)
            record["source_asset_key"] = asset_key
            record["source_video_group"] = video_group
            try:
                with Image.open(image_path) as image:
                    oriented = ImageOps.exif_transpose(image).convert("RGB")
                    rgb = np.asarray(oriented, dtype=np.uint8)
            except (OSError, UnidentifiedImageError, ValueError) as error:
                record["decode_error"] = str(error)
                record["width"] = None
                record["height"] = None
                record["phash"] = None
            else:
                record["width"] = int(rgb.shape[1])
                record["height"] = int(rgb.shape[0])
                record["phash"] = _perceptual_hash(rgb)
                if label_path.is_file():
                    boxes, errors = _parse_label(label_path, record["width"], record["height"])
                    record["boxes"] = boxes
                    label_errors.extend(
                        {"label_path": label_path.relative_to(root).as_posix(), "error": error}
                        for error in errors
                    )
            records.append(record)
        if label_root.is_dir():
            for label_path in label_root.rglob("*.txt"):
                if label_path.resolve() not in observed_labels:
                    unmatched_labels.append(label_path.relative_to(root).as_posix())
    return records, label_errors, sorted(unmatched_labels)


def _group_records(records: list[dict[str, Any]], *, duplicate_distance: int, suspicion_distance: int):
    groups = _Groups(len(records))
    exact: dict[str, list[int]] = defaultdict(list)
    assets: dict[str, list[int]] = defaultdict(list)
    videos: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        exact[record["image_sha256"]].append(index)
        assets[record["source_asset_key"]].append(index)
        if record["source_video_group"]:
            videos[record["source_video_group"]].append(index)
    for buckets in (exact, assets, videos):
        for indices in buckets.values():
            for index in indices[1:]:
                groups.union(indices[0], index)

    suspicious: list[dict[str, Any]] = []
    valid = [(index, record["phash"]) for index, record in enumerate(records) if record["phash"] is not None]
    for position, (left_index, left_hash) in enumerate(valid):
        for right_index, right_hash in valid[position + 1 :]:
            distance = (left_hash ^ right_hash).bit_count()
            if distance <= duplicate_distance:
                groups.union(left_index, right_index)
            elif distance <= suspicion_distance and len(suspicious) < 500:
                suspicious.append(
                    {
                        "left": records[left_index]["image_path"].name,
                        "right": records[right_index]["image_path"].name,
                        "distance": distance,
                    }
                )

    components: dict[int, list[int]] = defaultdict(list)
    for index in range(len(records)):
        components[groups.find(index)].append(index)
    ordered = sorted(components.values(), key=lambda values: min(records[index]["image_path"].as_posix() for index in values))
    for group_number, indices in enumerate(ordered, 1):
        group_id = f"group-{group_number:05d}"
        for index in indices:
            records[index]["group_id"] = group_id
    return ordered, suspicious


def audit(args: argparse.Namespace) -> dict[str, Any]:
    root = args.source_root.resolve()
    project_root = Path.cwd().resolve()
    records, label_errors, unmatched_labels = _collect(root)
    if not records:
        raise FileNotFoundError(f"No split YOLO images found under {root}")
    components, suspicious = _group_records(
        records,
        duplicate_distance=args.duplicate_distance,
        suspicion_distance=args.suspicion_distance,
    )
    cross_split_groups = []
    for indices in components:
        splits = sorted({records[index]["source_split"] for index in indices})
        if len(splits) > 1:
            cross_split_groups.append(
                {
                    "group_id": records[indices[0]]["group_id"],
                    "splits": splits,
                    "images": [records[index]["image_path"].relative_to(root).as_posix() for index in indices],
                }
            )

    valid_boxes = [box for record in records for box in record["boxes"]]
    class_counts = Counter(box["class_id"] for box in valid_boxes)
    widths = [box["pixel_width"] for box in valid_boxes]
    size_bins = Counter(
        "tiny_lt_32" if width < 32 else "small_32_64" if width < 64 else "medium_64_128" if width < 128 else "large_ge_128"
        for width in widths
    )
    split_summary = {}
    for split in sorted({record["source_split"] for record in records}):
        split_records = [record for record in records if record["source_split"] == split]
        split_summary[split] = {
            "images": len(split_records),
            "positive_images": sum(bool(record["boxes"]) for record in split_records),
            "negative_images": sum(not record["boxes"] for record in split_records),
            "boxes": sum(len(record["boxes"]) for record in split_records),
        }

    manifest_path = args.manifest_output.resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            row = {
                "schema_version": "consentguard-dataset-record-v2",
                "dataset_id": args.dataset_id,
                "source_url": args.source_url,
                "license_id": args.license_id,
                "image_path": record["image_path"].relative_to(project_root).as_posix(),
                "label_path": record["label_path"].relative_to(project_root).as_posix(),
                "source_split": record["source_split"],
                "group_id": record["group_id"],
                "source_asset_key": record["source_asset_key"],
                "source_video_group": record["source_video_group"] or None,
                "image_sha256": record["image_sha256"],
                "label_sha256": record["label_sha256"],
                "width": record["width"],
                "height": record["height"],
                "instances": len(record["boxes"]),
                "negative": not record["boxes"],
                "annotation_qa": "valid" if record["decode_error"] is None else "decode_error",
            }
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")

    blocking_errors = (
        sum(record["decode_error"] is not None for record in records)
        + sum(not record["label_path"].is_file() for record in records)
        + len(label_errors)
        + len(unmatched_labels)
        + len(cross_split_groups)
    )
    report = {
        "schema_version": "yolo-dataset-audit-v1",
        "dataset_id": args.dataset_id,
        "source_root": str(root),
        "source_url": args.source_url,
        "license_id": args.license_id,
        "archive_sha256": args.archive_sha256,
        "images": len(records),
        "decoded_images": sum(record["decode_error"] is None for record in records),
        "missing_labels": sum(not record["label_path"].is_file() for record in records),
        "unmatched_labels": unmatched_labels,
        "label_errors": label_errors,
        "boxes": len(valid_boxes),
        "class_counts": {str(key): value for key, value in sorted(class_counts.items())},
        "plate_width_bins": dict(sorted(size_bins.items())),
        "minimum_plate_width_pixels": min(widths) if widths else None,
        "median_plate_width_pixels": float(np.median(widths)) if widths else None,
        "splits": split_summary,
        "groups": len(components),
        "multi_image_groups": sum(len(indices) > 1 for indices in components),
        "cross_split_groups": cross_split_groups,
        "suspected_near_duplicate_pairs": suspicious,
        "duplicate_distance": args.duplicate_distance,
        "suspicion_distance": args.suspicion_distance,
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
        "blocking_errors": blocking_errors,
        "release_admitted": blocking_errors == 0 and bool(args.license_id and args.source_url),
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--license-id", required=True)
    parser.add_argument("--archive-sha256", default=None)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duplicate-distance", type=int, default=2)
    parser.add_argument("--suspicion-distance", type=int, default=6)
    args = parser.parse_args()
    print(json.dumps(audit(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
