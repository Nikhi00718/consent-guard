"""Audit Visual Redactions annotation geometry against decoded VISPR images.

The annotations are defined in their recorded ``image_width``/``image_height``
coordinate space.  A plain resize preserves aspect ratio, so independent X/Y
scales are valid only when those scales are approximately equal.  Crops,
stitches, and rotations are quarantined rather than silently stretched.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from consentguard.data_quality import geometry_status, image_display_size


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifests" / "visual_redactions_master.jsonl"
ANNOTATIONS = ROOT / "data" / "raw" / "visual_redactions"
REPORTS = ROOT / "reports"


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


def load_annotations() -> dict[str, dict]:
    result: dict[str, dict] = {}
    for split in ("train2017", "val2017", "test2017"):
        payload = json.loads((ANNOTATIONS / f"{split}.json").read_text(encoding="utf-8"))
        result.update(payload["annotations"])
    return result


def draw_overlay(image: np.ndarray, annotation: dict, status: str) -> np.ndarray:
    height, width = image.shape[:2]
    source_width = int(annotation["image_width"])
    source_height = int(annotation["image_height"])
    scale_x = width / source_width
    scale_y = height / source_height
    overlay = image.copy()
    colors = ((0, 0, 255), (0, 180, 255), (255, 80, 0), (180, 0, 255))

    for index, attribute in enumerate(annotation.get("attributes", [])):
        color = colors[index % len(colors)]
        for polygon in attribute.get("polygons", []):
            points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2).copy()
            points[:, 0] = np.clip(points[:, 0] * scale_x, 0, width - 1)
            points[:, 1] = np.clip(points[:, 1] * scale_y, 0, height - 1)
            points_i = np.round(points).astype(np.int32)
            cv2.fillPoly(overlay, [points_i], color)
            cv2.polylines(image, [points_i], True, color, max(1, round(min(width, height) / 300)))

    image = cv2.addWeighted(overlay, 0.28, image, 0.72, 0)
    title = (
        f"{status} | decoded {width}x{height} | annotation "
        f"{source_width}x{source_height} | sx={scale_x:.3f} sy={scale_y:.3f}"
    )
    cv2.rectangle(image, (0, 0), (width, min(height, 42)), (0, 0, 0), -1)
    cv2.putText(
        image,
        title,
        (8, min(height - 8, 28)),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(0.35, min(0.7, width / 1500)),
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return image


def make_montage(images: list[np.ndarray], output: Path) -> None:
    if not images:
        return
    tile_width, tile_height = 800, 520
    tiles = []
    for image in images:
        scale = min(tile_width / image.shape[1], tile_height / image.shape[0])
        resized = cv2.resize(
            image,
            (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
        canvas = np.full((tile_height, tile_width, 3), 28, dtype=np.uint8)
        y = (tile_height - resized.shape[0]) // 2
        x = (tile_width - resized.shape[1]) // 2
        canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
        tiles.append(canvas)
    if len(tiles) % 2:
        tiles.append(np.full_like(tiles[0], 28))
    rows = [np.hstack(tiles[index : index + 2]) for index in range(0, len(tiles), 2)]
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), np.vstack(rows)):
        raise RuntimeError(f"Could not write montage: {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tolerance", type=float, default=0.01)
    parser.add_argument("--overlay-id", action="append", default=[])
    parser.add_argument(
        "--report",
        type=Path,
        default=REPORTS / "visual_redactions_alignment_audit.json",
    )
    parser.add_argument(
        "--montage",
        type=Path,
        default=REPORTS / "visual_redactions_alignment_overlays.jpg",
    )
    args = parser.parse_args()
    if not 0 < args.tolerance < 0.25:
        raise ValueError("tolerance must be between 0 and 0.25")

    annotations = load_annotations()
    rows = [json.loads(line) for line in MANIFEST.read_text(encoding="utf-8").splitlines()]
    status_counts: Counter[str] = Counter()
    split_status_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    attr_status_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    records: list[dict] = []
    overlays: list[np.ndarray] = []
    overlay_ids = set(args.overlay_id)

    for row in rows:
        if not row.get("image_path"):
            status = "missing_image"
            status_counts[status] += 1
            split_status_counts[str(row["redactions_split"])][status] += 1
            continue
        image_path = ROOT / str(row["image_path"])
        try:
            decoded_width, decoded_height = image_display_size(image_path)
        except (OSError, ValueError):
            status = "decode_failed"
            status_counts[status] += 1
            split_status_counts[str(row["redactions_split"])][status] += 1
            continue
        source_width = int(row["image_width"])
        source_height = int(row["image_height"])
        status, direct_error, rotated_error = geometry_status(
            source_width,
            source_height,
            decoded_width,
            decoded_height,
            args.tolerance,
        )
        status_counts[status] += 1
        split_status_counts[str(row["redactions_split"])][status] += 1
        for attr_id in row.get("attribute_ids", []):
            attr_status_counts[str(attr_id)][status] += 1
        records.append(
            {
                "image_id": row["image_id"],
                "redactions_split": row["redactions_split"],
                "status": status,
                "image_path": row["image_path"],
                "decoded_width": decoded_width,
                "decoded_height": decoded_height,
                "annotation_width": source_width,
                "annotation_height": source_height,
                "direct_aspect_log_error": direct_error,
                "rotated_aspect_log_error": rotated_error,
            }
        )
        if row["image_id"] in overlay_ids:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"Could not decode overlay image: {image_path}")
            overlays.append(draw_overlay(image, annotations[row["image_id"]], status))

    report = {
        "tolerance_fraction": args.tolerance,
        "manifest_rows": len(rows),
        "status_counts": dict(status_counts),
        "split_status_counts": {
            split: dict(counts) for split, counts in sorted(split_status_counts.items())
        },
        "attribute_status_counts": {
            attr_id: dict(counts) for attr_id, counts in sorted(attr_status_counts.items())
        },
        "records": sorted(records, key=lambda item: str(item["image_id"])),
        "safe_for_independent_xy_scaling": status_counts["aligned_resize"],
        "quarantined": status_counts["rotation_candidate"]
        + status_counts["geometry_mismatch"]
        + status_counts["missing_image"]
        + status_counts["decode_failed"],
    }
    args.report = args.report if args.report.is_absolute() else ROOT / args.report
    args.montage = args.montage if args.montage.is_absolute() else ROOT / args.montage
    args.report.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(args.report, report)
    make_montage(overlays, args.montage)
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, indent=2))
    if overlays:
        print(f"Overlay montage: {args.montage}")


if __name__ == "__main__":
    main()
