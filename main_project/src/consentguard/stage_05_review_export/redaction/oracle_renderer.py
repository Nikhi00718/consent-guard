"""Render a destructive redaction using ground-truth polygons.

This is the Phase 2 oracle-mask proof of concept. It deliberately does not
pretend that a detector is working yet; it validates the renderer and export
boundary independently from model errors.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[5]
MANIFEST = ROOT / "data" / "manifests" / "visual_redactions_master.jsonl"
ANNOTATIONS = ROOT / "data" / "raw" / "visual_redactions"


def load_record(image_id: str) -> tuple[dict, dict]:
    manifest_row = None
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["image_id"] == image_id:
            manifest_row = row
            break
    if manifest_row is None:
        raise FileNotFoundError(f"No manifest record for image ID {image_id}")
    if not manifest_row["image_path"]:
        raise FileNotFoundError(f"Image is not extracted yet: {image_id}")

    annotation_path = ANNOTATIONS / f"{manifest_row['redactions_split']}.json"
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    return manifest_row, payload["annotations"][image_id]


def polygon_mask(record: dict, target_height: int, target_width: int) -> np.ndarray:
    source_height = int(record["image_height"])
    source_width = int(record["image_width"])
    scale_x = target_width / source_width
    scale_y = target_height / source_height
    mask = np.zeros((target_height, target_width), dtype=np.uint8)
    for attribute in record.get("attributes", []):
        for polygon in attribute.get("polygons", []):
            points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
            points[:, 0] *= scale_x
            points[:, 1] *= scale_y
            points[:, 0] = np.clip(points[:, 0], 0, target_width - 1)
            points[:, 1] = np.clip(points[:, 1], 0, target_height - 1)
            cv2.fillPoly(mask, [np.round(points).astype(np.int32)], 255)
    return mask


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output = args.output.resolve()

    manifest_row, annotation = load_record(args.image_id)
    image_path = ROOT / manifest_row["image_path"]
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not decode image: {image_path}")

    image_height, image_width = image.shape[:2]
    mask = polygon_mask(annotation, image_height, image_width)

    output = image.copy()
    output[mask > 0] = (0, 0, 0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), output):
        raise RuntimeError(f"Could not write output: {args.output}")

    report = {
        "image_id": args.image_id,
        "source": str(image_path.relative_to(ROOT)).replace("\\", "/"),
        "output": str(args.output.relative_to(ROOT)).replace("\\", "/"),
        "method": "solid_replace",
        "image_width": int(image.shape[1]),
        "image_height": int(image.shape[0]),
        "annotation_width": int(annotation["image_width"]),
        "annotation_height": int(annotation["image_height"]),
        "annotation_to_image_scale": [
            float(image.shape[1] / annotation["image_width"]),
            float(image.shape[0] / annotation["image_height"]),
        ],
        "instances": len(annotation.get("attributes", [])),
        "attribute_ids": sorted({a.get("attr_id") for a in annotation.get("attributes", [])}),
        "redacted_pixels": int(np.count_nonzero(mask)),
        "redacted_fraction": float(np.count_nonzero(mask) / mask.size),
    }
    report_path = args.output.with_suffix(args.output.suffix + ".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
