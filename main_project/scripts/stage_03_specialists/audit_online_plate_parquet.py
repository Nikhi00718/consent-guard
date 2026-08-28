"""Audit a Hugging Face Indian plate parquet before using it for training.

The public parquet examined by this script stores small plate crops while its
bounding-box columns refer to the original full-resolution image.  That makes
it useful for OCR/data provenance review, but unsafe to feed directly to a
plate detector until the original scenes and matching coordinate system are
available.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

from PIL import Image


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(path: Path) -> dict[str, object]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:  # pragma: no cover - command-line guard
        raise SystemExit("Install pyarrow to audit parquet files: python -m pip install pyarrow") from error

    table = parquet.read_table(path)
    rows = table.to_pylist()
    dimension_counts: Counter[tuple[int, int]] = Counter()
    source_counts: Counter[str] = Counter()
    state_counts: Counter[str] = Counter()
    out_of_bounds = 0
    invalid_boxes = 0
    crop_area_ratios: list[float] = []
    image_bytes = 0

    for row in rows:
        raw = row["image"].get("bytes") or b""
        image_bytes += len(raw)
        with Image.open(BytesIO(raw)) as image:
            width, height = image.size
        dimension_counts[(width, height)] += 1
        source_counts[str(row.get("source") or "unknown")] += 1
        state_counts[str(row.get("state") or "unknown")] += 1
        left, top = int(row["xmin"]), int(row["ymin"])
        right, bottom = int(row["xmax"]), int(row["ymax"])
        if right <= left or bottom <= top:
            invalid_boxes += 1
        if left < 0 or top < 0 or right > width or bottom > height:
            out_of_bounds += 1
        crop_area_ratios.append(max(0.0, (right - left) * (bottom - top) / max(width * height, 1)))

    ratios = sorted(crop_area_ratios)
    median_ratio = ratios[len(ratios) // 2] if ratios else None
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "path": str(path),
        "sha256": sha256(path),
        "compressed_bytes": path.stat().st_size,
        "embedded_image_bytes": image_bytes,
        "rows": len(rows),
        "columns": table.column_names,
        "unique_original_filenames": len({str(row.get("orig_filename")) for row in rows}),
        "unique_plate_text": len({str(row.get("plate_text")) for row in rows}),
        "source_counts": dict(source_counts),
        "state_counts_top20": dict(state_counts.most_common(20)),
        "unique_embedded_dimensions": len(dimension_counts),
        "most_common_embedded_dimensions": {
            f"{width}x{height}": count for (width, height), count in dimension_counts.most_common(10)
        },
        "invalid_boxes": invalid_boxes,
        "boxes_outside_embedded_image": out_of_bounds,
        "boxes_outside_embedded_image_rate": out_of_bounds / len(rows) if rows else None,
        "box_area_ratio_median_against_embedded_crop": median_ratio,
        "training_recommendation": (
            "Do not train a detector from this parquet: the embedded images are plate crops, "
            "but the coordinates are in the original-scene coordinate system. Obtain the "
            "original scenes and a clear license first."
        ),
        "license_status": "not_declared_on_dataset_card",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parquet", nargs="?", default="data/raw/online_hf/zenitsu09_indian_number_plate/train.parquet")
    parser.add_argument("--output", default="artifacts/evaluations/online_hf_zenitsu_plate_audit_2026-08-28.json")
    args = parser.parse_args()
    result = audit(Path(args.parquet))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
