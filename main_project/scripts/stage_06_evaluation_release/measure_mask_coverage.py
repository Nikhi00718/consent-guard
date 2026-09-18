"""Measure how much of each annotated private region the app actually erases.

Unlike the COCO metrics, this asks the only question that matters for a
redactor: after thresholds, fusion, expansion and dilation, is the annotated
region covered by the mask the user is about to burn in?

It drives the local HTTP API, so it measures the exact code path the interface
uses, including the tiled second pass. Start the app first:

    main_project\\scripts\\stage_05_review_export\\start_consentguard.ps1 -NoBrowser

Then, for example:

    python main_project/scripts/stage_06_evaluation_release/measure_mask_coverage.py ^
        --records data/processed/external/deepakat_indian_vehicle_number_plate_yolo_grouped/records_train.jsonl ^
        --max-images 20 --compare-single-pass ^
        --output reports/plate_mask_coverage.json

Records are read, never written. Pass a locked test split only when the
protocol is frozen.
"""

from __future__ import annotations

import argparse
import json
import statistics
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import requests
from PIL import Image

SINGLE_PASS_KEYS = [
    "global",
    "face-trained",
    "plate-trained",
    "handwriting-trained",
    "yunet-face",
    "lpd-yunet",
    "ppocr-text",
    "zxing-barcode",
]


def _boxes(record: dict[str, Any]) -> list[tuple[int, int, int, int]]:
    """Corner boxes from a processed record.

    ``bbox`` follows COCO's [x, y, width, height]; ``bbox_xyxy`` is corners.
    Mixing the two silently reports coverage for the wrong rectangle.
    """

    boxes = []
    for instance in record.get("instances", []):
        corners = instance.get("bbox_xyxy") or instance.get("box_xyxy")
        if corners and len(corners) == 4:
            left, top, right, bottom = (float(value) for value in corners)
        elif instance.get("bbox") and len(instance["bbox"]) == 4:
            left, top, width, height = (float(value) for value in instance["bbox"])
            right, bottom = left + width, top + height
        else:
            continue
        if right <= left or bottom <= top:
            continue
        boxes.append((int(left), int(top), int(np.ceil(right)), int(np.ceil(bottom))))
    return boxes


def _union_mask(base: str, image_path: Path, provider_keys: list[str] | None) -> np.ndarray:
    session = requests.post(f"{base}/v1/sessions", timeout=60).json()["session_id"]
    try:
        with image_path.open("rb") as handle:
            upload = requests.post(
                f"{base}/v1/sessions/{session}/assets",
                files={"asset": (image_path.name, handle, "application/octet-stream")},
                timeout=600,
            )
        upload.raise_for_status()
        payload: dict[str, Any] = {}
        if provider_keys is not None:
            payload["provider_keys"] = provider_keys
        analysis = requests.post(f"{base}/v1/sessions/{session}/analyze", json=payload, timeout=3600)
        analysis.raise_for_status()
        mask_response = requests.get(f"{base}/v1/sessions/{session}/masks/initial", timeout=300)
        mask_response.raise_for_status()
        with Image.open(BytesIO(mask_response.content)) as mask_image:
            return np.asarray(mask_image.convert("L"), dtype=np.uint8) > 0
    finally:
        requests.delete(f"{base}/v1/sessions/{session}", timeout=60)


def measure(
    base: str,
    records: list[dict[str, Any]],
    provider_keys: list[str] | None,
    *,
    coverage_threshold: float,
) -> dict[str, Any]:
    covered = instances = 0
    fractions: list[float] = []
    erased_fractions: list[float] = []
    failures: list[str] = []
    for record in records:
        image_path = Path(record["image_path"])
        if not image_path.is_absolute():
            image_path = Path("C:/consentGuard") / image_path
        boxes = _boxes(record)
        if not boxes or not image_path.is_file():
            continue
        try:
            mask = _union_mask(base, image_path, provider_keys)
        except Exception as error:  # noqa: BLE001 - measurement harness
            failures.append(f"{image_path.name}: {error!r}")
            continue
        erased_fractions.append(float(mask.mean()))
        for left, top, right, bottom in boxes:
            region = mask[max(0, top) : bottom, max(0, left) : right]
            if region.size == 0:
                continue
            instances += 1
            fraction = float(region.mean())
            fractions.append(fraction)
            if fraction >= coverage_threshold:
                covered += 1
    return {
        "providers": "all (includes tiled second pass)" if provider_keys is None else "single pass only",
        "images_measured": len(erased_fractions),
        "annotated_regions": instances,
        "regions_covered": covered,
        "coverage_rate": round(covered / instances, 4) if instances else None,
        "median_region_coverage": round(statistics.median(fractions), 4) if fractions else None,
        "mean_image_erased_fraction": round(float(np.mean(erased_fractions)), 4) if erased_fractions else None,
        "failures": failures,
    }


def measure_clean(base: str, records: list[dict[str, Any]], provider_keys: list[str] | None) -> dict[str, Any]:
    """On photos with nothing private in them, how much gets erased anyway?

    Coverage alone rewards erasing everything. This is the counterweight: a
    configuration that finds more plates by blacking out ordinary photos is
    not an improvement.
    """

    erased: list[float] = []
    failures: list[str] = []
    for record in records:
        image_path = Path(record["image_path"])
        if not image_path.is_absolute():
            image_path = Path("C:/consentGuard") / image_path
        if not image_path.is_file():
            continue
        try:
            erased.append(float(_union_mask(base, image_path, provider_keys).mean()))
        except Exception as error:  # noqa: BLE001 - measurement harness
            failures.append(f"{image_path.name}: {error!r}")
    touched = [fraction for fraction in erased if fraction > 0.0]
    return {
        "providers": "all (includes tiled second pass)" if provider_keys is None else "single pass only",
        "clean_images_measured": len(erased),
        "clean_images_with_any_erasure": len(touched),
        "false_alarm_rate": round(len(touched) / len(erased), 4) if erased else None,
        "mean_erased_fraction": round(float(np.mean(erased)), 4) if erased else None,
        "median_erased_fraction_when_touched": round(statistics.median(touched), 4) if touched else 0.0,
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--records", required=True, type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:7860")
    parser.add_argument("--max-images", type=int, default=20)
    parser.add_argument("--coverage-threshold", type=float, default=0.5)
    parser.add_argument("--compare-single-pass", action="store_true")
    parser.add_argument(
        "--clean-records",
        type=Path,
        help="Also measure false alarms on records marked negative_for_profile (nothing private).",
    )
    parser.add_argument("--max-clean-images", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.records, args.clean_records):
        if path is not None and "test" in path.name:
            parser.error("Refusing to measure a locked test split")

    rows = [json.loads(line) for line in args.records.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [row for row in rows if row.get("instances")][: args.max_images]
    report: dict[str, Any] = {
        "records": str(args.records),
        "coverage_threshold": args.coverage_threshold,
        "requested_images": args.max_images,
        "runs": [measure(args.base_url, rows, None, coverage_threshold=args.coverage_threshold)],
    }
    if args.compare_single_pass:
        report["runs"].append(
            measure(args.base_url, rows, SINGLE_PASS_KEYS, coverage_threshold=args.coverage_threshold)
        )
    if args.clean_records is not None:
        clean = [
            json.loads(line)
            for line in args.clean_records.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        clean = [row for row in clean if row.get("negative_for_profile") and not row.get("instances")]
        report["clean_records"] = str(args.clean_records)
        report["clean"] = measure_clean(args.base_url, clean[: args.max_clean_images], None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
