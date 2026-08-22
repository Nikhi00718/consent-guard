"""Profile processed training records for architecture and sampling decisions."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--records",
        type=Path,
        default=Path("data/processed/visual_redactions/records_train2017.jsonl"),
    )
    parser.add_argument("--output", type=Path, default=Path("reports/training_data_profile.json"))
    args = parser.parse_args()
    records_path = args.records if args.records.is_absolute() else ROOT / args.records
    output_path = args.output if args.output.is_absolute() else ROOT / args.output

    instance_counts: Counter[str] = Counter()
    image_counts: Counter[str] = Counter()
    size_counts: Counter[str] = Counter()
    dimensions: Counter[str] = Counter()
    records = 0
    total_instances = 0
    negative_images = 0
    maximum_instances = {"count": 0, "image_id": None}
    for line in records_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        records += 1
        dimensions[f"{record['width']}x{record['height']}"] += 1
        labels_in_image: set[str] = set()
        instance_total = len(record["instances"])
        if instance_total == 0:
            negative_images += 1
        total_instances += instance_total
        if instance_total > maximum_instances["count"]:
            maximum_instances = {"count": instance_total, "image_id": record["image_id"]}
        for instance in record["instances"]:
            label = str(instance["attr_id"])
            instance_counts[label] += 1
            labels_in_image.add(label)
            area = float(instance["area_pixels"])
            if area < 32**2:
                size_counts["small_lt_32_squared"] += 1
            elif area < 96**2:
                size_counts["medium_32_to_96_squared"] += 1
            else:
                size_counts["large_ge_96_squared"] += 1
        image_counts.update(labels_in_image)

    if not instance_counts:
        raise RuntimeError(f"No instances found in {records_path}")
    minimum_class, minimum_count = min(instance_counts.items(), key=lambda item: item[1])
    maximum_class, maximum_count = max(instance_counts.items(), key=lambda item: item[1])
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "records_path": str(records_path),
        "records_sha256": hashlib.sha256(records_path.read_bytes()).hexdigest(),
        "images": records,
        "negative_images": negative_images,
        "negative_image_fraction": negative_images / records,
        "instances": total_instances,
        "classes": len(instance_counts),
        "instances_per_image_mean": total_instances / records,
        "maximum_instances_in_one_image": maximum_instances,
        "instance_size_distribution": dict(size_counts),
        "instance_counts_by_class": dict(instance_counts.most_common()),
        "images_by_class": dict(image_counts.most_common()),
        "least_supported_class": {"class": minimum_class, "instances": minimum_count},
        "most_supported_class": {"class": maximum_class, "instances": maximum_count},
        "instance_imbalance_ratio": maximum_count / minimum_count,
        "unique_decoded_dimensions": len(dimensions),
        "most_common_dimensions": dict(dimensions.most_common(20)),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
