"""Validate the ConsentGuard dataset downloads without modifying them."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[3]
RAW_ROOT = WORKSPACE / "data" / "raw"

VISPR_SPLITS = {
    "train2017": 10_000,
    "val2017": 4_167,
    "test2017": 8_000,
}

REDACTION_FILE_SIZES = {
    "train2017.json": 71_097_278,
    "val2017.json": 30_457_149,
    "test2017.json": 57_474_452,
    "train2017-extra.zip": 65_824_250,
    "val2017-extra.zip": 27_110_759,
    "test2017-extra.zip": 51_760_164,
}


def status(label: str, state: str, detail: str) -> None:
    print(f"[{state:<7}] {label}: {detail}")


def validate_vispr_annotations() -> dict[str, str]:
    vispr_root = RAW_ROOT / "vispr"
    image_to_split: dict[str, str] = {}

    for split, expected_count in VISPR_SPLITS.items():
        split_root = vispr_root / split
        files = list(split_root.glob("*.json")) if split_root.is_dir() else []
        state = "OK" if len(files) == expected_count else "FAIL"
        status(
            f"VISPR {split} annotations",
            state,
            f"{len(files):,}/{expected_count:,} JSON files",
        )
        for path in files:
            image_to_split[path.stem] = split

    return image_to_split


def validate_redactions(image_to_vispr_split: dict[str, str]) -> None:
    redactions_root = RAW_ROOT / "visual_redactions"

    for file_name, expected_size in REDACTION_FILE_SIZES.items():
        path = redactions_root / file_name
        if not path.exists():
            status(f"Visual Redactions {file_name}", "PENDING", "not downloaded yet")
            continue
        current_size = path.stat().st_size
        state = "OK" if current_size == expected_size else "PENDING"
        status(
            f"Visual Redactions {file_name}",
            state,
            f"{current_size:,}/{expected_size:,} bytes",
        )

    for split in VISPR_SPLITS:
        path = redactions_root / f"{split}.json"
        expected_size = REDACTION_FILE_SIZES[path.name]
        if not path.exists() or path.stat().st_size != expected_size:
            continue

        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        records = payload["annotations"]
        instances = [
            attribute
            for record in records.values()
            for attribute in record["attributes"]
        ]
        classes = Counter(attribute["attr_id"] for attribute in instances)
        missing_from_vispr = [
            image_id for image_id in records if image_id not in image_to_vispr_split
        ]
        moved_between_splits = Counter(
            image_to_vispr_split[image_id]
            for image_id in records
            if image_id in image_to_vispr_split
            and image_to_vispr_split[image_id] != split
        )
        status(
            f"Visual Redactions {split} JSON",
            "OK" if not missing_from_vispr else "FAIL",
            (
                f"{len(records):,} images, {len(instances):,} instances, "
                f"{len(classes)} attribute IDs, "
                f"{len(missing_from_vispr)} IDs absent from VISPR, "
                f"cross-split VISPR IDs={dict(moved_between_splits)}"
            ),
        )

        extra_root = redactions_root / "annotations-extra" / split
        if extra_root.is_dir():
            extra_files = list(extra_root.rglob("*.json"))
            status(
                f"Visual Redactions {split} extracted weak metadata",
                "INFO",
                f"{len(extra_files):,} JSON files",
            )


def validate_vpd_inventory() -> None:
    inventory_path = RAW_ROOT / "vpd_public" / "vpd_download_dry_run.txt"
    if not inventory_path.exists():
        status("VPD public inventory", "PENDING", "dry-run inventory not found")
        return

    extensions: Counter[str] = Counter()
    listed_files = 0
    for line in inventory_path.read_text(encoding="utf-8").splitlines():
        if "\t" not in line or line.startswith("file\t"):
            continue
        file_name = line.split("\t", 1)[0]
        extensions[Path(file_name).suffix.lower() or "<none>"] += 1
        listed_files += 1

    video_count = sum(extensions[extension] for extension in (".mp4", ".mov", ".mkv"))
    annotation_count = sum(
        extensions[extension] for extension in (".json", ".jsonl", ".xml", ".yaml", ".yml")
    )
    status(
        "VPD public inventory",
        "INFO",
        (
            f"{listed_files:,} files; {video_count:,} videos; "
            f"{annotation_count} JSON/XML/YAML annotation files; "
            f"extensions={dict(extensions.most_common())}"
        ),
    )


def main() -> None:
    print(f"Workspace: {WORKSPACE}")
    image_to_vispr_split = validate_vispr_annotations()
    validate_redactions(image_to_vispr_split)
    validate_vpd_inventory()


if __name__ == "__main__":
    main()
