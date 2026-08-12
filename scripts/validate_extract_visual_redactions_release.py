"""Validate and safely extract the official Visual Redactions v1 image release."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tarfile
import time
from collections import Counter
from pathlib import Path, PurePosixPath

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_ROOT = ROOT / "data" / "raw" / "visual_redactions" / "image_archives"
ANNOTATION_ROOT = ROOT / "data" / "raw" / "visual_redactions"
IMAGE_ROOT = ROOT / "data" / "raw" / "visual_redactions" / "images"
STAGING_ROOT = ROOT / "data" / "interim" / "visual_redactions_release_extract"
REPORT_PATH = ROOT / "reports" / "visual_redactions_release_validation.json"
EXPECTED_BYTES = {
    "train2017": 7_816_171_895,
    "val2017": 3_158_549_744,
    "test2017": 5_633_037_591,
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}


def safe_members(members: list[tarfile.TarInfo], split: str) -> None:
    for member in members:
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts:
            raise RuntimeError(f"Unsafe archive path in {split}: {member.name}")
        if member.issym() or member.islnk() or member.isdev():
            raise RuntimeError(f"Unsupported archive entry in {split}: {member.name}")


def annotation_records(split: str) -> dict[str, dict]:
    payload = json.loads((ANNOTATION_ROOT / f"{split}.json").read_text(encoding="utf-8"))
    records = payload.get("annotations")
    if not isinstance(records, dict):
        raise RuntimeError(f"Malformed annotation mapping for {split}")
    return records


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def validate_archive(split: str, extract: bool) -> dict:
    archive_path = ARCHIVE_ROOT / f"{split}.tar.gz"
    actual_bytes = archive_path.stat().st_size
    if actual_bytes != EXPECTED_BYTES[split]:
        raise RuntimeError(
            f"{split} byte mismatch: expected {EXPECTED_BYTES[split]}, got {actual_bytes}"
        )

    start = time.monotonic()
    with tarfile.open(archive_path, mode="r:gz") as archive:
        members = archive.getmembers()
        safe_members(members, split)
        image_members = [
            member
            for member in members
            if member.isfile() and Path(member.name).suffix.lower() in IMAGE_SUFFIXES
        ]
        if not image_members:
            raise RuntimeError(f"No image files found in {archive_path}")
        member_stems = [Path(member.name).stem for member in image_members]
        if len(member_stems) != len(set(member_stems)):
            duplicates = [name for name, count in Counter(member_stems).items() if count > 1]
            raise RuntimeError(f"Duplicate image IDs inside {split}: {duplicates[:10]}")

        annotations = annotation_records(split)
        annotation_ids = set(annotations)
        archive_ids = set(member_stems)
        if archive_ids != annotation_ids:
            missing = sorted(annotation_ids - archive_ids)
            extra = sorted(archive_ids - annotation_ids)
            raise RuntimeError(
                f"{split} image/annotation identity mismatch: "
                f"missing={missing[:10]}, extra={extra[:10]}"
            )

        if extract:
            stage = STAGING_ROOT / split
            destination = IMAGE_ROOT / split
            if destination.exists() and any(destination.iterdir()):
                raise RuntimeError(f"Refusing to overwrite non-empty destination: {destination}")
            if stage.exists():
                shutil.rmtree(stage)
            stage.mkdir(parents=True, exist_ok=True)
            archive.extractall(path=stage, members=members)

            extracted = [
                path
                for path in stage.rglob("*")
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            ]
            if len(extracted) != len(image_members):
                raise RuntimeError(
                    f"{split} extraction count mismatch: {len(extracted)} vs {len(image_members)}"
                )
            destination.mkdir(parents=True, exist_ok=True)
            for source in extracted:
                target = destination / source.name
                if target.exists():
                    raise RuntimeError(f"Duplicate extracted target: {target}")
                shutil.move(str(source), str(target))
            shutil.rmtree(stage)

    return {
        "split": split,
        "archive": str(archive_path),
        "archive_bytes": actual_bytes,
        "tar_members": len(members),
        "image_members": len(image_members),
        "annotation_records": len(annotation_ids),
        "archive_identity_exact": True,
        "archive_validation_seconds": round(time.monotonic() - start, 3),
    }


def decode_release() -> dict:
    rows = []
    errors = []
    geometry = Counter()
    seen_ids: dict[str, str] = {}
    for split in EXPECTED_BYTES:
        records = annotation_records(split)
        paths = sorted(
            path for path in (IMAGE_ROOT / split).iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
        )
        if len(paths) != len(records):
            raise RuntimeError(f"{split} extracted count is {len(paths)}, expected {len(records)}")
        for path in paths:
            image_id = path.stem
            if image_id in seen_ids:
                errors.append(
                    {"split": split, "image_id": image_id, "error": "cross_split_duplicate_id"}
                )
            seen_ids[image_id] = split
            record = records.get(image_id)
            if record is None:
                errors.append({"split": split, "image_id": image_id, "error": "missing_annotation"})
                continue
            try:
                with Image.open(path) as image:
                    image.load()
                    decoded_size = image.size
                    decoded_mode = image.mode
            except Exception as error:  # PIL exposes format-specific exception types.
                errors.append(
                    {"split": split, "image_id": image_id, "error": f"decode_failed: {error}"}
                )
                continue
            annotation_size = (int(record["image_width"]), int(record["image_height"]))
            if decoded_size == annotation_size:
                status = "exact"
            elif decoded_size == annotation_size[::-1]:
                status = "rotated"
            else:
                source_ratio = annotation_size[0] / annotation_size[1]
                decoded_ratio = decoded_size[0] / decoded_size[1]
                status = "aligned_resize" if abs(source_ratio - decoded_ratio) <= 0.01 else "mismatch"
            geometry[status] += 1
            rows.append(
                {
                    "split": split,
                    "image_id": image_id,
                    "width": decoded_size[0],
                    "height": decoded_size[1],
                    "mode": decoded_mode,
                    "geometry_status": status,
                }
            )
    if errors:
        raise RuntimeError(f"Image release decode/identity validation failed: {errors[:10]}")
    return {
        "decoded_images": len(rows),
        "decode_errors": 0,
        "unique_image_ids": len(seen_ids),
        "geometry_status": dict(geometry),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract", action="store_true", help="Extract after archive validation")
    args = parser.parse_args()
    results = [validate_archive(split, args.extract) for split in EXPECTED_BYTES]
    payload = {
        "status": "valid",
        "same_release_only": True,
        "archive_results": results,
        "extracted": args.extract,
    }
    if args.extract:
        payload["image_validation"] = decode_release()
    atomic_json(REPORT_PATH, payload)
    printable = dict(payload)
    if "image_validation" in printable:
        printable["image_validation"] = {
            key: value for key, value in printable["image_validation"].items() if key != "rows"
        }
    print(json.dumps(printable, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
