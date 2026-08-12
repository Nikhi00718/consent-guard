"""Validate complete VISPR archives, extract safely, and rebuild records."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "vispr"
INTERIM = ROOT / "data" / "interim" / "vispr"
EXPECTED_BYTES = {
    "train": 22_377_737_446,
    "val": 9_435_240_408,
    "test": 17_367_722_420,
}


def validate_members(
    archive: tarfile.TarFile,
    destination: Path,
    expected_root_name: str,
) -> tuple[dict[str, int], int]:
    destination_resolved = destination.resolve()
    files: dict[str, int] = {}
    total_bytes = 0
    for member in archive:
        member_path = Path(member.name)
        if not member_path.parts or member_path.parts[0] != expected_root_name:
            raise RuntimeError(
                f"Archive member is outside expected root {expected_root_name!r}: {member.name}"
            )
        target = (destination / member.name).resolve()
        if target != destination_resolved and destination_resolved not in target.parents:
            raise RuntimeError(f"Unsafe archive path: {member.name}")
        if member.issym() or member.islnk():
            raise RuntimeError(f"Links are not allowed in dataset archives: {member.name}")
        if not member.isfile() and not member.isdir():
            raise RuntimeError(f"Special archive member is not allowed: {member.name}")
        if member.isfile():
            relative = member_path.relative_to(expected_root_name).as_posix()
            if relative in files:
                raise RuntimeError(f"Duplicate archive member: {member.name}")
            files[relative] = int(member.size)
            total_bytes += member.size
    return files, total_bytes


def verify_extracted_files(root: Path, expected_files: dict[str, int]) -> None:
    actual_files = {
        path.relative_to(root).as_posix(): path.stat().st_size
        for path in root.rglob("*")
        if path.is_file()
    }
    if actual_files != expected_files:
        missing = sorted(set(expected_files) - set(actual_files))[:5]
        extra = sorted(set(actual_files) - set(expected_files))[:5]
        wrong_size = sorted(
            path
            for path in set(actual_files) & set(expected_files)
            if actual_files[path] != expected_files[path]
        )[:5]
        raise RuntimeError(
            "Extracted directory does not match the validated archive manifest: "
            f"missing={missing}, extra={extra}, wrong_size={wrong_size}"
        )


def process_split(split: str, extract: bool) -> dict:
    archive_path = RAW / f"{split}2017.tar.gz"
    if not archive_path.is_file():
        raise FileNotFoundError(f"Missing archive: {archive_path}")
    actual = archive_path.stat().st_size
    expected = EXPECTED_BYTES[split]
    if actual != expected:
        raise RuntimeError(f"{split} archive is incomplete: {actual:,}/{expected:,} bytes")
    expected_root_name = f"{split}2017"
    with tarfile.open(archive_path, mode="r:gz") as archive:
        expected_files, expanded_bytes = validate_members(
            archive,
            INTERIM,
            expected_root_name,
        )
    result = {
        "split": split,
        "archive": str(archive_path),
        "archive_bytes": actual,
        "members_validated": len(expected_files),
        "expanded_file_bytes": expanded_bytes,
        "extracted": False,
    }
    if extract:
        INTERIM.mkdir(parents=True, exist_ok=True)
        output_directory = INTERIM / expected_root_name
        if output_directory.exists():
            if not output_directory.is_dir():
                raise RuntimeError(f"Extraction target is not a directory: {output_directory}")
            verify_extracted_files(output_directory, expected_files)
            result["reused_existing_extraction"] = True
        else:
            required_free = expanded_bytes + 2 * 1024**3
            free_bytes = shutil.disk_usage(INTERIM).free
            if free_bytes < required_free:
                raise RuntimeError(
                    f"Insufficient free space for safe staged extraction: {free_bytes:,} available, "
                    f"{required_free:,} required"
                )
            staging_root = Path(tempfile.mkdtemp(prefix=f".extract-{split}-", dir=INTERIM))
            try:
                with tarfile.open(archive_path, mode="r:gz") as archive:
                    archive.extractall(staging_root, filter="data")
                staged_directory = staging_root / expected_root_name
                if not staged_directory.is_dir():
                    raise RuntimeError(
                        f"Archive did not create expected directory: {staged_directory}"
                    )
                verify_extracted_files(staged_directory, expected_files)
                os.replace(staged_directory, output_directory)
            finally:
                # This is a process-owned unique staging path inside INTERIM;
                # never remove a user-provided or final dataset directory.
                if staging_root.exists():
                    shutil.rmtree(staging_root)
        result["extracted"] = True
        result["extracted_images"] = len(expected_files)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "val", "test", "all"), default="all")
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--rebuild-records", action="store_true")
    args = parser.parse_args()
    splits = tuple(EXPECTED_BYTES) if args.split == "all" else (args.split,)
    results = [process_split(split, args.extract) for split in splits]
    if args.rebuild_records:
        for script in ("build_master_manifest.py", "preprocess_visual_redactions.py", "validate_processed_records.py"):
            subprocess.run([sys.executable, str(ROOT / "scripts" / script)], cwd=ROOT, check=True)
    print(json.dumps({"passed": True, "results": results, "records_rebuilt": args.rebuild_records}, indent=2))


if __name__ == "__main__":
    main()
