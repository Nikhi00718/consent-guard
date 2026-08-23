"""Stage the licensed V2 train/validation data for a Kaggle Dataset.

The script preserves the relative paths embedded in the JSONL records, copies
only train/validation images, and refuses to admit any test record.  By
default it performs a dry-run and writes a small manifest; pass ``--copy`` to
materialize the approximately 10 GiB train/validation package.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
GLOBAL_ROOT = ROOT / "data" / "processed" / "visual_redactions_verified_visual_v2_negatives"
SPECIALIST_ROOT = ROOT / "data" / "processed" / "specialists"
SOURCES = {
    "global": GLOBAL_ROOT,
    "face": SPECIALIST_ROOT / "face",
    "plate": SPECIALIST_ROOT / "plate",
    "handwriting": SPECIALIST_ROOT / "handwriting",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _records(source: Path) -> list[Path]:
    return [source / "records_train2017.jsonl", source / "records_val2017.jsonl"] if source == GLOBAL_ROOT else [source / "records_train.jsonl", source / "records_val.jsonl"]


def build_manifest(output: Path, *, copy_files: bool) -> dict[str, Any]:
    image_paths: dict[str, Path] = {}
    record_files: list[Path] = []
    metadata_files: list[Path] = []
    for component, source in SOURCES.items():
        for record_path in _records(source):
            if not record_path.is_file():
                raise FileNotFoundError(record_path)
            record_files.append(record_path)
            for line in record_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                relative = Path(str(record["image_path"]))
                if "test" in relative.as_posix().lower():
                    raise ValueError(f"test-split record is not admissible: {relative}")
                resolved = ROOT / relative
                if not resolved.is_file():
                    raise FileNotFoundError(resolved)
                image_paths[relative.as_posix()] = resolved
        for name in ("class_map.json", "manifest.json"):
            path = source / name
            if path.is_file():
                metadata_files.append(path)

    entries = []
    total_bytes = 0
    for relative, source in sorted(image_paths.items()):
        size = source.stat().st_size
        total_bytes += size
        entry = {"path": relative, "bytes": size, "sha256": _sha256(source)}
        entries.append(entry)
        if copy_files:
            destination = output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

    if copy_files:
        for source in record_files + metadata_files:
            relative = source.relative_to(ROOT).as_posix()
            destination = output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

    manifest = {
        "schema_version": "consentguard-kaggle-data-v1",
        "copy_mode": "materialized" if copy_files else "dry_run",
        "test_split_used": False,
        "components": sorted(SOURCES),
        "record_files": [path.relative_to(ROOT).as_posix() for path in sorted(record_files)],
        "metadata_files": [path.relative_to(ROOT).as_posix() for path in sorted(metadata_files)],
        "unique_images": len(entries),
        "image_bytes": total_bytes,
        "image_gib": round(total_bytes / (1024**3), 3),
        "images": entries,
    }
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("reports/kaggle_trainval_data_manifest.json"))
    parser.add_argument("--dataset-dir", type=Path, default=Path("artifacts/kaggle/consentguard-v2-trainval"))
    parser.add_argument("--copy", action="store_true", help="materialize the train/validation package")
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    dataset_dir = args.dataset_dir if args.dataset_dir.is_absolute() else ROOT / args.dataset_dir
    manifest = build_manifest(dataset_dir, copy_files=args.copy)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in ("copy_mode", "test_split_used", "unique_images", "image_gib", "record_files")}, indent=2))


if __name__ == "__main__":
    main()
