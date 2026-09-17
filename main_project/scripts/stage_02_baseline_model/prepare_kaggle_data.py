"""Stage the licensed V2 train/validation data for a Kaggle Dataset.

The script preserves the relative paths embedded in the JSONL records, copies
only train/validation images, and refuses to admit any test record.  By
default it performs a dry-run and writes a small manifest; pass ``--copy`` to
materialize the approximately 10 GiB train/validation package.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[3]
GLOBAL_ROOT = ROOT / "data" / "processed" / "visual_redactions_verified_visual_v2_negatives"
SOURCES = {
    "baseline": GLOBAL_ROOT,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _decoded_image_identity(path: Path) -> tuple[str, int, int]:
    """Return a stable hash of the orientation-corrected RGB pixels."""

    with Image.open(path) as image:
        canonical = ImageOps.exif_transpose(image).convert("RGB")
        digest = hashlib.sha256(canonical.tobytes()).hexdigest()
        width, height = canonical.size
    return digest, width, height


def _image_manifest_entry(item: tuple[str, Path]) -> tuple[str, Path, dict[str, Any]]:
    """Read an image once and derive both transport and decoded identities."""

    relative, source = item
    payload = source.read_bytes()
    with Image.open(io.BytesIO(payload)) as image:
        canonical = ImageOps.exif_transpose(image).convert("RGB")
        pixel_sha256 = hashlib.sha256(canonical.tobytes()).hexdigest()
        width, height = canonical.size
    return relative, source, {
        "path": relative,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "pixel_sha256": pixel_sha256,
        "width": width,
        "height": height,
    }


def _records(source: Path) -> list[Path]:
    return [source / "records_train2017.jsonl", source / "records_val2017.jsonl"]


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
    items = sorted(image_paths.items())
    workers = min(8, max(1, os.cpu_count() or 1))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        prepared = executor.map(_image_manifest_entry, items)
        for index, (relative, source, entry) in enumerate(prepared, start=1):
            total_bytes += int(entry["bytes"])
            entries.append(entry)
            if copy_files:
                destination = output / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
            if index % 250 == 0 or index == len(items):
                print(f"Prepared baseline identities: {index}/{len(items)}", flush=True)

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
