"""Stage and optionally publish the full-scene plate adaptation Kaggle job.

The transport contains only the audited train/validation records and the
images referenced by those records.  It intentionally excludes every test
record.  Files are hard-linked on Windows when possible so staging does not
duplicate several gigabytes of image data on the local disk.

Authentication is delegated to Kaggle's private ``~/.kaggle/kaggle.json``;
this script never accepts or prints a token.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_USERNAME = "nikhil00718"
DATASET_SLUG = "consentguard-plate-full-scene-v1"
CODE_DATASET_SLUG = "consentguard-plate-full-scene-code-v1"
KERNEL_SLUG = "consentguard-plate-full-scene-training"
RECORD_ROOT = ROOT / "data/processed/external/plate_full_scene_research_v1"
INIT_CHECKPOINT = ROOT / "artifacts/checkpoints/specialist_plate_ccpd2020_india_finetune_5ep/best.pt"
CODE_ARCHIVE = ROOT / "artifacts/kaggle/consentguard-training-code-plate-full-scene-v1.zip"
STAGE_ROOT = ROOT / "artifacts/kaggle/consentguard-plate-full-scene-v1"
CODE_STAGE = ROOT / "artifacts/kaggle/consentguard-plate-full-scene-code-v1"
KERNEL_STAGE = ROOT / "artifacts/kaggle/kernels/plate_full_scene"
FIXED_TIME = (2026, 1, 1, 0, 0, 0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _link_or_copy(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size != source.stat().st_size:
            raise RuntimeError(f"Existing staged file has a different size: {destination}")
        return "existing"
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def _write_metadata(path: Path, *, dataset_id: str, title: str, description: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "id": dataset_id,
                "title": title,
                "isPrivate": True,
                "licenses": [{"name": "other"}],
                "description": description,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _add_file(archive: zipfile.ZipFile, path: Path) -> None:
    relative = path.relative_to(ROOT).as_posix()
    info = zipfile.ZipInfo(relative, FIXED_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    archive.writestr(info, path.read_bytes())


def build_code_archive() -> dict[str, Any]:
    required = [ROOT / "pyproject.toml"]
    required.extend(sorted((ROOT / "main_project/src").rglob("*.py")))
    required.append(ROOT / "main_project/scripts/stage_03_specialists/fine_tune_plate_from_checkpoint.py")
    required.extend(
        [
            ROOT / "main_project/configs/stage_03_specialists/train_plate_full_scene_research_v1_5ep.yaml",
            ROOT / "main_project/configs/stage_03_specialists/train_plate_full_scene_research_v1_highres_5ep.yaml",
        ]
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Kaggle code inputs are missing: {missing}")
    CODE_ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(CODE_ARCHIVE, "w") as archive:
        for path in sorted(set(required), key=lambda item: item.as_posix()):
            _add_file(archive, path)
    return {
        "path": str(CODE_ARCHIVE),
        "bytes": CODE_ARCHIVE.stat().st_size,
        "sha256": _sha256(CODE_ARCHIVE),
        "files": len(set(required)),
    }


def stage_data(username: str) -> dict[str, Any]:
    record_paths = [RECORD_ROOT / "records_train.jsonl", RECORD_ROOT / "records_val.jsonl"]
    support_paths = [RECORD_ROOT / "class_map.json", RECORD_ROOT / "manifest.json"]
    missing = [str(path) for path in [*record_paths, *support_paths, INIT_CHECKPOINT] if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Plate transport inputs are missing: {missing}")

    all_records: list[dict[str, Any]] = []
    records_by_split: dict[str, list[dict[str, Any]]] = {}
    split_counts: dict[str, int] = {}
    for path in record_paths:
        split = path.stem.removeprefix("records_")
        rows = []
        for original in _read_records(path):
            source_path = Path(original["image_path"])
            resolved = (source_path if source_path.is_absolute() else ROOT / source_path).resolve(strict=True)
            try:
                relative = resolved.relative_to(ROOT.resolve())
            except ValueError as error:
                raise RuntimeError(f"Record image is outside the repository: {resolved}") from error
            row = dict(original)
            row["image_path"] = relative.as_posix()
            rows.append(row)
        records_by_split[split] = rows
        split_counts[split] = len(rows)
        all_records.extend(rows)
    image_paths = sorted({Path(row["image_path"]) for row in all_records}, key=lambda path: path.as_posix())
    expected_hashes = {Path(row["image_path"]): row.get("image_sha256") for row in all_records}
    if any("test" in path.as_posix().lower() for path in record_paths):
        raise RuntimeError("A test records file entered the Kaggle transport")

    modes: dict[str, int] = {"hardlink": 0, "copy": 0, "existing": 0}
    total_image_bytes = 0
    for index, relative in enumerate(image_paths, start=1):
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError(f"Unsafe image path in records: {relative}")
        source = ROOT / relative
        if not source.is_file():
            raise FileNotFoundError(f"Record image is missing: {source}")
        expected = expected_hashes[relative]
        actual = _sha256(source)
        if expected and actual.lower() != str(expected).lower():
            raise RuntimeError(f"Image hash mismatch: {relative}")
        modes[_link_or_copy(source, STAGE_ROOT / relative)] += 1
        total_image_bytes += source.stat().st_size
        if index % 500 == 0 or index == len(image_paths):
            print(f"Staged and verified images: {index}/{len(image_paths)}", flush=True)

    staged_record_hashes = {}
    for split, rows in records_by_split.items():
        destination = STAGE_ROOT / RECORD_ROOT.relative_to(ROOT) / f"records_{split}.jsonl"
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        staged_record_hashes[split] = _sha256(destination)
    for path in support_paths:
        relative = path.relative_to(ROOT)
        modes[_link_or_copy(path, STAGE_ROOT / relative)] += 1
    checkpoint_relative = Path("artifacts/checkpoints/plate_initialization/best.pt")
    modes[_link_or_copy(INIT_CHECKPOINT, STAGE_ROOT / checkpoint_relative)] += 1

    source_manifest = json.loads((RECORD_ROOT / "manifest.json").read_text(encoding="utf-8"))
    if source_manifest.get("test_split_used") is not False:
        raise RuntimeError("Source manifest does not prove that the test split stayed locked")
    transport = {
        "schema_version": "consentguard-plate-kaggle-transport-v1",
        "dataset_id": f"{username}/{DATASET_SLUG}",
        "records": split_counts,
        "record_sha256": staged_record_hashes,
        "unique_images": len(image_paths),
        "image_bytes": total_image_bytes,
        "initialization_checkpoint": checkpoint_relative.as_posix(),
        "initialization_checkpoint_bytes": INIT_CHECKPOINT.stat().st_size,
        "initialization_checkpoint_sha256": _sha256(INIT_CHECKPOINT),
        "source_records_manifest_sha256": _sha256(RECORD_ROOT / "manifest.json"),
        "staging_modes": modes,
        "cross_split_hash_leakage": source_manifest.get("cross_split_hash_leakage"),
        "test_split_used": False,
    }
    (STAGE_ROOT / "transport_manifest.json").write_text(
        json.dumps(transport, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_metadata(
        STAGE_ROOT,
        dataset_id=f"{username}/{DATASET_SLUG}",
        title="ConsentGuard plate full-scene research v1",
        description=(
            "Private audited train/validation transport for the ConsentGuard plate adaptation experiment. "
            "No test records are included; original source licenses and terms remain authoritative."
        ),
    )
    return transport


def stage_code_and_kernel(username: str, code: dict[str, Any]) -> None:
    CODE_STAGE.mkdir(parents=True, exist_ok=True)
    _link_or_copy(CODE_ARCHIVE, CODE_STAGE / CODE_ARCHIVE.name)
    _write_metadata(
        CODE_STAGE,
        dataset_id=f"{username}/{CODE_DATASET_SLUG}",
        title="ConsentGuard plate full-scene training code v1",
        description="Deterministic code/config bundle for the private full-scene plate adaptation run.",
    )
    KERNEL_STAGE.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "notebooks/kaggle/consentguard_plate_full_scene_train.py", KERNEL_STAGE / "train.py")
    metadata = {
        "id": f"{username}/{KERNEL_SLUG}",
        "title": "ConsentGuard plate full-scene training",
        "code_file": "train.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [f"{username}/{DATASET_SLUG}", f"{username}/{CODE_DATASET_SLUG}"],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (KERNEL_STAGE / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (KERNEL_STAGE / "staged_code.json").write_text(json.dumps(code, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _kaggle(*args: str) -> None:
    subprocess.run([sys.executable, "-m", "kaggle", *args], cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--username", default=DEFAULT_USERNAME)
    parser.add_argument("--upload", action="store_true", help="Create both private Kaggle datasets")
    parser.add_argument("--push", action="store_true", help="Push the GPU kernel after both datasets exist")
    args = parser.parse_args()
    if args.push and not args.upload:
        parser.error("--push requires --upload for the first publication")
    if args.upload:
        credential = Path.home() / ".kaggle/kaggle.json"
        if not credential.is_file():
            raise FileNotFoundError(f"Kaggle credential is missing: {credential}")

    code = build_code_archive()
    transport = stage_data(args.username)
    stage_code_and_kernel(args.username, code)
    if args.upload:
        _kaggle("datasets", "create", "-p", str(STAGE_ROOT), "--dir-mode", "zip")
        _kaggle("datasets", "create", "-p", str(CODE_STAGE), "--dir-mode", "zip")
    if args.push:
        _kaggle("kernels", "push", "-p", str(KERNEL_STAGE))
    print(
        json.dumps(
            {
                "code": code,
                "transport": transport,
                "data_dataset": f"{args.username}/{DATASET_SLUG}",
                "code_dataset": f"{args.username}/{CODE_DATASET_SLUG}",
                "kernel": f"{args.username}/{KERNEL_SLUG}",
                "uploaded": args.upload,
                "kernel_pushed": args.push,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
