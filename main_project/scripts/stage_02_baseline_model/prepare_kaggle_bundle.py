"""Build a deterministic code/config bundle for a Kaggle GPU training session."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
FIXED_TIME = (2026, 1, 1, 0, 0, 0)


def add_file(archive: zipfile.ZipFile, path: Path) -> None:
    relative = path.relative_to(ROOT).as_posix()
    info = zipfile.ZipInfo(relative, FIXED_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    archive.writestr(info, path.read_bytes())


def build_bundle(config: Path, output: Path) -> dict:
    config = config if config.is_absolute() else ROOT / config
    output = output if output.is_absolute() else ROOT / output
    required = [ROOT / "pyproject.toml", config]
    required.extend(sorted((ROOT / "main_project" / "src").rglob("*.py")))
    required.extend(sorted((ROOT / "main_project" / "scripts" / "stage_02_baseline_model").glob("*.py")))
    required.append(ROOT / "main_project" / "scripts" / "stage_03_specialists" / "prepare_external_specialist.py")
    required.append(ROOT / "main_project" / "scripts" / "stage_03_specialists" / "fine_tune_plate_from_checkpoint.py")
    required.extend(sorted((ROOT / "main_project" / "configs" / "stage_03_specialists").glob("train_*.*yaml")))
    required.append(ROOT / "main_project" / "configs" / "kaggle" / "dataset_catalog.yaml")
    if any(not path.is_file() for path in required):
        raise FileNotFoundError("Kaggle bundle input is missing")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w") as archive:
        for path in sorted(set(required), key=lambda item: item.as_posix()):
            add_file(archive, path)
    return {
        "output": str(output),
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "files": len(set(required)),
        "config": str(config.relative_to(ROOT)),
        "data_included": False,
        "checkpoint_included": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/kaggle/stage02-training-code.zip"))
    args = parser.parse_args()
    print(json.dumps(build_bundle(args.config, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
