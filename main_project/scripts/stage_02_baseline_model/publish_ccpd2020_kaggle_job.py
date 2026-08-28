"""Stage and publish the dedicated CCPD2020 plate-training Kaggle job.

This publisher is intentionally separate from the legacy multi-model publisher:
the first new experiment must use official CCPD2020 data, not the older Indian
plate mirror.  It never accepts a token argument; Kaggle reads the user's
private ``~/.kaggle/kaggle.json`` credential.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
USERNAME = "nikhil00718"
CODE_DATASET = "consentguard-training-code-ccpd2020-v2"
KERNEL = "consentguard-plate-ccpd2020-training"


def _kaggle(*args: str) -> None:
    subprocess.run([sys.executable, "-m", "kaggle", *args], cwd=ROOT, check=True)


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


def stage_assets(username: str) -> tuple[Path, Path]:
    code_archive = ROOT / "artifacts/kaggle/consentguard-training-code-ccpd2020.zip"
    kernel_source = ROOT / "notebooks/kaggle"
    if not code_archive.is_file():
        raise FileNotFoundError(
            "Build consentguard-training-code-ccpd2020.zip with prepare_kaggle_bundle.py first"
        )

    code_dir = ROOT / "artifacts/kaggle" / CODE_DATASET
    code_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(code_archive, code_dir / code_archive.name)
    _write_metadata(
        code_dir,
        dataset_id=f"{username}/{CODE_DATASET}",
        title="ConsentGuard CCPD2020 training code",
        description="Reproducible ConsentGuard code and configuration for the CCPD2020 plate run.",
    )

    template = json.loads((kernel_source / "kernel-metadata.json").read_text(encoding="utf-8"))
    entrypoint = (kernel_source / "consentguard_train.py").read_text(encoding="utf-8")
    marker = 'PACKAGED_COMPONENT = "baseline"'
    if marker not in entrypoint:
        raise RuntimeError("Kaggle entrypoint component marker is missing")
    kernel_dir = ROOT / "artifacts/kaggle/kernels/plate_ccpd2020"
    kernel_dir.mkdir(parents=True, exist_ok=True)
    (kernel_dir / "consentguard_train.py").write_text(
        entrypoint.replace(marker, 'PACKAGED_COMPONENT = "plate_ccpd2020"', 1),
        encoding="utf-8",
    )
    (kernel_dir / "component.txt").write_text("plate_ccpd2020\n", encoding="utf-8")
    metadata = dict(template)
    metadata.update(
        {
            "id": f"{username}/{KERNEL}",
            "title": "ConsentGuard plate CCPD2020 training",
            "dataset_sources": [f"{username}/{CODE_DATASET}"],
        }
    )
    (kernel_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return code_dir, kernel_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--username", default=USERNAME)
    parser.add_argument("--upload", action="store_true", help="Create the private code dataset")
    parser.add_argument("--push", action="store_true", help="Push the dedicated Kaggle kernel")
    args = parser.parse_args()
    if args.upload or args.push:
        credential = Path.home() / ".kaggle" / "kaggle.json"
        if not credential.is_file():
            raise FileNotFoundError(f"Kaggle credential is missing: {credential}")
    code_dir, kernel_dir = stage_assets(args.username)
    if args.upload:
        _kaggle("datasets", "create", "-p", str(code_dir), "--dir-mode", "zip")
    if args.push:
        _kaggle("kernels", "push", "-p", str(kernel_dir))
    print(
        json.dumps(
            {
                "username": args.username,
                "ccpd_source": "Zenodo record 15647076 (downloaded and MD5-verified at runtime)",
                "code_dataset": f"{args.username}/{CODE_DATASET}",
                "kernel": f"{args.username}/{KERNEL}",
                "staged": True,
                "uploaded": args.upload,
                "kernel_pushed": args.push,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
