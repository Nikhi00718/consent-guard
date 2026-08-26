"""Upload prepared private Kaggle datasets and optionally push the GPU kernel.

The script never accepts a token argument. Kaggle authentication must already
exist in the user's private ``~/.kaggle/kaggle.json`` file.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
COMPONENTS = ("baseline", "face", "plate", "handwriting")
PUBLIC_SOURCES = {
    "baseline": [],
    "face": ["aiacademymaterials/wider-face-detection"],
    "plate": ["kedarsai/indian-license-plates-with-labels"],
    "handwriting": [],
}


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


def _kaggle(*args: str) -> None:
    subprocess.run([sys.executable, "-m", "kaggle", *args], cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--username", required=True)
    parser.add_argument("--upload-datasets", action="store_true")
    parser.add_argument("--push-kernel", action="store_true")
    args = parser.parse_args()

    credential_dir = Path.home() / ".kaggle"
    credential_candidates = (
        credential_dir / "access_token",
        credential_dir / "kaggle.json",
    )
    has_credential = any(path.is_file() for path in credential_candidates) or bool(
        os.environ.get("KAGGLE_API_TOKEN")
    )
    if (args.upload_datasets or args.push_kernel) and not has_credential:
        raise FileNotFoundError(
            f"Kaggle credential is missing under {credential_dir}. Create an access token in Kaggle account settings; never commit it."
        )

    data_dir = ROOT / "artifacts" / "kaggle" / "consentguard-v2-trainval"
    code_archive = ROOT / "artifacts" / "kaggle" / "consentguard-training-code.zip"
    if not data_dir.is_dir() or not code_archive.is_file():
        raise FileNotFoundError("Run prepare_kaggle_data.py --copy and prepare_kaggle_bundle.py first")

    processed_name = "visual_redactions_verified_visual_v2_negatives"
    processed_source = data_dir / "data" / "processed" / processed_name
    records_dir = ROOT / "artifacts" / "kaggle" / "consentguard-v2-records"
    shutil.copytree(
        processed_source,
        records_dir / "data" / "processed" / processed_name,
        dirs_exist_ok=True,
    )
    records_reports = records_dir / "reports"
    records_reports.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        ROOT / "reports" / "kaggle_trainval_data_manifest.json",
        records_reports / "kaggle_trainval_data_manifest.json",
    )
    _write_metadata(
        records_dir,
        dataset_id=f"{args.username}/consentguard-v2-records",
        title="ConsentGuard V2 verified train validation records",
        description=(
            "Private verified train and validation annotations plus per-image SHA-256 manifest. "
            "The locked test split is excluded; original Visual Redactions terms remain authoritative."
        ),
    )
    code_dir = ROOT / "artifacts" / "kaggle" / "consentguard-training-code"
    code_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(code_archive, code_dir / code_archive.name)
    _write_metadata(
        code_dir,
        dataset_id=f"{args.username}/consentguard-training-code",
        title="ConsentGuard reproducible training code",
        description="Private reproducible training code and configuration bundle for ConsentGuard research runs.",
    )

    kernel_source = ROOT / "notebooks" / "kaggle"
    kernel_template = json.loads((kernel_source / "kernel-metadata.json").read_text(encoding="utf-8"))
    kernel_stages: dict[str, Path] = {}
    entrypoint = (kernel_source / "consentguard_train.py").read_text(encoding="utf-8")
    marker = 'PACKAGED_COMPONENT = "baseline"'
    if marker not in entrypoint:
        raise RuntimeError("Kaggle entrypoint component marker is missing")
    for component in COMPONENTS:
        kernel_stage = ROOT / "artifacts" / "kaggle" / "kernels" / component
        kernel_stage.mkdir(parents=True, exist_ok=True)
        (kernel_stage / "consentguard_train.py").write_text(
            entrypoint.replace(marker, f'PACKAGED_COMPONENT = "{component}"', 1),
            encoding="utf-8",
        )
        (kernel_stage / "component.txt").write_text(component + "\n", encoding="utf-8")
        kernel = dict(kernel_template)
        kernel["id"] = f"{args.username}/consentguard-{component}-model-training"
        kernel["title"] = f"ConsentGuard {component} model training"
        private_sources = [f"{args.username}/consentguard-training-code"]
        if component == "baseline":
            private_sources.insert(0, f"{args.username}/consentguard-v2-records")
            private_sources.insert(1, f"{args.username}/consentguard-v2-image-patch")
            private_sources.append("meteharunakcay/visual-redactions")
        kernel["dataset_sources"] = private_sources + PUBLIC_SOURCES[component]
        (kernel_stage / "kernel-metadata.json").write_text(
            json.dumps(kernel, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        kernel_stages[component] = kernel_stage

    if args.upload_datasets:
        _kaggle("datasets", "create", "-p", str(records_dir), "--dir-mode", "zip")
        _kaggle("datasets", "create", "-p", str(code_dir), "--dir-mode", "zip")
    if args.push_kernel:
        if not args.upload_datasets:
            raise ValueError("--push-kernel requires --upload-datasets so its private sources exist")
        for component in COMPONENTS:
            _kaggle("kernels", "push", "-p", str(kernel_stages[component]))

    print(
        json.dumps(
            {
                "username": args.username,
                "data_dataset": f"{args.username}/consentguard-v2-records",
                "code_dataset": f"{args.username}/consentguard-training-code",
                "kernels": [
                    f"{args.username}/consentguard-{component}-model-training"
                    for component in COMPONENTS
                ],
                "uploaded": args.upload_datasets,
                "kernel_pushed": args.push_kernel,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
