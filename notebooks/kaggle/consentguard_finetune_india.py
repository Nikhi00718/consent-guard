"""Kaggle entrypoint for CCPD2020 -> Indian plate fine-tuning."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
REPO = WORK / "consentguard"


def _ensure_p100_compatible_torch() -> None:
    """Install a CUDA build that still includes Tesla P100 (sm_60) kernels."""
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import torch; "
                "print(int(torch.cuda.is_available())); "
                "print(torch.cuda.get_device_capability(0)[0] if torch.cuda.is_available() else -1)"
            ),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    lines = probe.stdout.strip().splitlines()
    cuda_available = bool(lines and lines[0] == "1")
    major = int(lines[1]) if len(lines) > 1 and lines[1].lstrip("-").isdigit() else -1
    print(f"Initial CUDA probe: available={cuda_available}, compute_major={major}", flush=True)
    if cuda_available and major >= 7:
        return

    print("Installing P100-compatible torch 2.7.1/cu118 wheels...", flush=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--disable-pip-version-check",
            "--no-cache-dir",
            "--force-reinstall",
            "--index-url",
            "https://download.pytorch.org/whl/cu118",
            "--extra-index-url",
            "https://pypi.org/simple",
            "torch==2.7.1+cu118",
            "torchvision==0.22.1+cu118",
        ],
        check=True,
    )


def _mount(fragment: str) -> Path:
    # Kaggle may expose attached datasets directly under /kaggle/input or
    # beneath /kaggle/input/datasets/<owner>/<slug>. Search both layouts.
    fragment = fragment.lower()
    matches = sorted(
        path
        for path in INPUT.rglob("*")
        if path.is_dir() and fragment in path.name.lower()
    )
    if not matches:
        candidates = [str(path) for path in INPUT.rglob("*") if path.is_dir()]
        raise FileNotFoundError(
            f"No Kaggle input mount contains {fragment!r}; candidates={candidates}"
        )
    return matches[0]


def _extract_code() -> None:
    mount = _mount("consentguard-training-code-india-finetune-v1")
    archives = sorted(mount.rglob("*.zip"))
    if REPO.exists():
        shutil.rmtree(REPO)
    REPO.mkdir(parents=True, exist_ok=True)
    if archives:
        with zipfile.ZipFile(archives[0]) as archive:
            archive.extractall(REPO)
        return

    # Datasets uploaded with Kaggle's --dir-mode zip are mounted as their
    # individual files, so there is no .zip file to extract.
    project_root = next(
        (path.parent for path in mount.rglob("main_project") if path.is_dir()),
        None,
    )
    if project_root is None:
        files = [str(path.relative_to(mount)) for path in mount.rglob("*")][:40]
        raise FileNotFoundError(
            f"No code archive or project tree found below {mount}; files={files}"
        )
    shutil.copytree(project_root, REPO, dirs_exist_ok=True)


def main() -> None:
    print("Kaggle input mounts:", [path.name for path in INPUT.iterdir()], flush=True)
    _ensure_p100_compatible_torch()
    _extract_code()
    # The Kaggle kernel is intentionally offline.  Installing the editable
    # project would create an isolated build environment and try to download
    # setuptools; direct script execution already adds main_project/src to
    # sys.path, so no package installation is needed.
    run_env = os.environ.copy()
    run_env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO / "main_project/src"), run_env.get("PYTHONPATH", "")]
    )

    indian_mount = _mount("indian-license-plates-with-labels")
    checkpoint_mount = _mount("consentguard-ccpd2020-checkpoint-v1")
    checkpoints = sorted(checkpoint_mount.rglob("best.pt"))
    if len(checkpoints) != 1:
        raise RuntimeError(f"Expected exactly one CCPD checkpoint, found {checkpoints}")

    preparer = REPO / "main_project/scripts/stage_03_specialists/prepare_external_specialist.py"
    records = REPO / "data/processed/external/indian_license_plates_with_labels"
    subprocess.run(
        [
            sys.executable,
            str(preparer),
            "--component",
            "plate",
            "--source-root",
            str(indian_mount),
            "--output",
            str(records),
            "--seed",
            "1337",
            "--plate-format",
            "yolo",
        ],
        cwd=REPO,
        env=run_env,
        check=True,
    )

    fine_tuner = REPO / "main_project/scripts/stage_03_specialists/fine_tune_plate_from_checkpoint.py"
    config = REPO / "main_project/configs/stage_03_specialists/train_plate_ccpd2020_india_finetune_5ep.yaml"
    subprocess.run(
        [
            sys.executable,
            str(fine_tuner),
            "--config",
            str(config),
            "--init-checkpoint",
            str(checkpoints[0]),
            "--device",
            "cuda",
        ],
        cwd=REPO,
        env=run_env,
        check=True,
    )


if __name__ == "__main__":
    main()
