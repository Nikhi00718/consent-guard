"""Kaggle entrypoint for the audited full-scene plate adaptation candidate."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
REPO = WORK / "consentguard"
DEFAULT_CONFIG = "main_project/configs/stage_03_specialists/train_plate_full_scene_research_v1_highres_5ep.yaml"


def _one(pattern: str) -> Path:
    matches = sorted(INPUT.rglob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one {pattern!r} below /kaggle/input, found: {matches}")
    return matches[0]


transport_manifest = _one("transport_manifest.json")
transport_root = transport_manifest.parent
transport = json.loads(transport_manifest.read_text(encoding="utf-8"))
if transport.get("test_split_used") is not False:
    raise RuntimeError("Transport manifest does not keep the test split locked")
if transport.get("cross_split_hash_leakage") != 0:
    raise RuntimeError("Transport manifest reports cross-split image leakage")

code_archives = sorted(INPUT.rglob("consentguard-training-code-plate-full-scene-v1.zip"))
if len(code_archives) == 1:
    REPO.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(code_archives[0]) as archive:
        archive.extractall(REPO)
    code_delivery = {"mode": "archive", "source": str(code_archives[0])}
elif not code_archives:
    # Kaggle expands ZIP files inside dataset versions created with
    # ``--dir-mode zip``.  Locate the unique extracted project root and copy it
    # into /kaggle/working so checkpoints and metrics are writable outputs.
    project_files = sorted(INPUT.rglob("pyproject.toml"))
    project_roots = [
        path.parent
        for path in project_files
        if (path.parent / DEFAULT_CONFIG).is_file()
    ]
    if len(project_roots) != 1:
        raise RuntimeError(
            "Expected one extracted ConsentGuard code root below /kaggle/input, "
            f"found: {project_roots}"
        )
    shutil.copytree(project_roots[0], REPO)
    code_delivery = {"mode": "expanded_dataset", "source": str(project_roots[0])}
else:
    raise RuntimeError(f"Expected at most one code archive below /kaggle/input, found: {code_archives}")

data_link = REPO / "data"
data_link.symlink_to(transport_root / "data", target_is_directory=True)
checkpoint = transport_root / transport["initialization_checkpoint"]
if not checkpoint.is_file():
    raise FileNotFoundError(f"Initialization checkpoint is missing: {checkpoint}")

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", str(REPO)], check=True)
config = os.environ.get("CONSENTGUARD_PLATE_CONFIG", DEFAULT_CONFIG)
command = [
    sys.executable,
    str(REPO / "main_project/scripts/stage_03_specialists/fine_tune_plate_from_checkpoint.py"),
    "--config",
    str(REPO / config),
    "--init-checkpoint",
    str(checkpoint),
    "--device",
    "cuda",
]
print(
    json.dumps(
        {"code_delivery": code_delivery, "command": command, "transport": transport},
        indent=2,
        sort_keys=True,
    ),
    flush=True,
)
subprocess.run(command, cwd=REPO, check=True)
