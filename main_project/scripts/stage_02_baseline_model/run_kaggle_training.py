"""Run reproducible ConsentGuard training components on a Kaggle GPU.

Run one component per Kaggle session by default.  ``--component all`` is
available for a deliberate long run, but the handbook recommends separate
checkpointed sessions because free sessions are time-limited.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[3]
TRAINER = ROOT / "main_project" / "scripts" / "stage_02_baseline_model" / "train_maskrcnn.py"
COMPONENT_CONFIGS = {
    "baseline": ROOT / "main_project/configs/stage_02_baseline_model/train_maskrcnn_moderate_v2_negatives_10ep.yaml",
    "face": ROOT / "main_project/configs/stage_03_specialists/train_face_maskrcnn_5ep.yaml",
    "plate": ROOT / "main_project/configs/stage_03_specialists/train_plate_maskrcnn_5ep.yaml",
    "handwriting": ROOT / "main_project/configs/stage_03_specialists/train_handwriting_maskrcnn_5ep.yaml",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ensure_data_link(data_root: Path) -> None:
    """Expose a read-only Kaggle Dataset under the repo-relative data path."""

    source = data_root / "data" / "processed"
    target = ROOT / "data" / "processed"
    if not source.is_dir():
        raise FileNotFoundError(f"Expected Kaggle Dataset directory: {source}")
    if target.is_symlink():
        if target.resolve() != source.resolve():
            target.unlink()
        else:
            return
    if target.exists():
        # A checked-out repo may already contain a matching data tree.
        if (target / "visual_redactions_verified_visual_v2_negatives").is_dir():
            return
        raise RuntimeError(f"Refusing to overwrite existing data directory: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(source, target_is_directory=True)


def _write_config(component: str, seed: int, output_dir: Path) -> Path:
    source = COMPONENT_CONFIGS[component]
    values = yaml.safe_load(source.read_text(encoding="utf-8"))
    values.setdefault("experiment", {})["seed"] = int(seed)
    values["experiment"]["output_dir"] = str(output_dir.relative_to(ROOT)).replace("\\", "/")
    values.setdefault("training", {})["device"] = "cuda"
    destination = ROOT / "artifacts" / "kaggle" / "configs" / f"{component}_seed{seed}.yaml"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")
    return destination


def run_component(component: str, seed: int, *, epochs: int | None) -> dict[str, Any]:
    output_dir = ROOT / "artifacts" / "checkpoints" / f"kaggle_{component}_seed{seed}"
    config = _write_config(component, seed, output_dir)
    log_path = ROOT / "artifacts" / "kaggle" / "logs" / f"{component}_seed{seed}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(TRAINER), "--config", str(config), "--device", "cuda"]
    if epochs is not None:
        command.extend(("--epochs", str(epochs)))
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            command,
            cwd=ROOT,
            env={**os.environ, "PYTHONPATH": str(ROOT / "main_project" / "src")},
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return {
        "component": component,
        "seed": seed,
        "config": str(config.relative_to(ROOT)).replace("\\", "/"),
        "config_sha256": _sha256(config),
        "output_dir": str(output_dir.relative_to(ROOT)).replace("\\", "/"),
        "log": str(log_path.relative_to(ROOT)).replace("\\", "/"),
        "return_code": result.returncode,
        "duration_seconds": round(time.time() - started, 2),
        "passed": result.returncode == 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component", choices=(*COMPONENT_CONFIGS, "all"), default="baseline")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1337])
    parser.add_argument("--data-root", type=Path, required=True, help="Kaggle Dataset mount containing data/processed")
    parser.add_argument("--epochs", type=int, help="override the config epoch count")
    parser.add_argument("--manifest", type=Path, default=Path("reports/kaggle_training_manifest.json"))
    args = parser.parse_args()
    if any(seed < 0 for seed in args.seeds):
        parser.error("seeds must be non-negative")
    if args.epochs is not None and args.epochs < 1:
        parser.error("--epochs must be positive")
    data_root = args.data_root if args.data_root.is_absolute() else ROOT / args.data_root
    _ensure_data_link(data_root)
    components = list(COMPONENT_CONFIGS) if args.component == "all" else [args.component]
    runs = [run_component(component, seed, epochs=args.epochs) for component in components for seed in args.seeds]
    manifest = {
        "schema_version": "consentguard-kaggle-training-v1",
        "test_split_used": False,
        "components": components,
        "seeds": args.seeds,
        "runs": runs,
        "passed": all(run["passed"] for run in runs),
    }
    output = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    if not manifest["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
