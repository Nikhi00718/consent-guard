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
SPECIALIST_PREPARER = ROOT / "main_project" / "scripts" / "stage_03_specialists" / "prepare_external_specialist.py"
COMPONENT_CONFIGS = {
    "baseline": ROOT / "main_project/configs/stage_02_baseline_model/train_maskrcnn_moderate_v2_negatives_10ep.yaml",
    "face": ROOT / "main_project/configs/stage_03_specialists/train_face_widerface_fasterrcnn.yaml",
    "plate": ROOT / "main_project/configs/stage_03_specialists/train_plate_india_fasterrcnn.yaml",
    "plate_ccpd2020": ROOT / "main_project/configs/stage_03_specialists/train_plate_ccpd2020_fasterrcnn.yaml",
    "handwriting": ROOT / "main_project/configs/stage_03_specialists/train_handwriting_hiertext_maskrcnn.yaml",
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


def _write_config(
    component: str,
    seed: int,
    output_dir: Path,
    data_override: dict[str, str] | None = None,
) -> Path:
    source = COMPONENT_CONFIGS[component]
    values = yaml.safe_load(source.read_text(encoding="utf-8"))
    values.setdefault("experiment", {})["seed"] = int(seed)
    values["experiment"]["output_dir"] = str(output_dir.relative_to(ROOT)).replace("\\", "/")
    values.setdefault("training", {})["device"] = "cuda"
    if data_override:
        values.setdefault("data", {}).update(data_override)
    destination = ROOT / "artifacts" / "kaggle" / "configs" / f"{component}_seed{seed}.yaml"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")
    return destination


def run_component(
    component: str,
    seed: int,
    *,
    epochs: int | None,
    data_override: dict[str, str] | None = None,
) -> dict[str, Any]:
    output_dir = ROOT / "artifacts" / "checkpoints" / f"kaggle_{component}_seed{seed}"
    config = _write_config(component, seed, output_dir, data_override)
    log_path = ROOT / "artifacts" / "kaggle" / "logs" / f"{component}_seed{seed}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(TRAINER), "--config", str(config), "--device", "cuda"]
    if epochs is not None:
        command.extend(("--epochs", str(epochs)))
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env={
                **os.environ,
                "PYTHONPATH": str(ROOT / "main_project" / "src"),
                "PYTORCH_CUDA_ALLOC_CONF": os.environ.get(
                    "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
                ),
            },
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        return_code = process.wait()
    print(log_path.read_text(encoding="utf-8", errors="replace"), end="", flush=True)
    return {
        "component": component,
        "seed": seed,
        "config": str(config.relative_to(ROOT)).replace("\\", "/"),
        "config_sha256": _sha256(config),
        "output_dir": str(output_dir.relative_to(ROOT)).replace("\\", "/"),
        "log": str(log_path.relative_to(ROOT)).replace("\\", "/"),
        "return_code": return_code,
        "duration_seconds": round(time.time() - started, 2),
        "passed": return_code == 0,
    }


def _prepare_external(component: str, source_root: Path | None, seed: int) -> dict[str, str]:
    output = ROOT / "artifacts" / "kaggle" / "processed" / component
    expected = {
        "train_records": output / "records_train.jsonl",
        "val_records": output / "records_val.jsonl",
        "class_map": output / "class_map.json",
    }
    if source_root is not None:
        preparer_component = "plate" if component == "plate_ccpd2020" else component
        command = [
            sys.executable,
            str(SPECIALIST_PREPARER),
            "--component",
            preparer_component,
            "--source-root",
            str(source_root),
            "--output",
            str(output),
            "--seed",
            str(seed),
        ]
        if component == "plate_ccpd2020":
            command.extend(("--plate-format", "ccpd"))
        result = subprocess.run(
            command,
            cwd=ROOT,
            env={**os.environ, "PYTHONPATH": str(ROOT / "main_project" / "src")},
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to prepare the {component} specialist dataset")
    missing = [path for path in expected.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{component} requires --{component}-root or prebuilt records; missing: {missing}"
        )
    return {key: str(path) for key, path in expected.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component", choices=(*COMPONENT_CONFIGS, "all"), default="baseline")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1337])
    parser.add_argument("--data-root", type=Path, help="Private Kaggle Dataset mount containing data/processed")
    parser.add_argument("--face-root", type=Path, help="WIDER FACE Kaggle Dataset mount")
    parser.add_argument("--plate-root", type=Path, help="Indian plate Kaggle Dataset mount")
    parser.add_argument("--ccpd-root", type=Path, help="Official CCPD2020 Kaggle Dataset mount")
    parser.add_argument("--handwriting-root", type=Path, help="HierText train/validation root")
    parser.add_argument("--epochs", type=int, help="override the config epoch count")
    parser.add_argument("--manifest", type=Path, default=Path("reports/kaggle_training_manifest.json"))
    args = parser.parse_args()
    if any(seed < 0 for seed in args.seeds):
        parser.error("seeds must be non-negative")
    if args.epochs is not None and args.epochs < 1:
        parser.error("--epochs must be positive")
    components = list(COMPONENT_CONFIGS) if args.component == "all" else [args.component]
    if "baseline" in components:
        if args.data_root is None:
            parser.error("--data-root is required for baseline training")
        data_root = args.data_root if args.data_root.is_absolute() else ROOT / args.data_root
        _ensure_data_link(data_root)
    source_roots = {
        "face": args.face_root,
        "plate": args.plate_root,
        "plate_ccpd2020": args.ccpd_root,
        "handwriting": args.handwriting_root,
    }
    overrides = {
        component: _prepare_external(component, source_roots[component], args.seeds[0])
        for component in components
        if component != "baseline"
    }
    runs = [
        run_component(component, seed, epochs=args.epochs, data_override=overrides.get(component))
        for component in components
        for seed in args.seeds
    ]
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
