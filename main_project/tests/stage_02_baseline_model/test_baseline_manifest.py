from __future__ import annotations

from pathlib import Path

from importlib.util import module_from_spec, spec_from_file_location


ROOT = Path(__file__).resolve().parents[3]


def _verify_manifest(path: Path) -> dict:
    script = ROOT / "main_project" / "scripts" / "stage_02_baseline_model" / "verify_baseline.py"
    spec = spec_from_file_location("verify_baseline", script)
    assert spec and spec.loader
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.verify_manifest(path)


def test_frozen_baseline_artifacts_match_manifest() -> None:
    result = _verify_manifest(ROOT / "baselines" / "baseline-v0.1.json")
    assert result["baseline_id"] == "baseline-v0.1"
    assert result["passed"] is True
