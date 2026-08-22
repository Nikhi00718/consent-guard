from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def test_manifest_builder_cannot_scan_vispr_pixels() -> None:
    source = (
        ROOT / "main_project" / "scripts" / "stage_01_data" / "build_master_manifest.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    string_literals = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    assert "vispr" not in {value.casefold() for value in string_literals}
    assert '"visual_redactions" / "images" / split' in source


def test_processed_validator_requires_release_and_split_provenance() -> None:
    source = (
        ROOT / "main_project" / "scripts" / "stage_01_data" / "validate_processed_records.py"
    ).read_text(encoding="utf-8")
    assert 'record.get("image_release") != "visual_redactions_v1"' in source
    assert 'record.get("image_split") != expected_split' in source
    assert "image_path_outside_same_release_split" in source
