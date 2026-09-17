from __future__ import annotations

import importlib.util
from pathlib import Path

from PIL import Image


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "stage_02_baseline_model"
    / "prepare_kaggle_data.py"
)
SPEC = importlib.util.spec_from_file_location("prepare_kaggle_data", SCRIPT)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def test_decoded_image_identity_is_metadata_independent(tmp_path: Path) -> None:
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    image = Image.new("RGB", (8, 6), (10, 20, 30))
    image.save(first)
    image.save(second, pnginfo=None)

    assert module._decoded_image_identity(first) == module._decoded_image_identity(second)


def test_decoded_image_identity_applies_exif_orientation(tmp_path: Path) -> None:
    path = tmp_path / "rotated.jpg"
    exif = Image.Exif()
    exif[274] = 6
    Image.new("RGB", (12, 8), "white").save(path, exif=exif)

    _, width, height = module._decoded_image_identity(path)
    assert (width, height) == (8, 12)
