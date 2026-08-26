from __future__ import annotations

import gzip
import importlib.util
import json
from pathlib import Path

import cv2
from PIL import Image


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "stage_03_specialists"
    / "prepare_external_specialist.py"
)
SPEC = importlib.util.spec_from_file_location("prepare_external_specialist", SCRIPT)
assert SPEC and SPEC.loader
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def _image(path: Path, size: tuple[int, int] = (100, 80)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, "white").save(path)


def test_widerface_converter_uses_official_train_and_val(tmp_path: Path) -> None:
    for split in ("train", "val"):
        _image(tmp_path / f"WIDER_{split}" / "images" / "0--Parade" / f"{split}.jpg")
        annotation = tmp_path / "wider_face_split" / f"wider_face_{split}_bbx_gt.txt"
        annotation.parent.mkdir(parents=True, exist_ok=True)
        annotation.write_text(
            f"0--Parade/{split}.jpg\n1\n10 12 20 24 0 0 0 0 0 0\n",
            encoding="utf-8",
        )
    train, validation = module.convert_widerface(tmp_path)
    assert len(train) == len(validation) == 1
    assert train[0]["instances"][0]["attr_id"] == "a105_face_all"
    assert train[0]["instances"][0]["bbox"] == [10.0, 12.0, 20.0, 24.0]


def test_widerface_zero_count_placeholder_is_skipped(tmp_path: Path) -> None:
    annotation = tmp_path / "wider_face_train_bbx_gt.txt"
    annotation.write_text(
        "0--Parade/empty.jpg\n0\n0 0 0 0 0 0 0 0 0 0\n"
        "0--Parade/face.jpg\n1\n1 2 10 12 0 0 0 0 0 0\n",
        encoding="utf-8",
    )
    assert module._parse_wider_annotations(annotation) == [
        ("0--Parade/empty.jpg", []),
        ("0--Parade/face.jpg", [(1.0, 2.0, 10.0, 12.0)]),
    ]


def test_indian_plate_converter_is_deterministic_and_separate(tmp_path: Path) -> None:
    for index in range(10):
        _image(tmp_path / "images" / f"plate-{index}.jpg")
        label = tmp_path / "labels" / f"plate-{index}.txt"
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text("0 0.5 0.5 0.4 0.25\n", encoding="utf-8")
    first = module.convert_indian_plates(tmp_path, seed=1337)
    second = module.convert_indian_plates(tmp_path, seed=1337)
    assert [item["image_id"] for item in first[1]] == [item["image_id"] for item in second[1]]
    assert len(first[0]) == 8
    assert len(first[1]) == 2
    assert first[0][0]["instances"][0]["attr_id"] == "a108_license_plate_all"


def test_indian_plate_converter_applies_exif_orientation(tmp_path: Path) -> None:
    image_path = tmp_path / "images" / "rotated.jpg"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    exif = Image.Exif()
    exif[274] = 6
    Image.new("RGB", (100, 80), "white").save(image_path, exif=exif)
    label = tmp_path / "labels" / "rotated.txt"
    label.parent.mkdir(parents=True, exist_ok=True)
    label.write_text("0 0.5 0.5 0.4 0.25\n", encoding="utf-8")

    train, validation = module.convert_indian_plates(tmp_path, seed=1337)
    record = (train + validation)[0]
    decoded = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    assert decoded is not None
    assert (decoded.shape[1], decoded.shape[0]) == (record["width"], record["height"])
    assert (record["width"], record["height"]) == (80, 100)
    assert record["instances"][0]["bbox"] == [24.0, 37.5, 32.0, 25.0]


def test_hiertext_converter_keeps_only_handwritten_lines(tmp_path: Path) -> None:
    annotations = {}
    for split in ("train", "validation"):
        _image(tmp_path / split / f"{split}-image.jpg")
        payload = {
            "annotations": [
                {
                    "image_id": f"{split}-image",
                    "image_width": 100,
                    "image_height": 80,
                    "paragraphs": [
                        {
                            "lines": [
                                {
                                    "handwritten": True,
                                    "vertices": [[5, 5], [45, 5], [45, 20], [5, 20]],
                                },
                                {
                                    "handwritten": False,
                                    "vertices": [[5, 30], [45, 30], [45, 45], [5, 45]],
                                },
                            ]
                        }
                    ],
                }
            ]
        }
        path = tmp_path / "gt" / f"{split}.jsonl.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            json.dump(payload, handle)
        annotations[split] = path
    train, validation = module.convert_hiertext(tmp_path)
    assert len(train) == len(validation) == 1
    assert len(train[0]["instances"]) == 1
    assert train[0]["instances"][0]["attr_id"] == "a26_handwriting"


def test_manifest_is_fail_closed_to_test_data(tmp_path: Path) -> None:
    record = {
        "instances": [],
    }
    manifest = module.write_records("plate", tmp_path, [record], [record])
    assert manifest["test_split_used"] is False
    assert manifest["rectangle_masks_from_boxes"] is True
    assert json.loads((tmp_path / "class_map.json").read_text(encoding="utf-8"))["background"] == 0
