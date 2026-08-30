from __future__ import annotations

import sys
from pathlib import Path


SCRIPT_ROOT = Path(__file__).resolve().parents[2] / "scripts" / "stage_03_specialists"
sys.path.insert(0, str(SCRIPT_ROOT))

import audit_yolo_detection_dataset as audit_module  # noqa: E402
import prepare_grouped_yolo_plate as grouped_module  # noqa: E402


def test_asset_keys_group_numbered_and_timestamped_video_frames() -> None:
    assert audit_module._asset_keys(Path("video12-0042.jpg"))[1] == "video12"
    assert audit_module._asset_keys(Path("road.mp4-t-12_frame-8.rf.abcdef0123456789.jpg"))[1] == "road"
    assert audit_module._asset_keys(Path("independent-still.jpg"))[1] == ""


def test_yolo_label_parser_accepts_valid_box_and_rejects_overflow(tmp_path: Path) -> None:
    label = tmp_path / "label.txt"
    label.write_text("0 0.5 0.5 0.4 0.2\n0 0.95 0.5 0.2 0.2\n", encoding="utf-8")

    boxes, errors = audit_module._parse_label(label, 100, 80)

    assert len(boxes) == 1
    assert boxes[0]["pixel_width"] == 40
    assert boxes[0]["pixel_height"] == 16
    assert errors == ["line 2: box extends outside image"]


def test_group_assignment_is_deterministic_and_never_splits_a_group() -> None:
    rows = [
        {"group_id": "same"},
        {"group_id": "same"},
        {"group_id": "a"},
        {"group_id": "b"},
        {"group_id": "c"},
        {"group_id": "d"},
        {"group_id": "e"},
    ]

    first, assigned, targets = grouped_module._assign_groups(rows, 1337, 0.2, 0.2)
    second, _, _ = grouped_module._assign_groups(rows, 1337, 0.2, 0.2)

    assert first == second
    assert set(first) == {"same", "a", "b", "c", "d", "e"}
    assert sum(assigned.values()) == len(rows)
    assert sum(targets.values()) == len(rows)
    assert first["same"] in {"train", "val", "test"}
