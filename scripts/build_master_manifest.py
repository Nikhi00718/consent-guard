"""Build a same-release Visual Redactions image/annotation manifest."""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MANIFESTS = ROOT / "data" / "manifests"


def write_json_atomic(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_jsonl_atomic(path: Path, rows: list[dict[str, object]]) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def rel(path: Path | None) -> str | None:
    return str(path.relative_to(ROOT)).replace("\\", "/") if path else None


def collect_images(split: str) -> dict[str, list[Path]]:
    """Collect pixels only from the matching Visual Redactions release split."""

    result: dict[str, list[Path]] = {}
    root = RAW / "visual_redactions" / "images" / split
    if not root.is_dir():
        return result
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".gif"}:
            result.setdefault(path.stem, []).append(path)
    return result


def main() -> None:
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for redaction_split in ("train2017", "val2017", "test2017"):
        image_paths = collect_images(redaction_split)
        source = RAW / "visual_redactions" / f"{redaction_split}.json"
        payload = json.loads(source.read_text(encoding="utf-8"))
        for image_id, record in sorted(payload["annotations"].items()):
            paths = sorted(image_paths.get(image_id, []), key=lambda p: str(p))
            attributes = record.get("attributes", [])
            rows.append(
                {
                    "image_id": image_id,
                    "redactions_split": redaction_split,
                    "image_release": "visual_redactions_v1",
                    "image_split": redaction_split,
                    "image_path": rel(paths[0]) if paths else None,
                    "duplicate_image_paths": [rel(path) for path in paths[1:]],
                    "image_width": record.get("image_width"),
                    "image_height": record.get("image_height"),
                    "instance_count": len(attributes),
                    "attribute_ids": sorted({item.get("attr_id") for item in attributes}),
                    "weak_metadata_path": rel(
                        RAW
                        / "visual_redactions"
                        / "annotations-extra"
                        / redaction_split
                        / f"{image_id}.json"
                    ),
                }
            )

    output = MANIFESTS / "visual_redactions_master.jsonl"
    write_jsonl_atomic(output, rows)

    summary = {
        "rows": len(rows),
        "available_images": sum(bool(row["image_path"]) for row in rows),
        "missing_images": sum(not row["image_path"] for row in rows),
        "duplicate_image_ids": sum(bool(row["duplicate_image_paths"]) for row in rows),
        "by_redactions_split": dict(Counter(str(row["redactions_split"]) for row in rows)),
        "image_release": "visual_redactions_v1",
        "same_release_only": True,
        "by_available": dict(
            Counter("available" if row["image_path"] else "missing" for row in rows)
        ),
        "manifest": rel(output),
    }
    write_json_atomic(MANIFESTS / "visual_redactions_master_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
