"""Convert model-specific public datasets to ConsentGuard training records.

The converters deliberately keep each specialist on its own source domain:
WIDER FACE for faces, an India-specific YOLO dataset for plates, and HierText
for handwriting polygons. They never read the ConsentGuard locked test split.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageOps


CLASS_NAMES = {
    "face": "a105_face_all",
    "plate": "a108_license_plate_all",
    "handwriting": "a26_handwriting",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rectangle(x: float, y: float, width: float, height: float) -> list[float]:
    return [x, y, x + width, y, x + width, y + height, x, y + height]


def _instance(
    *,
    instance_id: int,
    class_name: str,
    bbox: tuple[float, float, float, float],
    polygon: list[float] | None = None,
) -> dict[str, Any]:
    x, y, width, height = bbox
    points = polygon or _rectangle(x, y, width, height)
    return {
        "area_pixels": float(max(0.0, width) * max(0.0, height)),
        "attr_id": class_name,
        "bbox": [float(x), float(y), float(width), float(height)],
        "class_id": 1,
        "instance_id": int(instance_id),
        "iscrowd": False,
        "official_eval_ignore_small": False,
        "polygons": [points],
        "source_area_pixels": float(max(0.0, width) * max(0.0, height)),
    }


def _record(
    *,
    image_path: Path,
    image_id: str,
    width: int,
    height: int,
    split: str,
    source: str,
    instances: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "excluded_instances": [],
        "geometry_status": "external_source",
        "geometry_tolerance_fraction": 0.0,
        "height": int(height),
        "image_id": image_id,
        "image_path": str(image_path.resolve()),
        "image_release": source,
        "image_split": split,
        "instances": instances,
        "negative_for_profile": not instances,
        "redactions_split": split,
        "source_annotation_height": int(height),
        "source_annotation_width": int(width),
        "specialist_negative": not instances,
        "specialist_source_class_id": 1,
        "width": int(width),
    }


def _find_one(root: Path, names: Iterable[str]) -> Path:
    for name in names:
        direct = root / name
        if direct.is_file():
            return direct
    candidates = [path for path in root.rglob("*") if path.is_file() and path.name in set(names)]
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected one of {list(names)} under {root}; found {len(candidates)} matches")
    return candidates[0]


def _find_image(root: Path, relative: str, split: str) -> Path:
    candidates = (
        root / relative,
        root / f"WIDER_{split}" / "images" / relative,
        root / f"WIDER_{split.lower()}" / "images" / relative,
        root / split / relative,
        root / f"{Path(relative).stem}.jpg",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    matches = list(root.rglob(Path(relative).name))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"Could not resolve image {relative!r} under {root}")


def _parse_wider_annotations(path: Path) -> list[tuple[str, list[tuple[float, float, float, float]]]]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    parsed: list[tuple[str, list[tuple[float, float, float, float]]]] = []
    index = 0
    while index < len(lines):
        relative = lines[index].strip()
        index += 1
        if not relative:
            continue
        count = int(lines[index].strip())
        index += 1
        boxes: list[tuple[float, float, float, float]] = []
        # Some WIDER FACE mirrors retain an all-zero placeholder row after an
        # image whose declared face count is zero. Consume it before reading
        # the next image record.
        if count == 0 and index < len(lines):
            placeholder = lines[index].split()
            try:
                numeric_placeholder = len(placeholder) >= 4 and all(
                    float(value) == 0.0 for value in placeholder
                )
            except ValueError:
                numeric_placeholder = False
            if numeric_placeholder:
                index += 1
        for _ in range(count):
            values = [float(value) for value in lines[index].split()]
            index += 1
            if len(values) < 4:
                continue
            x, y, width, height = values[:4]
            invalid = int(values[7]) if len(values) > 7 else 0
            if invalid or width <= 1 or height <= 1:
                continue
            boxes.append((x, y, width, height))
        parsed.append((relative, boxes))
    return parsed


def convert_widerface(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    class_name = CLASS_NAMES["face"]
    results: dict[str, list[dict[str, Any]]] = {}
    for split, annotation_name in (
        ("train", "wider_face_train_bbx_gt.txt"),
        ("val", "wider_face_val_bbx_gt.txt"),
    ):
        annotation = _find_one(root, (annotation_name,))
        records = []
        for relative, boxes in _parse_wider_annotations(annotation):
            image_path = _find_image(root, relative, split)
            with Image.open(image_path) as image:
                width, height = image.size
            instances = [
                _instance(instance_id=i, class_name=class_name, bbox=box)
                for i, box in enumerate(boxes)
            ]
            records.append(
                _record(
                    image_path=image_path,
                    image_id=f"widerface-{split}-{Path(relative).stem}",
                    width=width,
                    height=height,
                    split=split,
                    source="WIDER_FACE",
                    instances=instances,
                )
            )
        results[split] = records
    return results["train"], results["val"]


def _plate_pairs(root: Path) -> list[tuple[Path, Path]]:
    image_root = next((path for path in (root / "images", root / "Images") if path.is_dir()), root)
    label_root = next((path for path in (root / "labels", root / "Labels") if path.is_dir()), root)
    pairs = []
    for image in sorted(path for path in image_root.rglob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}):
        relative = image.relative_to(image_root).with_suffix(".txt")
        label = label_root / relative
        if label.is_file():
            pairs.append((image, label))
    if not pairs:
        raise FileNotFoundError(f"No image/YOLO-label pairs found under {root}")
    return pairs


def convert_indian_plates(root: Path, *, seed: int = 1337, validation_fraction: float = 0.2) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not 0.05 <= validation_fraction <= 0.5:
        raise ValueError("validation_fraction must be in [0.05, 0.5]")
    class_name = CLASS_NAMES["plate"]
    pairs = _plate_pairs(root)
    indices = list(range(len(pairs)))
    random.Random(seed).shuffle(indices)
    validation_count = max(1, round(len(indices) * validation_fraction))
    validation_indices = set(indices[:validation_count])
    results = {"train": [], "val": []}
    for pair_index, (image_path, label_path) in enumerate(pairs):
        split = "val" if pair_index in validation_indices else "train"
        with Image.open(image_path) as image:
            width, height = ImageOps.exif_transpose(image).size
        instances = []
        for instance_id, line in enumerate(label_path.read_text(encoding="utf-8", errors="replace").splitlines()):
            if not line.strip():
                continue
            values = [float(value) for value in line.split()]
            if len(values) < 5:
                continue
            _, centre_x, centre_y, box_width, box_height = values[:5]
            x = (centre_x - box_width / 2.0) * width
            y = (centre_y - box_height / 2.0) * height
            w = box_width * width
            h = box_height * height
            x = max(0.0, min(float(width), x))
            y = max(0.0, min(float(height), y))
            w = max(0.0, min(float(width) - x, w))
            h = max(0.0, min(float(height) - y, h))
            if w > 1 and h > 1:
                instances.append(_instance(instance_id=instance_id, class_name=class_name, bbox=(x, y, w, h)))
        results[split].append(
            _record(
                image_path=image_path,
                image_id=f"indian-plate-{image_path.stem}",
                width=width,
                height=height,
                split=split,
                source="kaggle-kedarsai-indian-license-plates-with-labels",
                instances=instances,
            )
        )
    return results["train"], results["val"]


def _read_hiertext(path: Path) -> Iterable[dict[str, Any]]:
    """Stream the large pretty-printed ``annotations`` array without loading it all."""

    opener = gzip.open if path.suffix == ".gz" else open
    decoder = json.JSONDecoder()
    with opener(path, "rt", encoding="utf-8") as handle:
        buffer = ""
        array_started = False
        eof = False
        while True:
            if not eof and (not array_started or len(buffer) < 1024 * 1024):
                chunk = handle.read(1024 * 1024)
                if chunk:
                    buffer += chunk
                else:
                    eof = True
            if not array_started:
                key_index = buffer.find('"annotations"')
                if key_index < 0:
                    if eof:
                        raise ValueError(f"HierText file has no annotations array: {path}")
                    buffer = buffer[-64:]
                    continue
                array_index = buffer.find("[", key_index)
                if array_index < 0:
                    if eof:
                        raise ValueError(f"HierText annotations array is malformed: {path}")
                    continue
                buffer = buffer[array_index + 1 :]
                array_started = True
            buffer = buffer.lstrip(" \t\r\n,")
            if buffer.startswith("]"):
                return
            try:
                item, consumed = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                if eof:
                    raise ValueError(f"Could not decode HierText annotations from {path}")
                chunk = handle.read(1024 * 1024)
                if chunk:
                    buffer += chunk
                else:
                    eof = True
                continue
            if not isinstance(item, dict):
                raise ValueError(f"HierText annotation must be an object, received {type(item).__name__}")
            yield item
            buffer = buffer[consumed:]


def _hiertext_image(root: Path, split: str, image_id: str) -> Path:
    names = (f"{image_id}.jpg", f"{image_id}.jpeg", f"{image_id}.png")
    for directory in (root / split, root / "images" / split, root):
        for name in names:
            candidate = directory / name
            if candidate.is_file():
                return candidate
    matches = [path for name in names for path in root.rglob(name)]
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"Could not resolve HierText image {image_id} under {root}")


def convert_hiertext(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    class_name = CLASS_NAMES["handwriting"]
    results: dict[str, list[dict[str, Any]]] = {}
    for split, names in (
        ("train", ("train.jsonl", "train.jsonl.gz")),
        ("validation", ("validation.jsonl", "validation.jsonl.gz")),
    ):
        annotation_path = _find_one(root, names)
        records = []
        for annotation in _read_hiertext(annotation_path):
            image_id = str(annotation["image_id"])
            image_path = _hiertext_image(root, split, image_id)
            width = int(annotation.get("image_width") or 0)
            height = int(annotation.get("image_height") or 0)
            if width < 1 or height < 1:
                with Image.open(image_path) as image:
                    width, height = image.size
            instances = []
            for paragraph in annotation.get("paragraphs", []):
                for line in paragraph.get("lines", []):
                    if not bool(line.get("handwritten", False)):
                        continue
                    points = [(float(x), float(y)) for x, y in line.get("vertices", [])]
                    if len(points) < 3:
                        continue
                    xs = [point[0] for point in points]
                    ys = [point[1] for point in points]
                    x, y = min(xs), min(ys)
                    w, h = max(xs) - x, max(ys) - y
                    if w <= 1 or h <= 1:
                        continue
                    polygon = [coordinate for point in points for coordinate in point]
                    instances.append(
                        _instance(
                            instance_id=len(instances),
                            class_name=class_name,
                            bbox=(x, y, w, h),
                            polygon=polygon,
                        )
                    )
            records.append(
                _record(
                    image_path=image_path,
                    image_id=f"hiertext-{split}-{image_id}",
                    width=width,
                    height=height,
                    split=split,
                    source="HierText",
                    instances=instances,
                )
            )
        results[split] = records
    return results["train"], results["validation"]


def write_records(component: str, output: Path, train: list[dict[str, Any]], validation: list[dict[str, Any]]) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    class_name = CLASS_NAMES[component]
    class_map_path = output / "class_map.json"
    class_map_path.write_text(json.dumps({"background": 0, class_name: 1}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summaries = {}
    for split, records in (("train", train), ("val", validation)):
        path = output / f"records_{split}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
        summaries[split] = {
            "records": len(records),
            "positive_images": sum(bool(record["instances"]) for record in records),
            "negative_images": sum(not record["instances"] for record in records),
            "instances": sum(len(record["instances"]) for record in records),
            "sha256": _sha256(path),
        }
    manifest = {
        "schema_version": "external-specialist-records-v1",
        "component": component,
        "class_name": class_name,
        "test_split_used": False,
        "rectangle_masks_from_boxes": component in {"face", "plate"},
        "splits": summaries,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component", choices=sorted(CLASS_NAMES), required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()
    source = args.source_root.resolve()
    if args.component == "face":
        train, validation = convert_widerface(source)
    elif args.component == "plate":
        train, validation = convert_indian_plates(source, seed=args.seed)
    else:
        train, validation = convert_hiertext(source)
    print(json.dumps(write_records(args.component, args.output.resolve(), train, validation), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
