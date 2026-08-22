"""Comprehensive, reproducible ConsentGuard data and model audit.

This module powers the executed Jupyter notebook and produces machine-readable
tables plus publication-ready figures.  It deliberately keeps the released
test split out of model evaluation and qualitative model inspection.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib.patches import Patch
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from consentguard.shared.paths import project_path  # noqa: E402
from consentguard.stage_01_data.dataset import (  # noqa: E402
    VisualRedactionsDataset,
    detection_collate,
)
from consentguard.stage_02_baseline_model.config import load_training_config  # noqa: E402
from consentguard.stage_02_baseline_model.metrics import evaluate_instance_segmentation  # noqa: E402
from consentguard.stage_02_baseline_model.models import build_instance_segmentation_model  # noqa: E402


CONFIG_PATH = PROJECT_ROOT / "configs" / "train_maskrcnn_verified_visual.yaml"
CHECKPOINT_PATH = PROJECT_ROOT / "artifacts" / "checkpoints" / "maskrcnn_verified_visual_v3" / "best.pt"
TRAINING_RESULT_PATH = CHECKPOINT_PATH.parent / "training_result.json"
TRAINING_METRICS_PATH = CHECKPOINT_PATH.parent / "metrics.jsonl"
OVERFIT_RESULT_PATH = PROJECT_ROOT / "artifacts" / "checkpoints" / "maskrcnn_verified_overfit" / "training_result.json"
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed" / "visual_redactions_verified_visual"
ALIGNMENT_AUDIT_PATH = PROJECT_ROOT / "reports" / "visual_redactions_alignment_audit.json"
SPLIT_AUDIT_PATH = PROJECT_ROOT / "reports" / "split_leakage_audit.json"
PROCESSED_VALIDATION_PATH = PROJECT_ROOT / "reports" / "full_audit_processed_validation.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "reports" / "full_audit"

SPLITS = ("train2017", "val2017", "test2017")
AUDIT_SPLITS = ("train2017", "val2017")
PALETTE = {
    1: (244, 114, 182),
    2: (255, 159, 28),
    3: (0, 122, 255),
    4: (214, 39, 40),
    5: (148, 103, 189),
    6: (44, 160, 44),
    7: (140, 86, 75),
    8: (23, 190, 207),
    9: (188, 189, 34),
}
SHORT_NAMES = {
    "a105_face_all": "Face",
    "a108_license_plate_all": "License plate",
    "a109_person_body": "Person body",
    "a110_nudity_all": "Nudity",
    "a26_handwriting": "Handwriting",
    "a39_disability_physical": "Physical disability",
    "a43_medicine": "Medicine",
    "a7_fingerprint": "Fingerprint",
    "a8_signature": "Signature",
}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)!r}")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")
    os.replace(temporary, path)


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _model_shape(height: int, width: int, short_side: int = 640, max_long_side: int = 1024) -> tuple[int, int]:
    scale = short_side / float(min(height, width))
    if max(height, width) * scale > max_long_side:
        scale = max_long_side / float(max(height, width))
    return max(1, round(height * scale)), max(1, round(width * scale))


def _entropy(gray: np.ndarray) -> float:
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).reshape(-1)
    probability = hist / max(1.0, float(hist.sum()))
    probability = probability[probability > 0]
    return float(-(probability * np.log2(probability)).sum())


def _dhash(gray: np.ndarray) -> int:
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    comparisons = small[:, 1:] > small[:, :-1]
    value = 0
    for bit in comparisons.reshape(-1):
        value = (value << 1) | int(bit)
    return value


def _size_bucket(area: float) -> str:
    if area < 32.0**2:
        return "small"
    if area < 96.0**2:
        return "medium"
    return "large"


def _polygon_mask(
    instance: dict[str, Any],
    source_height: int,
    source_width: int,
    target_height: int,
    target_width: int,
) -> np.ndarray:
    mask = np.zeros((target_height, target_width), dtype=np.uint8)
    sx = target_width / float(source_width)
    sy = target_height / float(source_height)
    for polygon in instance["polygons"]:
        points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2).copy()
        points[:, 0] *= sx
        points[:, 1] *= sy
        cv2.fillPoly(mask, [np.round(points).astype(np.int32)], 1)
    return mask


def _safe_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def _progress(message: str) -> None:
    print(message, flush=True)


def load_records() -> tuple[dict[str, list[dict[str, Any]]], dict[str, int], dict[int, str]]:
    class_map = _load_json(PROCESSED_ROOT / "class_map.json")
    id_to_name = {int(class_id): str(name) for name, class_id in class_map.items() if int(class_id) != 0}
    records = {
        split: _load_jsonl(PROCESSED_ROOT / f"records_{split}.jsonl")
        for split in SPLITS
    }
    return records, class_map, id_to_name


def profile_records(
    records_by_split: dict[str, list[dict[str, Any]]],
    id_to_name: dict[int, str],
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    """Decode every selected image and profile every polygon instance."""

    image_rows: list[dict[str, Any]] = []
    instance_rows: list[dict[str, Any]] = []
    decode_failures: list[dict[str, Any]] = []
    total = sum(len(records) for records in records_by_split.values())
    completed = 0
    for split in SPLITS:
        for record in records_by_split[split]:
            completed += 1
            if completed == 1 or completed % 100 == 0 or completed == total:
                _progress(f"[data] decoded/profiled {completed}/{total} selected records")
            image_path = project_path(record["image_path"])
            bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if bgr is None:
                decode_failures.append({"split": split, "image_id": record["image_id"], "path": str(image_path)})
                continue
            height, width = bgr.shape[:2]
            expected = (int(record["height"]), int(record["width"]))
            dimension_match = (height, width) == expected
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            analysis_scale = min(1.0, 512.0 / max(height, width))
            analysis_width = max(32, round(width * analysis_scale))
            analysis_height = max(32, round(height * analysis_scale))
            gray_small = cv2.resize(gray, (analysis_width, analysis_height), interpolation=cv2.INTER_AREA)
            median = float(np.median(gray_small))
            low = int(max(0, 0.66 * median))
            high = int(min(255, max(low + 1, 1.33 * median)))
            edges = cv2.Canny(gray_small, low, high)
            no_edge = (edges == 0).astype(np.uint8)
            edge_distance = cv2.distanceTransform(no_edge, cv2.DIST_L2, 3)
            near_edge_baseline = float(np.mean(edge_distance <= 2.0))
            model_height, model_width = _model_shape(height, width)
            scale_x = model_width / float(width)
            scale_y = model_height / float(height)
            record_edge_support: list[float] = []
            record_edge_lift: list[float] = []
            record_oob_points = 0
            record_points = 0
            class_ids = sorted({int(instance["class_id"]) for instance in record["instances"]})

            for instance in record["instances"]:
                class_id = int(instance["class_id"])
                class_name = id_to_name[class_id]
                all_points = np.concatenate(
                    [np.asarray(polygon, dtype=np.float32).reshape(-1, 2) for polygon in instance["polygons"]],
                    axis=0,
                )
                oob = (
                    (all_points[:, 0] < 0)
                    | (all_points[:, 0] > width)
                    | (all_points[:, 1] < 0)
                    | (all_points[:, 1] > height)
                )
                oob_points = int(oob.sum())
                record_oob_points += oob_points
                record_points += int(len(all_points))

                mask = _polygon_mask(instance, height, width, analysis_height, analysis_width)
                boundary = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, np.ones((3, 3), dtype=np.uint8)) > 0
                if np.any(boundary):
                    edge_support = float(np.mean(edge_distance[boundary] <= 2.0))
                    edge_distance_median = float(np.median(edge_distance[boundary]))
                else:
                    edge_support = float("nan")
                    edge_distance_median = float("nan")
                edge_lift = edge_support / max(near_edge_baseline, 1e-6) if math.isfinite(edge_support) else float("nan")
                if math.isfinite(edge_support):
                    record_edge_support.append(edge_support)
                    record_edge_lift.append(edge_lift)

                x, y, box_width, box_height = (float(value) for value in instance["bbox"])
                model_area = float(instance.get("area_pixels", 0.0)) * scale_x * scale_y
                source_area_fraction = float(instance.get("area_pixels", 0.0)) / max(1.0, height * width)
                bbox_area = max(1e-6, box_width * box_height)
                border_margin_x = max(1.0, width * 0.01)
                border_margin_y = max(1.0, height * 0.01)
                touches_border = (
                    x <= border_margin_x
                    or y <= border_margin_y
                    or x + box_width >= width - border_margin_x
                    or y + box_height >= height - border_margin_y
                )
                instance_rows.append(
                    {
                        "split": split,
                        "image_id": str(record["image_id"]),
                        "class_id": class_id,
                        "class_name": class_name,
                        "class_short": SHORT_NAMES.get(class_name, class_name),
                        "source_area_pixels": float(instance.get("area_pixels", 0.0)),
                        "source_area_fraction": source_area_fraction,
                        "model_area_pixels": model_area,
                        "size_bucket": _size_bucket(model_area),
                        "bbox_width_fraction": box_width / max(1.0, width),
                        "bbox_height_fraction": box_height / max(1.0, height),
                        "bbox_aspect_ratio": box_width / max(box_height, 1e-6),
                        "mask_bbox_fill": float(instance.get("area_pixels", 0.0)) / bbox_area,
                        "polygon_count": len(instance["polygons"]),
                        "polygon_points": int(len(all_points)),
                        "oob_points": oob_points,
                        "touches_border": bool(touches_border),
                        "edge_support_2px": edge_support,
                        "edge_lift": edge_lift,
                        "boundary_edge_distance_median": edge_distance_median,
                    }
                )

            image_rows.append(
                {
                    "split": split,
                    "image_id": str(record["image_id"]),
                    "image_path": _safe_relative(Path(image_path)),
                    "width": width,
                    "height": height,
                    "megapixels": width * height / 1_000_000.0,
                    "aspect_ratio": width / max(1.0, height),
                    "model_width": model_width,
                    "model_height": model_height,
                    "dimension_match": bool(dimension_match),
                    "brightness_mean": float(gray_small.mean()),
                    "contrast_std": float(gray_small.std()),
                    "blur_laplacian_var": float(cv2.Laplacian(gray_small, cv2.CV_64F).var()),
                    "entropy_bits": _entropy(gray_small),
                    "dark_fraction": float(np.mean(gray_small < 20)),
                    "bright_fraction": float(np.mean(gray_small > 235)),
                    "edge_density": float(np.mean(edges > 0)),
                    "near_edge_baseline": near_edge_baseline,
                    "annotation_edge_support_median": float(np.median(record_edge_support)) if record_edge_support else float("nan"),
                    "annotation_edge_lift_median": float(np.median(record_edge_lift)) if record_edge_lift else float("nan"),
                    "instance_count": len(record["instances"]),
                    "class_count": len(class_ids),
                    "class_ids": ",".join(str(value) for value in class_ids),
                    "polygon_points": record_points,
                    "oob_points": record_oob_points,
                    "dhash_hex": f"{_dhash(gray_small):016x}",
                    "file_size_bytes": image_path.stat().st_size,
                }
            )
    return pd.DataFrame(image_rows), pd.DataFrame(instance_rows), decode_failures


def cross_split_near_duplicates(image_df: pd.DataFrame, threshold: int = 5) -> list[dict[str, Any]]:
    values = [
        (row.split, row.image_id, int(row.dhash_hex, 16))
        for row in image_df.itertuples(index=False)
    ]
    candidates: list[dict[str, Any]] = []
    for left_index, (left_split, left_id, left_hash) in enumerate(values):
        for right_split, right_id, right_hash in values[left_index + 1 :]:
            if left_split == right_split:
                continue
            distance = (left_hash ^ right_hash).bit_count()
            if distance <= threshold:
                candidates.append(
                    {
                        "left_split": left_split,
                        "left_image_id": left_id,
                        "right_split": right_split,
                        "right_image_id": right_id,
                        "dhash_hamming": distance,
                    }
                )
    return sorted(candidates, key=lambda item: item["dhash_hamming"])


def class_support_tables(
    records_by_split: dict[str, list[dict[str, Any]]],
    id_to_name: dict[int, str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split, records in records_by_split.items():
        instance_counts: Counter[int] = Counter()
        image_counts: Counter[int] = Counter()
        for record in records:
            labels = [int(instance["class_id"]) for instance in record["instances"]]
            instance_counts.update(labels)
            image_counts.update(set(labels))
        for class_id, class_name in sorted(id_to_name.items()):
            rows.append(
                {
                    "split": split,
                    "class_id": class_id,
                    "class_name": class_name,
                    "class_short": SHORT_NAMES.get(class_name, class_name),
                    "images": image_counts[class_id],
                    "instances": instance_counts[class_id],
                    "image_prevalence": image_counts[class_id] / max(1, len(records)),
                }
            )
    return pd.DataFrame(rows)


def sampler_profile(records: list[dict[str, Any]]) -> dict[str, Any]:
    class_counts = Counter(
        int(instance["class_id"])
        for record in records
        for instance in record["instances"]
    )
    power = 0.5
    raw_weights: list[float] = []
    for record in records:
        labels = {int(instance["class_id"]) for instance in record["instances"]}
        raw_weights.append(max(class_counts[label] ** (-power) for label in labels))
    median = sorted(raw_weights)[len(raw_weights) // 2]
    maximum = median * 5.0
    weights = np.asarray([min(weight, maximum) for weight in raw_weights], dtype=np.float64)
    probabilities = weights / weights.sum()
    draws = len(records)
    expected_unique = float(np.sum(1.0 - np.power(1.0 - probabilities, draws)))
    expected_instances: Counter[int] = Counter()
    expected_images: Counter[int] = Counter()
    for probability, record in zip(probabilities, records):
        labels = [int(instance["class_id"]) for instance in record["instances"]]
        for label in labels:
            expected_instances[label] += probability * draws
        for label in set(labels):
            expected_images[label] += probability * draws
    return {
        "images": len(records),
        "draws_per_epoch": draws,
        "expected_unique_images_per_epoch": expected_unique,
        "expected_duplicate_draws_per_epoch": draws - expected_unique,
        "minimum_image_draw_probability": float(probabilities.min()),
        "maximum_image_draw_probability": float(probabilities.max()),
        "max_to_min_probability_ratio": float(probabilities.max() / probabilities.min()),
        "expected_instances_per_epoch": {str(key): float(value) for key, value in sorted(expected_instances.items())},
        "expected_images_per_epoch": {str(key): float(value) for key, value in sorted(expected_images.items())},
    }


def split_shift_table(image_df: pd.DataFrame, instance_df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "brightness_mean",
        "contrast_std",
        "blur_laplacian_var",
        "entropy_bits",
        "edge_density",
        "megapixels",
        "instance_count",
        "annotation_edge_lift_median",
    ]
    rows: list[dict[str, Any]] = []
    train = image_df[image_df["split"] == "train2017"]
    val = image_df[image_df["split"] == "val2017"]
    for metric in metrics:
        left = train[metric].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
        right = val[metric].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
        pooled = math.sqrt((left.var(ddof=1) + right.var(ddof=1)) / 2.0) if len(left) > 1 and len(right) > 1 else 0.0
        standardized_difference = (right.mean() - left.mean()) / pooled if pooled > 0 else 0.0
        ks = ks_2samp(left, right, alternative="two-sided", method="auto")
        rows.append(
            {
                "metric": metric,
                "train_mean": float(left.mean()),
                "val_mean": float(right.mean()),
                "standardized_difference_val_minus_train": float(standardized_difference),
                "ks_statistic": float(ks.statistic),
                "ks_pvalue": float(ks.pvalue),
            }
        )

    train_class = instance_df[instance_df["split"] == "train2017"]["class_id"].value_counts().sort_index()
    val_class = instance_df[instance_df["split"] == "val2017"]["class_id"].value_counts().sort_index()
    classes = sorted(set(train_class.index) | set(val_class.index))
    train_distribution = np.asarray([train_class.get(class_id, 0) for class_id in classes], dtype=float)
    val_distribution = np.asarray([val_class.get(class_id, 0) for class_id in classes], dtype=float)
    train_distribution /= train_distribution.sum()
    val_distribution /= val_distribution.sum()
    rows.append(
        {
            "metric": "class_distribution_jensen_shannon",
            "train_mean": 0.0,
            "val_mean": float(jensenshannon(train_distribution, val_distribution, base=2.0)),
            "standardized_difference_val_minus_train": float("nan"),
            "ks_statistic": float("nan"),
            "ks_pvalue": float("nan"),
        }
    )
    return pd.DataFrame(rows)


def _build_eval_loader(records_path: Path, *, limit: int | None = None) -> DataLoader:
    dataset = VisualRedactionsDataset(
        records_path,
        short_side=640,
        max_long_side=1024,
        crop_size=None,
        crop_probability=0.0,
        horizontal_flip_probability=0.0,
        brightness_contrast_probability=0.0,
        training=False,
        limit=limit,
        subset_seed=1337,
    )
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
        collate_fn=detection_collate,
    )


def _load_checkpoint_model(device: torch.device) -> tuple[torch.nn.Module, Any]:
    config = load_training_config(CONFIG_PATH)
    model = build_instance_segmentation_model(
        copy.deepcopy(config.section("model")),
        num_classes=config.num_classes,
        min_size=int(config.section("data")["short_side"]),
        max_size=int(config.section("data")["max_long_side"]),
    )
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, config


def evaluate_final_checkpoint(output_dir: Path, force: bool = False) -> dict[str, Any]:
    result_path = output_dir / "model_train_evaluation.json"
    if result_path.is_file() and not force:
        _progress(f"[model] using cached full-train evaluation: {result_path}")
        return _load_json(result_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _progress(f"[model] loading final v3 checkpoint on {device}")
    model, config = _load_checkpoint_model(device)
    train_loader = _build_eval_loader(PROCESSED_ROOT / "records_train2017.jsonl")
    started = time.perf_counter()
    _progress("[model] evaluating all 501 training images; validation remains from the completed run")
    metrics = evaluate_instance_segmentation(
        model,
        train_loader,
        device,
        score_threshold=0.0,
        class_metrics=True,
        class_map=config.class_map,
    )
    payload = {
        "checkpoint": _safe_relative(CHECKPOINT_PATH),
        "device": str(device),
        "seconds": time.perf_counter() - started,
        "metrics": metrics,
    }
    _write_json(result_path, payload)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    _progress(f"[model] full-train evaluation complete in {payload['seconds']:.1f}s")
    return payload


def _read_training_curves() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    events = _load_jsonl(TRAINING_METRICS_PATH)
    train_rows = [event for event in events if event.get("event") == "train_step"]
    evaluation_rows: list[dict[str, Any]] = []
    epoch_rows: list[dict[str, Any]] = []
    current_epoch = 0
    for event in events:
        if event.get("event") == "evaluation":
            metrics = event["metrics"]
            evaluation_rows.append(
                {
                    "epoch": current_epoch + 1,
                    "global_step": event["global_step"],
                    "segm_map": metrics["segm_map"],
                    "segm_map_50": metrics["segm_map_50"],
                    "segm_mar_100": metrics["segm_mar_100"],
                    "bbox_map": metrics["bbox_map"],
                }
            )
        elif event.get("event") == "epoch_complete":
            current_epoch = int(event["epoch"])
            epoch_rows.append(
                {
                    "epoch": current_epoch + 1,
                    "global_step": event["global_step"],
                    "mean_loss": event["train"]["mean_loss"],
                    "seconds": event["train"]["seconds"],
                }
            )
    # Evaluation is logged before epoch_complete; infer epochs by order.
    for index, row in enumerate(evaluation_rows):
        row["epoch"] = index + 1
    return pd.DataFrame(train_rows), pd.DataFrame(evaluation_rows), pd.DataFrame(epoch_rows)


def _theme() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#fbfcfe",
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "font.family": "DejaVu Sans",
            "savefig.dpi": 180,
        }
    )


def _save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_download_geometry(alignment: dict[str, Any], output_dir: Path) -> Path:
    counts = alignment["status_counts"]
    labels = ["Aligned resize", "Rotation candidate", "Geometry mismatch", "Missing image"]
    keys = ["aligned_resize", "rotation_candidate", "geometry_mismatch", "missing_image"]
    values = [int(counts.get(key, 0)) for key in keys]
    colors = ["#2ca02c", "#ffbf00", "#d62728", "#6c757d"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1.15, 1]})
    axes[0].barh(labels[::-1], values[::-1], color=colors[::-1])
    axes[0].set_title("Source image/annotation geometry audit")
    axes[0].set_xlabel("Image records")
    for index, value in enumerate(values[::-1]):
        axes[0].text(value + max(values) * 0.012, index, f"{value:,}", va="center", fontsize=9)
    aligned = values[0]
    total = sum(values)
    axes[1].pie(
        [aligned, total - aligned],
        labels=["Eligible geometry", "Quarantined / missing"],
        autopct="%1.1f%%",
        startangle=90,
        colors=["#2ca02c", "#e8eaed"],
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    axes[1].set_title(f"Only {aligned:,} of {total:,} source pairs are resize-aligned")
    fig.suptitle("Download completeness is not the problem; source-pair geometry is", fontsize=15, fontweight="bold")
    path = output_dir / "01_download_and_geometry.png"
    _save_figure(fig, path)
    return path


def plot_class_support(support: pd.DataFrame, output_dir: Path) -> Path:
    data = support[support["split"].isin(AUDIT_SPLITS)].copy()
    order = (
        data[data["split"] == "train2017"].sort_values("instances", ascending=True)["class_short"].tolist()
    )
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.4), sharey=True)
    split_palette = {"train2017": "#1677ff", "val2017": "#ff7a45"}
    for axis, metric, title in zip(axes, ("instances", "images"), ("Labeled instances", "Images containing class")):
        sns.barplot(
            data=data,
            y="class_short",
            x=metric,
            hue="split",
            order=order,
            palette=split_palette,
            ax=axis,
        )
        axis.set_xscale("log")
        axis.set_title(title + " (log scale)")
        axis.set_xlabel("Count")
        axis.set_ylabel("")
        axis.legend(
            handles=[
                Patch(facecolor=split_palette["train2017"], label="Train"),
                Patch(facecolor=split_palette["val2017"], label="Validation"),
            ],
            title="Split",
        )
        axis.axvline(30, color="#d62728", linestyle="--", linewidth=1, alpha=0.8)
    fig.suptitle("The verified visual dataset is severely long-tailed", fontsize=15, fontweight="bold")
    fig.text(0.5, 0.01, "Dashed line: 30 samples - below this, class-level generalization estimates are especially unstable.", ha="center", fontsize=9)
    fig.subplots_adjust(top=0.88, bottom=0.10, wspace=0.10)
    path = output_dir / "02_class_support.png"
    _save_figure(fig, path)
    return path


def plot_object_sizes(instance_df: pd.DataFrame, output_dir: Path) -> Path:
    data = instance_df[instance_df["split"].isin(AUDIT_SPLITS)].copy()
    size_counts = (
        data.groupby(["class_short", "size_bucket"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["small", "medium", "large"], fill_value=0)
    )
    size_percent = size_counts.div(size_counts.sum(axis=1), axis=0) * 100.0
    size_percent = size_percent.sort_values("small", ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2), gridspec_kw={"width_ratios": [1.15, 1]})
    size_percent.plot(
        kind="barh",
        stacked=True,
        color=["#d62728", "#ffbf00", "#2ca02c"],
        ax=axes[0],
    )
    axes[0].set_title("COCO size bucket after 640/1024 resize")
    axes[0].set_xlabel("Instances (%)")
    axes[0].set_ylabel("")
    axes[0].legend(title="Object size", loc="lower right")
    sns.ecdfplot(
        data=data,
        x="model_area_pixels",
        hue="split",
        palette={"train2017": "#1677ff", "val2017": "#ff7a45"},
        ax=axes[1],
    )
    axes[1].set_xscale("log")
    axes[1].axvline(32**2, color="#d62728", linestyle="--", linewidth=1)
    axes[1].axvline(96**2, color="#ffbf00", linestyle="--", linewidth=1)
    axes[1].set_title("Cumulative object area distribution")
    axes[1].set_xlabel("Mask area at model input (pixels, log scale)")
    axes[1].set_ylabel("Cumulative fraction")
    fig.suptitle("Small regions are a core difficulty, especially for text-like privacy cues", fontsize=15, fontweight="bold")
    path = output_dir / "03_object_sizes.png"
    _save_figure(fig, path)
    return path


def plot_image_quality(image_df: pd.DataFrame, shift_df: pd.DataFrame, output_dir: Path) -> Path:
    data = image_df[image_df["split"].isin(AUDIT_SPLITS)].copy()
    data["log_blur"] = np.log10(data["blur_laplacian_var"].clip(lower=1e-3))
    metrics = [
        ("brightness_mean", "Mean brightness"),
        ("contrast_std", "Contrast (pixel SD)"),
        ("log_blur", "Sharpness, log10 Laplacian variance"),
        ("megapixels", "Decoded resolution (megapixels)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    for axis, (metric, label) in zip(axes.flat, metrics):
        sns.histplot(
            data=data,
            x=metric,
            hue="split",
            bins=32,
            stat="density",
            common_norm=False,
            element="step",
            fill=False,
            palette={"train2017": "#1677ff", "val2017": "#ff7a45"},
            ax=axis,
        )
        axis.set_title(label)
        axis.set_xlabel(label)
        axis.set_ylabel("Density")
    shift_lookup = shift_df.set_index("metric")
    subtitle = " | ".join(
        f"{name}: |SMD|={abs(_finite(shift_lookup.loc[name, 'standardized_difference_val_minus_train'])):.2f}"
        for name in ("brightness_mean", "contrast_std", "megapixels")
        if name in shift_lookup.index
    )
    fig.suptitle("Train-validation image-quality comparison", fontsize=15, fontweight="bold")
    fig.text(0.5, 0.01, subtitle + " (0.2 small, 0.5 moderate, 0.8 large shift)", ha="center", fontsize=9)
    fig.subplots_adjust(top=0.89, bottom=0.09, hspace=0.43, wspace=0.24)
    path = output_dir / "04_image_quality_shift.png"
    _save_figure(fig, path)
    return path


def plot_annotation_quality(image_df: pd.DataFrame, instance_df: pd.DataFrame, output_dir: Path) -> Path:
    images = image_df[image_df["split"].isin(AUDIT_SPLITS)].copy()
    instances = instance_df[instance_df["split"].isin(AUDIT_SPLITS)].copy()
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.2))
    sns.histplot(
        data=instances.replace([np.inf, -np.inf], np.nan).dropna(subset=["edge_lift"]),
        x="edge_lift",
        hue="split",
        bins=45,
        stat="density",
        common_norm=False,
        element="step",
        fill=False,
        palette={"train2017": "#1677ff", "val2017": "#ff7a45"},
        ax=axes[0, 0],
    )
    axes[0, 0].axvline(1.0, color="#d62728", linestyle="--", linewidth=1)
    axes[0, 0].set_xlim(0, min(8, float(instances["edge_lift"].quantile(0.99))))
    axes[0, 0].set_title("Boundary-to-image-edge support lift")
    axes[0, 0].set_xlabel("Lift over random image pixels (proxy, not ground truth)")

    sns.boxplot(
        data=instances,
        y="class_short",
        x="edge_lift",
        hue="split",
        showfliers=False,
        palette={"train2017": "#1677ff", "val2017": "#ff7a45"},
        ax=axes[0, 1],
    )
    axes[0, 1].set_xlim(0, min(8, float(instances["edge_lift"].quantile(0.95))))
    axes[0, 1].set_title("Edge-support proxy by privacy class")
    axes[0, 1].set_xlabel("Median boundary edge lift")
    axes[0, 1].set_ylabel("")

    flags = pd.DataFrame(
        {
            "Check": ["Dimension mismatch", "Out-of-bounds polygon points", "Masks touching image border", "Very tiny masks (<16x16 area)"],
            "Count": [
                int((~images["dimension_match"]).sum()),
                int((instances["oob_points"] > 0).sum()),
                int(instances["touches_border"].sum()),
                int((instances["model_area_pixels"] < 16**2).sum()),
            ],
        }
    )
    axes[1, 0].barh(flags["Check"][::-1], flags["Count"][::-1], color=["#8c8c8c", "#fa8c16", "#d46b08", "#cf1322"][::-1])
    axes[1, 0].set_title("Structural / difficulty flags")
    axes[1, 0].set_xlabel("Instances / images")
    for index, value in enumerate(flags["Count"][::-1]):
        axes[1, 0].text(value + max(1, flags["Count"].max()) * 0.02, index, f"{value:,}", va="center", fontsize=9)

    sns.scatterplot(
        data=images,
        x="instance_count",
        y="annotation_edge_lift_median",
        hue="split",
        alpha=0.65,
        s=26,
        palette={"train2017": "#1677ff", "val2017": "#ff7a45"},
        ax=axes[1, 1],
    )
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_title("Crowding vs annotation-edge evidence")
    axes[1, 1].set_xlabel("Instances in image")
    axes[1, 1].set_ylabel("Median edge lift (log scale)")
    fig.suptitle("Accepted geometry is structurally valid, but visual alignment still needs review", fontsize=15, fontweight="bold")
    fig.subplots_adjust(top=0.89, bottom=0.08, hspace=0.46, wspace=0.32)
    path = output_dir / "05_annotation_quality.png"
    _save_figure(fig, path)
    return path


def plot_cooccurrence(records_by_split: dict[str, list[dict[str, Any]]], id_to_name: dict[int, str], output_dir: Path) -> Path:
    class_ids = sorted(id_to_name)
    matrix = np.zeros((len(class_ids), len(class_ids)), dtype=int)
    for split in AUDIT_SPLITS:
        for record in records_by_split[split]:
            labels = sorted({int(instance["class_id"]) for instance in record["instances"]})
            for left in labels:
                for right in labels:
                    matrix[class_ids.index(left), class_ids.index(right)] += 1
    names = [SHORT_NAMES.get(id_to_name[class_id], id_to_name[class_id]) for class_id in class_ids]
    fig, axis = plt.subplots(figsize=(9.5, 8.2))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=names,
        yticklabels=names,
        cbar_kws={"label": "Images"},
        ax=axis,
    )
    axis.set_title("Class co-occurrence in train + validation images")
    axis.set_xlabel("")
    axis.set_ylabel("")
    plt.xticks(rotation=35, ha="right")
    plt.yticks(rotation=0)
    path = output_dir / "06_class_cooccurrence.png"
    _save_figure(fig, path)
    return path


def plot_training_curves(output_dir: Path) -> Path:
    train_df, eval_df, epoch_df = _read_training_curves()
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5))
    train_df = train_df.sort_values("global_step")
    train_df["loss_rolling"] = train_df["loss"].rolling(15, min_periods=1, center=True).median()
    axes[0, 0].plot(train_df["global_step"], train_df["loss"], color="#9ec5fe", linewidth=0.7, alpha=0.7)
    axes[0, 0].plot(train_df["global_step"], train_df["loss_rolling"], color="#1677ff", linewidth=2)
    axes[0, 0].set_title("Logged training loss")
    axes[0, 0].set_xlabel("Optimizer step")
    axes[0, 0].set_ylabel("Loss")

    component_names = ["loss_classifier", "loss_box_reg", "loss_mask", "loss_objectness", "loss_rpn_box_reg"]
    for name in component_names:
        if name in train_df:
            axes[0, 1].plot(train_df["global_step"], train_df[name].rolling(15, min_periods=1, center=True).median(), label=name.replace("loss_", ""))
    axes[0, 1].set_title("Rolling median loss components")
    axes[0, 1].set_xlabel("Optimizer step")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].legend(fontsize=8, ncol=2)

    for metric, color in (("segm_map", "#d62728"), ("segm_map_50", "#ff7a45"), ("bbox_map", "#2ca02c")):
        axes[1, 0].plot(eval_df["epoch"], eval_df[metric].clip(lower=1e-7), marker="o", markersize=3, label=metric, color=color)
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_title("Validation metrics remain near zero")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("COCO metric (log scale)")
    axes[1, 0].legend()

    axes[1, 1].plot(epoch_df["epoch"], epoch_df["mean_loss"], marker="o", color="#722ed1")
    axes[1, 1].set_title("Epoch-average training loss")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Mean loss")
    axes[1, 1].axvline(18, color="#8c8c8c", linestyle="--", linewidth=1, label="LR decay")
    axes[1, 1].legend()
    fig.suptitle("Optimization loss improved, but validation localization did not", fontsize=15, fontweight="bold")
    fig.subplots_adjust(top=0.88, bottom=0.08, hspace=0.46, wspace=0.22)
    path = output_dir / "07_training_curves.png"
    _save_figure(fig, path)
    return path


def plot_model_performance(train_metrics: dict[str, Any], val_metrics: dict[str, Any], output_dir: Path) -> Path:
    global_rows = []
    for split, metrics in (("Train", train_metrics), ("Validation", val_metrics)):
        for key, label in (("segm_map", "Mask AP"), ("segm_map_50", "Mask AP50"), ("bbox_map", "Box AP"), ("segm_mar_100", "Mask AR100")):
            global_rows.append({"Split": split, "Metric": label, "Value": max(1e-7, float(metrics[key]))})
    global_df = pd.DataFrame(global_rows)
    per_class_rows = []
    names = sorted(set(train_metrics.get("per_class", {})) | set(val_metrics.get("per_class", {})))
    for name in names:
        for split, metrics in (("Train", train_metrics), ("Validation", val_metrics)):
            value = metrics.get("per_class", {}).get(name, {}).get("map", -1)
            per_class_rows.append(
                {
                    "Class": SHORT_NAMES.get(name, name),
                    "Split": split,
                    "Mask AP": np.nan if value is None or float(value) < 0 else max(1e-7, float(value)),
                }
            )
    per_class_df = pd.DataFrame(per_class_rows)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2), gridspec_kw={"width_ratios": [0.9, 1.45]})
    sns.barplot(data=global_df, x="Metric", y="Value", hue="Split", palette={"Train": "#1677ff", "Validation": "#ff4d4f"}, ax=axes[0])
    axes[0].set_yscale("log")
    axes[0].set_title("Final checkpoint: train vs validation")
    axes[0].set_ylabel("COCO score (log scale)")
    axes[0].set_xlabel("")
    axes[0].tick_params(axis="x", rotation=25)
    sns.barplot(data=per_class_df, y="Class", x="Mask AP", hue="Split", palette={"Train": "#1677ff", "Validation": "#ff4d4f"}, ax=axes[1])
    axes[1].set_xscale("log")
    axes[1].set_title("Per-class mask AP")
    axes[1].set_xlabel("COCO mask AP (log scale; floor=1e-7)")
    axes[1].set_ylabel("")
    fig.suptitle("The train/validation gap identifies underfit versus generalization failure", fontsize=15, fontweight="bold")
    path = output_dir / "08_model_train_vs_validation.png"
    _save_figure(fig, path)
    return path


def _draw_record_overlay(record: dict[str, Any], id_to_name: dict[int, str], max_side: int = 900) -> np.ndarray:
    image_path = project_path(record["image_path"])
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return np.zeros((360, 540, 3), dtype=np.uint8)
    height, width = image.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale < 1.0:
        image = cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    overlay = image.copy()
    for instance in record["instances"]:
        class_id = int(instance["class_id"])
        color = PALETTE[class_id]
        for polygon in instance["polygons"]:
            points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2) * scale
            points_i = np.round(points).astype(np.int32)
            if class_id == 4:
                # Avoid reproducing explicit private regions in a report preview.
                cv2.fillPoly(overlay, [points_i], (30, 30, 30))
            else:
                cv2.fillPoly(overlay, [points_i], color)
            cv2.polylines(image, [points_i], True, color, 2, cv2.LINE_AA)
    image = cv2.addWeighted(overlay, 0.30, image, 0.70, 0)
    return image


def _grid_image(panels: list[tuple[np.ndarray, str]], columns: int, panel_width: int = 430, panel_height: int = 300) -> np.ndarray:
    rows = math.ceil(len(panels) / columns)
    canvas = np.full((rows * (panel_height + 38), columns * panel_width, 3), 248, dtype=np.uint8)
    for index, (panel, title) in enumerate(panels):
        row, column = divmod(index, columns)
        h, w = panel.shape[:2]
        scale = min(panel_width / max(1, w), panel_height / max(1, h))
        resized = cv2.resize(panel, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
        y0 = row * (panel_height + 38) + (panel_height - resized.shape[0]) // 2
        x0 = column * panel_width + (panel_width - resized.shape[1]) // 2
        canvas[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized
        text_y = row * (panel_height + 38) + panel_height + 24
        cv2.putText(canvas, title[:63], (column * panel_width + 8, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (25, 25, 25), 1, cv2.LINE_AA)
    return canvas


def create_ground_truth_montages(
    records_by_split: dict[str, list[dict[str, Any]]],
    image_df: pd.DataFrame,
    instance_df: pd.DataFrame,
    id_to_name: dict[int, str],
    output_dir: Path,
) -> tuple[Path, Path, list[str]]:
    record_lookup = {
        (split, str(record["image_id"])): record
        for split, records in records_by_split.items()
        for record in records
    }
    representative_panels: list[tuple[np.ndarray, str]] = []
    selected_ids: list[str] = []
    manual_path = output_dir / "manual_visual_review.csv"
    if manual_path.is_file():
        # Lock the montage to the exact samples whose human judgments are
        # recorded in the report.  This avoids selection changes caused by CSV
        # floating-point round trips when cached profiles are reused.
        manual = pd.read_csv(manual_path, dtype={"image_id": str})
        for row in manual.itertuples(index=False):
            split = str(row.split)
            image_id = str(row.image_id)
            record = record_lookup[(split, image_id)]
            panel = _draw_record_overlay(record, id_to_name)
            label = f"{'Train' if split == 'train2017' else 'Val'} | {row.focus_class} | {image_id}"
            representative_panels.append((panel, label))
            selected_ids.append(f"{split}:{image_id}")
    else:
        for class_id, class_name in sorted(id_to_name.items()):
            for split in AUDIT_SPLITS:
                candidates = instance_df[(instance_df["split"] == split) & (instance_df["class_id"] == class_id)]
                if candidates.empty:
                    continue
                median = candidates["edge_lift"].replace([np.inf, -np.inf], np.nan).median()
                candidates = candidates.assign(distance=(candidates["edge_lift"] - median).abs()).sort_values(["distance", "model_area_pixels"], ascending=[True, False])
                image_id = str(candidates.iloc[0]["image_id"])
                record = record_lookup[(split, image_id)]
                panel = _draw_record_overlay(record, id_to_name)
                label = f"{'Train' if split == 'train2017' else 'Val'} | {SHORT_NAMES.get(class_name, class_name)} | {image_id}"
                representative_panels.append((panel, label))
                selected_ids.append(f"{split}:{image_id}")
    representative = _grid_image(representative_panels, columns=3)
    representative_path = output_dir / "09_representative_ground_truth_grid.jpg"
    cv2.imwrite(str(representative_path), representative, [cv2.IMWRITE_JPEG_QUALITY, 91])

    suspicious_images = (
        image_df[image_df["split"].isin(AUDIT_SPLITS)]
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["annotation_edge_lift_median"])
        .sort_values(["annotation_edge_lift_median", "instance_count"], ascending=[True, False])
    )
    suspicious_panels: list[tuple[np.ndarray, str]] = []
    used_classes: Counter[int] = Counter()
    for row in suspicious_images.itertuples(index=False):
        record = record_lookup[(row.split, str(row.image_id))]
        labels = sorted({int(instance["class_id"]) for instance in record["instances"]})
        if all(used_classes[label] >= 3 for label in labels) and len(suspicious_panels) < 10:
            continue
        panel = _draw_record_overlay(record, id_to_name)
        class_label = "+".join(SHORT_NAMES.get(id_to_name[label], id_to_name[label]) for label in labels[:2])
        title = f"{row.split.replace('2017','')} | lift={row.annotation_edge_lift_median:.2f} | {class_label} | {row.image_id}"
        suspicious_panels.append((panel, title))
        used_classes.update(labels)
        if len(suspicious_panels) >= 12:
            break
    suspicious = _grid_image(suspicious_panels, columns=3)
    suspicious_path = output_dir / "10_low_edge_support_review_grid.jpg"
    cv2.imwrite(str(suspicious_path), suspicious, [cv2.IMWRITE_JPEG_QUALITY, 91])
    return representative_path, suspicious_path, selected_ids


def _select_prediction_records(
    records_by_split: dict[str, list[dict[str, Any]]],
    id_to_name: dict[int, str],
) -> list[tuple[str, dict[str, Any]]]:
    selected: list[tuple[str, dict[str, Any]]] = []
    target_classes = [1, 3, 2, 5, 8, 9]
    for split in AUDIT_SPLITS:
        used_ids: set[str] = set()
        for class_id in target_classes:
            candidates = [
                record
                for record in records_by_split[split]
                if str(record["image_id"]) not in used_ids
                and any(int(instance["class_id"]) == class_id for instance in record["instances"])
            ]
            if not candidates:
                continue
            candidates.sort(key=lambda record: (len(record["instances"]), str(record["image_id"])))
            record = candidates[len(candidates) // 2]
            selected.append((split, record))
            used_ids.add(str(record["image_id"]))
            if len([item for item in selected if item[0] == split]) >= 5:
                break
    return selected


@torch.inference_mode()
def create_prediction_montage(
    records_by_split: dict[str, list[dict[str, Any]]],
    id_to_name: dict[int, str],
    output_dir: Path,
    force: bool = False,
) -> tuple[Path, dict[str, Any]]:
    path = output_dir / "11_final_model_predictions_grid.jpg"
    summary_path = output_dir / "prediction_sample_summary.json"
    if path.is_file() and summary_path.is_file() and not force:
        return path, _load_json(summary_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _progress(f"[model] rendering qualitative predictions on {device}")
    model, _ = _load_checkpoint_model(device)
    selected = _select_prediction_records(records_by_split, id_to_name)
    panels: list[tuple[np.ndarray, str]] = []
    rows: list[dict[str, Any]] = []
    for split, record in selected:
        dataset = VisualRedactionsDataset(
            PROCESSED_ROOT / f"records_{split}.jsonl",
            short_side=640,
            max_long_side=1024,
            training=False,
        )
        index = next(index for index, item in enumerate(dataset.records) if str(item["image_id"]) == str(record["image_id"]))
        image_tensor, target = dataset[index]
        prediction = model([image_tensor.to(device)])[0]
        image = (image_tensor.mul(255).byte().permute(1, 2, 0).cpu().numpy())[:, :, ::-1].copy()
        overlay = image.copy()
        # Ground truth contours in green.
        for mask in target["masks"].cpu().numpy():
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(image, contours, -1, (0, 230, 0), 2, cv2.LINE_AA)
        scores = prediction["scores"].detach().cpu().numpy()
        keep = np.where(scores >= 0.10)[0][:8]
        if len(keep) == 0:
            keep = np.arange(min(3, len(scores)))
        predicted_labels: list[str] = []
        for prediction_index in keep:
            class_id = int(prediction["labels"][prediction_index].item())
            score = float(prediction["scores"][prediction_index].item())
            mask = prediction["masks"][prediction_index, 0].detach().cpu().numpy() >= 0.5
            color = PALETTE.get(class_id, (0, 0, 255))
            overlay[mask] = color
            box = prediction["boxes"][prediction_index].detach().cpu().numpy().astype(int)
            cv2.rectangle(image, tuple(box[:2]), tuple(box[2:]), color, 1, cv2.LINE_AA)
            predicted_labels.append(f"{SHORT_NAMES.get(id_to_name.get(class_id, str(class_id)), str(class_id))}:{score:.2f}")
        image = cv2.addWeighted(overlay, 0.25, image, 0.75, 0)
        gt_names = sorted({SHORT_NAMES.get(id_to_name[int(instance["class_id"])], id_to_name[int(instance["class_id"])]) for instance in record["instances"]})
        title = f"{split.replace('2017','')} | {record['image_id']} | top={scores[0]:.2f}" if len(scores) else f"{split} | {record['image_id']} | no detections"
        panels.append((image, title))
        rows.append(
            {
                "split": split,
                "image_id": str(record["image_id"]),
                "ground_truth_classes": gt_names,
                "top_score": float(scores[0]) if len(scores) else 0.0,
                "predictions_at_0_10": int(np.sum(scores >= 0.10)),
                "top_predictions": predicted_labels,
            }
        )
    grid = _grid_image(panels, columns=2, panel_width=650, panel_height=420)
    cv2.imwrite(str(path), grid, [cv2.IMWRITE_JPEG_QUALITY, 91])
    payload = {"legend": "green contours=ground truth; translucent colored regions/boxes=predictions", "samples": rows}
    _write_json(summary_path, payload)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return path, payload


def build_summary(
    records_by_split: dict[str, list[dict[str, Any]]],
    support_df: pd.DataFrame,
    image_df: pd.DataFrame,
    instance_df: pd.DataFrame,
    shift_df: pd.DataFrame,
    alignment: dict[str, Any],
    split_audit: dict[str, Any],
    near_duplicates: list[dict[str, Any]],
    sampler: dict[str, Any],
    train_evaluation: dict[str, Any],
    prediction_summary: dict[str, Any],
) -> dict[str, Any]:
    training_result = _load_json(TRAINING_RESULT_PATH)
    overfit_result = _load_json(OVERFIT_RESULT_PATH)
    val_metrics = training_result["last_evaluation"]
    train_metrics = train_evaluation["metrics"]
    train_support = support_df[support_df["split"] == "train2017"].copy()
    rare = train_support[(train_support["images"] < 30) | (train_support["instances"] < 50)]
    source_total = int(alignment["manifest_rows"])
    aligned = int(alignment["status_counts"]["aligned_resize"])
    oob_instances = int((instance_df[instance_df["split"].isin(AUDIT_SPLITS)]["oob_points"] > 0).sum())
    dimension_mismatches = int((~image_df["dimension_match"]).sum())
    tiny_instances = int((instance_df[instance_df["split"].isin(AUDIT_SPLITS)]["model_area_pixels"] < 16**2).sum())
    all_audit_instances = instance_df[instance_df["split"].isin(AUDIT_SPLITS)]
    edge_median = float(all_audit_instances["edge_lift"].replace([np.inf, -np.inf], np.nan).median())
    train_map = float(train_metrics["segm_map"])
    val_map = float(val_metrics["segm_map"])
    generalization_ratio = val_map / train_map if train_map > 0 else 0.0
    manual_review_path = DEFAULT_OUTPUT / "manual_visual_review.csv"
    manual_review = pd.read_csv(manual_review_path) if manual_review_path.is_file() else pd.DataFrame()
    manual_counts = (
        {str(key): int(value) for key, value in manual_review["review_result"].value_counts().items()}
        if not manual_review.empty
        else {}
    )
    if train_map < 0.02:
        checkpoint_diagnosis = "severe full-training-set underfit or inconsistent/noisy supervision"
    elif generalization_ratio < 0.10:
        checkpoint_diagnosis = "severe generalization collapse after partial training-set fit"
    else:
        checkpoint_diagnosis = "model learns some transferable signal, but performance remains inadequate"

    verdict = {
        "primary_blocker": "invalid cross-dataset identity join: Visual Redactions masks were paired with different VISPR pixels",
        "secondary_blocker": "after repairing pixels, rare-class scarcity and a single-model modality mismatch remain",
        "do_not_do": "Do not start another Mask R-CNN, Mask2Former, or any other model on the current records.",
        "recommended_next_action": "Download the separate official Visual Redactions image archives, require pixel-identity evidence, rebuild records, and manually approve a stratified benchmark before retraining.",
        "checkpoint_diagnosis": checkpoint_diagnosis,
    }
    return {
        "generated_at_local": time.strftime("%Y-%m-%d %H:%M:%S"),
        "scope": {
            "source_manifest_records": source_total,
            "selected_records_scanned": int(len(image_df)),
            "selected_instances_scanned": int(len(instance_df)),
            "model_evaluation_splits": ["train2017", "val2017"],
            "test_protocol": "Test split structurally scanned but not used for model evaluation or qualitative prediction selection.",
        },
        "download_integrity": {
            "vispr_splits_complete": True,
            "visual_redactions_annotations_complete": True,
            "visual_redactions_correct_image_archives_downloaded": False,
            "visual_redactions_required_image_archives": {
                "train2017": {
                    "url": "https://datasets.d2.mpi-inf.mpg.de/orekondy18cvpr/v1/images/train2017.tar.gz",
                    "bytes": 7816171895,
                },
                "val2017": {
                    "url": "https://datasets.d2.mpi-inf.mpg.de/orekondy18cvpr/v1/images/val2017.tar.gz",
                    "bytes": 3158549744,
                },
                "test2017": {
                    "url": "https://datasets.d2.mpi-inf.mpg.de/orekondy18cvpr/v1/images/test2017.tar.gz",
                    "bytes": 5633037591,
                },
            },
            "visual_redactions_required_total_bytes": 16607759230,
            "current_training_pixel_source": "VISPR 2017 archives - invalid for the Visual Redactions masks despite similar IDs",
            "vpd_public_repository_complete": True,
            "vpd_annotation_files_found": 0,
            "vpd_note": "The local public VPD repository contains 2,462 videos but no verified 100K-image box/mask annotation package, so it is not training data for this run.",
        },
        "geometry": {
            "aligned_resize": aligned,
            "rotation_candidate": int(alignment["status_counts"]["rotation_candidate"]),
            "geometry_mismatch": int(alignment["status_counts"]["geometry_mismatch"]),
            "missing_image": int(alignment["status_counts"]["missing_image"]),
            "eligible_fraction": aligned / source_total,
        },
        "processed_data": {
            "images_by_split": {split: len(records_by_split[split]) for split in SPLITS},
            "instances_by_split": {
                split: int((instance_df["split"] == split).sum()) for split in SPLITS
            },
            "dimension_mismatches": dimension_mismatches,
            "instances_with_out_of_bounds_points": oob_instances,
            "tiny_instances_under_16x16_area_train_val": tiny_instances,
            "median_boundary_edge_lift_train_val": edge_median,
            "manual_stratified_review": {
                "images_reviewed": int(len(manual_review)),
                "result_counts": manual_counts,
                "minimum_clearly_mismatched_fraction": (
                    manual_counts.get("clearly_mismatched", 0) / len(manual_review)
                    if len(manual_review)
                    else None
                ),
                "method": "One median edge-lift example per class and split; visual judgment recorded in manual_visual_review.csv.",
            },
            "rare_train_classes": [
                {
                    "class": row.class_short,
                    "images": int(row.images),
                    "instances": int(row.instances),
                }
                for row in rare.itertuples(index=False)
            ],
        },
        "leakage": {
            "official_full_audit_passed": bool(split_audit.get("passed")),
            "official_near_duplicate_candidates": len(split_audit.get("near_cross_split_candidates", [])),
            "selected_dhash_candidates_at_hamming_5": len(near_duplicates),
        },
        "sampler": sampler,
        "distribution_shift": shift_df.to_dict(orient="records"),
        "model": {
            "checkpoint": _safe_relative(CHECKPOINT_PATH),
            "epochs": int(training_result["completed_epoch"]) + 1,
            "global_steps": int(training_result["global_step"]),
            "full_train_mask_map": train_map,
            "full_train_mask_map_50": float(train_metrics["segm_map_50"]),
            "validation_mask_map": val_map,
            "validation_mask_map_50": float(val_metrics["segm_map_50"]),
            "validation_to_train_map_ratio": generalization_ratio,
            "eight_image_overfit_best_mask_map": float(overfit_result["best_map"]),
            "eight_image_overfit_final_mask_map_50": float(overfit_result["last_evaluation"]["segm_map_50"]),
            "train_evaluation_seconds": float(train_evaluation["seconds"]),
            "train_metrics": train_metrics,
            "validation_metrics": val_metrics,
            "prediction_samples": prediction_summary,
        },
        "verdict": verdict,
    }


def run_audit(output_dir: str | Path = DEFAULT_OUTPUT, *, force_model_eval: bool = False) -> dict[str, Any]:
    output_dir = Path(output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    _theme()
    _progress(f"ConsentGuard full audit output: {output_dir}")
    records_by_split, class_map, id_to_name = load_records()
    alignment = _load_json(ALIGNMENT_AUDIT_PATH)
    split_audit = _load_json(SPLIT_AUDIT_PATH)

    image_csv = output_dir / "image_metrics.csv"
    instance_csv = output_dir / "instance_metrics.csv"
    failures_json = output_dir / "decode_failures.json"
    if image_csv.is_file() and instance_csv.is_file() and failures_json.is_file() and not force_model_eval:
        _progress("[data] using cached per-image and per-instance profiles")
        image_df = pd.read_csv(image_csv, dtype={"image_id": str, "dhash_hex": str})
        instance_df = pd.read_csv(instance_csv, dtype={"image_id": str})
        decode_failures = _load_json(failures_json)
    else:
        image_df, instance_df, decode_failures = profile_records(records_by_split, id_to_name)
        image_df.to_csv(image_csv, index=False)
        instance_df.to_csv(instance_csv, index=False)
        _write_json(failures_json, decode_failures)

    support_df = class_support_tables(records_by_split, id_to_name)
    support_df.to_csv(output_dir / "class_support.csv", index=False)
    shift_df = split_shift_table(image_df, instance_df)
    shift_df.to_csv(output_dir / "train_validation_shift.csv", index=False)
    near_duplicates = cross_split_near_duplicates(image_df, threshold=5)
    _write_json(output_dir / "selected_near_duplicate_candidates.json", near_duplicates)
    sampler = sampler_profile(records_by_split["train2017"])
    _write_json(output_dir / "sampler_profile.json", sampler)

    figures: list[Path] = []
    figures.append(plot_download_geometry(alignment, output_dir))
    figures.append(plot_class_support(support_df, output_dir))
    figures.append(plot_object_sizes(instance_df, output_dir))
    figures.append(plot_image_quality(image_df, shift_df, output_dir))
    figures.append(plot_annotation_quality(image_df, instance_df, output_dir))
    figures.append(plot_cooccurrence(records_by_split, id_to_name, output_dir))
    figures.append(plot_training_curves(output_dir))

    train_evaluation = evaluate_final_checkpoint(output_dir, force=force_model_eval)
    training_result = _load_json(TRAINING_RESULT_PATH)
    figures.append(plot_model_performance(train_evaluation["metrics"], training_result["last_evaluation"], output_dir))
    representative_path, suspicious_path, representative_ids = create_ground_truth_montages(
        records_by_split, image_df, instance_df, id_to_name, output_dir
    )
    figures.extend([representative_path, suspicious_path])
    prediction_path, prediction_summary = create_prediction_montage(
        records_by_split, id_to_name, output_dir, force=force_model_eval
    )
    figures.append(prediction_path)

    summary = build_summary(
        records_by_split,
        support_df,
        image_df,
        instance_df,
        shift_df,
        alignment,
        split_audit,
        near_duplicates,
        sampler,
        train_evaluation,
        prediction_summary,
    )
    summary["artifacts"] = {
        "figures": [_safe_relative(path) for path in figures],
        "tables": [
            _safe_relative(image_csv),
            _safe_relative(instance_csv),
            _safe_relative(output_dir / "class_support.csv"),
            _safe_relative(output_dir / "train_validation_shift.csv"),
        ],
        "representative_image_ids": representative_ids,
    }
    _write_json(output_dir / "audit_summary.json", summary)
    _progress(json.dumps({"verdict": summary["verdict"], "model": {key: value for key, value in summary["model"].items() if key in {"full_train_mask_map", "validation_mask_map", "eight_image_overfit_best_mask_map"}}}, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force-model-eval", action="store_true")
    args = parser.parse_args()
    run_audit(args.output_dir, force_model_eval=args.force_model_eval)


if __name__ == "__main__":
    main()
