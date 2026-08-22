"""Build deterministic datasets, samplers, and loaders for Stage 02 training."""

from __future__ import annotations

from collections import Counter
from typing import Any

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from consentguard.shared.runtime import seed_worker, seeded_generator
from consentguard.stage_01_data.dataset import VisualRedactionsDataset, detection_collate
from consentguard.stage_02_baseline_model.config import TrainingConfig


def build_datasets(
    config: TrainingConfig,
) -> tuple[VisualRedactionsDataset, VisualRedactionsDataset | None]:
    data = config.section("data")
    seed = int(config.section("experiment")["seed"])
    shared = {
        "short_side": data["short_side"],
        "max_long_side": data["max_long_side"],
        "crop_size": data["crop_size"],
        "crop_probability": float(data["crop_probability"]),
        "crop_context_factor": float(data["crop_context_factor"]),
        "min_crop_visibility": float(data["min_crop_visibility"]),
        "horizontal_flip_probability": float(data["horizontal_flip_probability"]),
        "brightness_contrast_probability": float(data["brightness_contrast_probability"]),
        "subset_seed": seed,
    }
    train_dataset = VisualRedactionsDataset(
        data["train_records"], training=True, limit=data["max_train_samples"], **shared
    )
    val_dataset = None
    if bool(config.section("evaluation")["enabled"]):
        val_dataset = VisualRedactionsDataset(
            data["val_records"], training=False, limit=data["max_val_samples"], **shared
        )
    return train_dataset, val_dataset


def _class_balanced_sampler(
    dataset: VisualRedactionsDataset,
    data_config: dict[str, Any],
    seed: int,
) -> WeightedRandomSampler:
    class_counts = Counter(
        int(instance["class_id"])
        for record in dataset.records
        for instance in record["instances"]
    )
    power = float(data_config["class_balance_power"])
    raw_weights: list[float | None] = []
    for record in dataset.records:
        labels = {int(instance["class_id"]) for instance in record["instances"]}
        raw_weights.append(
            max(class_counts[label] ** (-power) for label in labels) if labels else None
        )
    positive_weights = sorted(weight for weight in raw_weights if weight is not None)
    if not positive_weights:
        raise RuntimeError("Class-balanced sampling requires at least one positive instance")
    median = positive_weights[len(positive_weights) // 2]
    negative_weight = median * float(data_config["negative_sampling_weight"])
    maximum = median * float(data_config["max_class_balance_ratio"])
    weights = torch.tensor(
        [min(negative_weight if weight is None else weight, maximum) for weight in raw_weights],
        dtype=torch.double,
    )
    weights /= weights.mean()
    return WeightedRandomSampler(
        weights,
        num_samples=len(dataset),
        replacement=True,
        generator=seeded_generator(seed),
    )


def build_data_loaders(config: TrainingConfig) -> tuple[DataLoader, DataLoader | None]:
    train_dataset, val_dataset = build_datasets(config)
    data = config.section("data")
    training = config.section("training")
    seed = int(config.section("experiment")["seed"])
    workers = int(data["num_workers"])
    loader_options: dict[str, Any] = {
        "num_workers": workers,
        "collate_fn": detection_collate,
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": seed_worker,
        "persistent_workers": workers > 0,
    }
    if workers > 0:
        loader_options["prefetch_factor"] = 2
    sampler = (
        _class_balanced_sampler(train_dataset, data, seed)
        if data["sampling"] == "class_balanced"
        else None
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(training["batch_size"]),
        shuffle=sampler is None,
        sampler=sampler,
        generator=seeded_generator(seed),
        **loader_options,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            generator=seeded_generator(seed + 1),
            **loader_options,
        )
    return train_loader, val_loader
