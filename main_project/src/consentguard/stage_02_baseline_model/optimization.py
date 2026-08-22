"""Optimizer and learning-rate scheduler factories for Stage 02."""

from __future__ import annotations

from typing import Any

import torch


def build_optimizer(model: torch.nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    name = str(config["name"]).lower()
    if name == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=float(config["learning_rate"]),
            momentum=float(config.get("momentum", 0.9)),
            weight_decay=float(config.get("weight_decay", 0.0)),
        )
    if name == "adamw":
        return torch.optim.AdamW(
            parameters,
            lr=float(config["learning_rate"]),
            weight_decay=float(config.get("weight_decay", 0.0)),
        )
    raise ValueError(f"Unsupported optimizer: {name}")


def build_scheduler(optimizer: torch.optim.Optimizer, config: dict[str, Any]):
    name = str(config["name"]).lower()
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=int(config["step_size"]), gamma=float(config["gamma"])
        )
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(config.get("t_max", 20)),
            eta_min=float(config.get("eta_min", 0.0)),
        )
    if name == "none":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    raise ValueError(f"Unsupported scheduler: {name}")
