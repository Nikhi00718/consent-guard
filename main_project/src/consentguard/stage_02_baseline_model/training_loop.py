"""Reliable single-GPU training loop for the Visual Redactions localizer."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from consentguard.shared.paths import project_path
from consentguard.shared.runtime import (
    atomic_json_dump,
    atomic_link_or_copy,
    atomic_torch_save,
    environment_snapshot,
)
from consentguard.stage_01_data.dataset import target_for_model
from consentguard.stage_02_baseline_model.config import TrainingConfig
from consentguard.stage_02_baseline_model.metrics import evaluate_instance_segmentation
from consentguard.stage_02_baseline_model.optimization import build_optimizer, build_scheduler


class MaskRCNNTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        config: TrainingConfig,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        self.training_config = config.section("training")
        self.evaluation_config = config.section("evaluation")
        self.optimizer = build_optimizer(model, config.section("optimizer"))
        self.base_learning_rates = [group["lr"] for group in self.optimizer.param_groups]
        self.scheduler = build_scheduler(self.optimizer, config.section("scheduler"))
        self.amp_enabled = bool(self.training_config["amp"]) and device.type == "cuda"
        self.scaler = torch.amp.GradScaler(device.type, enabled=self.amp_enabled)
        self.output_dir = config.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as error:
            raise RuntimeError("TensorBoard is missing; install requirements/base.txt") from error
        self.writer = SummaryWriter(log_dir=str(self.output_dir / "tensorboard"))
        self.metrics_path = self.output_dir / "metrics.jsonl"
        self.start_epoch = 0
        self.global_step = 0
        self.best_map = -math.inf
        self.last_evaluation: dict[str, Any] | None = None
        atomic_json_dump(config.as_dict(), self.output_dir / "resolved_config.json")
        atomic_json_dump(environment_snapshot(), self.output_dir / "environment.json")

    def _log(self, payload: dict[str, Any]) -> None:
        record = {"time_unix": time.time(), **payload}
        with self.metrics_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def checkpoint_payload(self, epoch: int) -> dict[str, Any]:
        loader_generator = getattr(self.train_loader, "generator", None)
        sampler_generator = getattr(getattr(self.train_loader, "sampler", None), "generator", None)
        return {
            "schema_version": 1,
            "epoch": epoch,
            "global_step": self.global_step,
            "best_map": self.best_map,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "scaler_state": self.scaler.state_dict(),
            "config": self.config.as_dict(),
            "class_map": self.config.class_map,
            "environment": environment_snapshot(),
            "last_evaluation": self.last_evaluation,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "train_loader_generator_state": (
                loader_generator.get_state() if loader_generator is not None else None
            ),
            "train_sampler_generator_state": (
                sampler_generator.get_state() if sampler_generator is not None else None
            ),
        }

    def save_checkpoint(self, epoch: int, *, is_best: bool = False) -> Path:
        payload = self.checkpoint_payload(epoch)
        last_path = self.output_dir / "last.pt"
        atomic_torch_save(payload, last_path)
        if (epoch + 1) % int(self.training_config["save_every_epochs"]) == 0:
            atomic_link_or_copy(last_path, self.output_dir / f"epoch-{epoch + 1:03d}.pt")
        if is_best:
            atomic_link_or_copy(last_path, self.output_dir / "best.pt")
        return last_path

    def resume(self, checkpoint_path: str | Path) -> None:
        checkpoint_path = project_path(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint.get("class_map") != self.config.class_map:
            raise RuntimeError("Checkpoint class_map does not match the active dataset class_map")
        saved_config = checkpoint.get("config", {})
        for section_name in ("model", "optimizer", "scheduler"):
            if saved_config.get(section_name) != self.config.values.get(section_name):
                raise RuntimeError(
                    f"Checkpoint {section_name} configuration does not match the active run. "
                    "Resume with the original architecture/optimizer/scheduler config; only runtime limits such as "
                    "epochs and max_steps may be changed."
                )
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state"])
        self.scaler.load_state_dict(checkpoint.get("scaler_state", {}))
        self.start_epoch = int(checkpoint["epoch"]) + 1
        self.global_step = int(checkpoint["global_step"])
        self.best_map = float(checkpoint.get("best_map", -math.inf))
        if checkpoint.get("torch_rng_state") is not None:
            torch.set_rng_state(checkpoint["torch_rng_state"])
        if torch.cuda.is_available() and checkpoint.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
        loader_generator = getattr(self.train_loader, "generator", None)
        if loader_generator is not None and checkpoint.get("train_loader_generator_state") is not None:
            loader_generator.set_state(checkpoint["train_loader_generator_state"])
        sampler_generator = getattr(getattr(self.train_loader, "sampler", None), "generator", None)
        if sampler_generator is not None and checkpoint.get("train_sampler_generator_state") is not None:
            sampler_generator.set_state(checkpoint["train_sampler_generator_state"])
        self._log({"event": "resumed", "checkpoint": str(checkpoint_path), "global_step": self.global_step})

    def _optimizer_step(self) -> None:
        optimizer_config = self.config.section("optimizer")
        warmup_steps = int(optimizer_config.get("warmup_steps", 0))
        if warmup_steps > 0 and self.global_step < warmup_steps:
            start_factor = float(optimizer_config.get("warmup_start_factor", 0.001))
            progress = float(self.global_step + 1) / float(warmup_steps)
            factor = start_factor + (1.0 - start_factor) * progress
            for group, base_learning_rate in zip(self.optimizer.param_groups, self.base_learning_rates):
                group["lr"] = base_learning_rate * factor
        gradient_clip = float(self.config.section("optimizer").get("gradient_clip_norm", 0.0))
        if gradient_clip > 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        self.global_step += 1

    def _train_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        accumulation = int(self.training_config["gradient_accumulation_steps"])
        max_steps = self.training_config.get("max_steps")
        running_loss = 0.0
        micro_batches = 0
        epoch_started = time.perf_counter()

        for batch_index, (images, targets) in enumerate(self.train_loader):
            if max_steps is not None and self.global_step >= int(max_steps):
                break
            images = [image.to(self.device, non_blocking=True) for image in images]
            model_targets = [target_for_model(target, self.device) for target in targets]
            try:
                with torch.amp.autocast(self.device.type, enabled=self.amp_enabled):
                    loss_components = self.model(images, model_targets)
                    loss = sum(loss_components.values())
                if not torch.isfinite(loss):
                    component_values = {name: float(value.detach().cpu()) for name, value in loss_components.items()}
                    raise FloatingPointError(f"Non-finite training loss: {component_values}")
                # Do not underweight the final, shorter accumulation group at
                # the natural end of an epoch.
                group_start = (batch_index // accumulation) * accumulation
                group_size = min(accumulation, len(self.train_loader) - group_start)
                self.scaler.scale(loss / group_size).backward()
            except torch.OutOfMemoryError as error:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise RuntimeError(
                    "CUDA ran out of memory. Use the Stage 02 4 GB config, close GPU applications, "
                    "or lower data.crop_size/max_long_side and proposal counts."
                ) from error

            micro_batches += 1
            loss_value = float(loss.detach().cpu())
            running_loss += loss_value
            end_of_loader = batch_index + 1 == len(self.train_loader)
            if micro_batches % accumulation == 0 or end_of_loader:
                self._optimizer_step()
                payload = {
                    "event": "train_step",
                    "epoch": epoch,
                    "global_step": self.global_step,
                    "loss": loss_value,
                    "learning_rate": self.optimizer.param_groups[0]["lr"],
                    **{name: float(value.detach().cpu()) for name, value in loss_components.items()},
                }
                if self.global_step % int(self.training_config["log_every_steps"]) == 0 or self.global_step == 1:
                    self._log(payload)
                    self.writer.add_scalar("train/loss", loss_value, self.global_step)
                    for name, value in loss_components.items():
                        self.writer.add_scalar(f"train/{name}", float(value.detach().cpu()), self.global_step)
                if max_steps is not None and self.global_step >= int(max_steps):
                    break

        if micro_batches == 0:
            return {"mean_loss": float("nan"), "seconds": time.perf_counter() - epoch_started}
        return {"mean_loss": running_loss / micro_batches, "seconds": time.perf_counter() - epoch_started}

    def evaluate(self) -> dict[str, Any] | None:
        if self.val_loader is None or not bool(self.evaluation_config["enabled"]):
            return None
        metrics = evaluate_instance_segmentation(
            self.model,
            self.val_loader,
            self.device,
            score_threshold=float(self.evaluation_config["score_threshold"]),
            class_metrics=bool(self.evaluation_config["class_metrics"]),
            max_batches=self.evaluation_config["max_batches"],
            class_map=self.config.class_map,
        )
        self.last_evaluation = metrics
        self._log({"event": "evaluation", "global_step": self.global_step, "metrics": metrics})
        for name in (
            "primary_map",
            "bbox_map",
            "bbox_map_50",
            "bbox_map_75",
            "bbox_mar_100",
            "segm_map",
            "segm_map_50",
            "segm_map_75",
            "segm_mar_100",
        ):
            if isinstance(metrics.get(name), (float, int)):
                self.writer.add_scalar(f"validation/{name}", metrics[name], self.global_step)
        return metrics

    def train(self) -> dict[str, Any]:
        epochs = int(self.training_config["epochs"])
        max_steps = self.training_config.get("max_steps")
        completed_epoch = self.start_epoch - 1
        try:
            for epoch in range(self.start_epoch, epochs):
                train_metrics = self._train_epoch(epoch)
                completed_epoch = epoch
                self.scheduler.step()
                evaluation = None
                should_evaluate = (
                    bool(self.evaluation_config["enabled"])
                    and (epoch + 1) % int(self.training_config["evaluate_every_epochs"]) == 0
                )
                if should_evaluate:
                    evaluation = self.evaluate()
                current_map = float(evaluation["primary_map"]) if evaluation and "primary_map" in evaluation else -math.inf
                is_best = current_map > self.best_map
                if is_best:
                    self.best_map = current_map
                checkpoint = self.save_checkpoint(epoch, is_best=is_best)
                self._log(
                    {
                        "event": "epoch_complete",
                        "epoch": epoch,
                        "global_step": self.global_step,
                        "train": train_metrics,
                        "checkpoint": str(checkpoint),
                    }
                )
                if max_steps is not None and self.global_step >= int(max_steps):
                    break
        finally:
            self.writer.flush()
            self.writer.close()

        result = {
            "completed_epoch": completed_epoch,
            "global_step": self.global_step,
            "best_map": None if self.best_map == -math.inf else self.best_map,
            "last_evaluation": self.last_evaluation,
            "checkpoint": str(self.output_dir / "last.pt"),
        }
        if self.device.type == "cuda":
            total_memory = torch.cuda.get_device_properties(self.device).total_memory
            peak_allocated = torch.cuda.max_memory_allocated(self.device)
            peak_reserved = torch.cuda.max_memory_reserved(self.device)
            result["cuda_memory"] = {
                "peak_allocated_bytes": peak_allocated,
                "peak_reserved_bytes": peak_reserved,
                "total_bytes": total_memory,
                "peak_reserved_fraction": peak_reserved / total_memory,
            }
        atomic_json_dump(result, self.output_dir / "training_result.json")
        return result
