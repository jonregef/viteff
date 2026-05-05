from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal
import logging
import time

from torch import nn
import torch
import trackio

from src.dataloader import BatchSizeCurriculum


@dataclass
class TrainerState:
    unit: Literal["step", "epoch"]
    model: nn.Module
    model_ema: torch.optim.swa_utils.AveragedModel | None
    optimizer: torch.optim.Optimizer
    scheduler: Any
    curriculum: BatchSizeCurriculum
    step: int = 0
    epoch: int = 0
    last_loss: float = 0.0
    samples_seen_delta: int = 0

    @property
    def now(self) -> int:
        return self.step if self.unit == "step" else self.epoch


class Hook:
    frequency: int = 1

    def on_train_start(self, state: TrainerState) -> None: ...

    def on_train_end(self, state: TrainerState) -> None: ...

    def on_tick(self, state: TrainerState) -> None:
        if state.now % self.frequency == 0:
            self.step(state)

    def step(self, state: TrainerState) -> None: ...


class LoggingHook(Hook):
    def __init__(self, frequency: int) -> None:
        self.frequency = frequency
        self._start_time = 0.0

    def on_train_start(self, state: TrainerState) -> None:
        self._start_time = time.perf_counter()

    def step(self, state: TrainerState) -> None:
        throughput = state.samples_seen_delta / (time.perf_counter() - self._start_time)
        metrics = {
            "step": state.step,
            "epoch": state.epoch,
            "batch_size": state.curriculum.at(state.now),
            "throughput": throughput,
            "loss": state.last_loss,
        }
        logging.info(", ".join(f"{k}: {v:.4g}" for k, v in metrics.items()))
        metrics.pop("step", None)
        trackio.log(metrics)
        state.samples_seen_delta = 0
        self._start_time = time.perf_counter()


class CheckpointHook(Hook):
    def __init__(
        self, frequency: int, directory: Path, config_dump: dict[str, Any]
    ) -> None:
        self.frequency = frequency
        self.directory = directory
        self.config_dump = config_dump

    def step(self, state: TrainerState) -> None:
        torch.save(
            {
                "model": state.model.state_dict(),
                "optimizer": state.optimizer.state_dict(),
                "scheduler": state.scheduler.state_dict(),
                "step": state.step,
                "epoch": state.epoch,
                "model_ema": (
                    state.model_ema.module.state_dict() if state.model_ema else None
                ),
                "config": self.config_dump,
            },
            self.directory / f"{state.step:08d}.pt",
        )
        logging.info("saved")


class ValidationHook(Hook):
    def __init__(
        self, frequency: int, validate_fn: Callable[[nn.Module], dict[str, float]]
    ) -> None:
        self.frequency = frequency
        self.validate_fn = validate_fn

    def step(self, state: TrainerState) -> None:
        eval_target = state.model_ema.module if state.model_ema else state.model
        logging.info("validating...")
        metrics = self.validate_fn(eval_target)
        logging.info(", ".join(f"{k}: {v:.4f}" for k, v in metrics.items()))
        trackio.log(metrics)
