from typing import Literal

from timm.optim.muon import Muon
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.scheduler.scheduler import Scheduler
from torch import nn
from torch.optim import Optimizer


def build_optimizer(
    model: nn.Module,
    learning_rate: float,
    weight_decay: float,
    momentum: float,
    nesterov: bool,
    algorithm: Literal["adam", "muon", "adamuon"] = "muon",
) -> Optimizer:
    return Muon(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.999),
        momentum=momentum,
        nesterov=nesterov,
        adjust_lr_fn=None if algorithm == "adamuon" else "match_rms_adamw",
        algo="adamuon" if algorithm == "adamuon" else "muon",
    )


def build_scheduler(
    optimizer: Optimizer, unit: Literal["epoch", "step"], total: int, warmup: int
) -> Scheduler:
    return CosineLRScheduler(
        optimizer,
        t_initial=total,
        lr_min=1e-6,
        warmup_t=warmup,
        warmup_lr_init=1e-6,
        t_in_epochs=unit == "epoch",
    )
