from typing import Literal

from timm.optim.muon import Muon
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.scheduler.scheduler import Scheduler
from torch import nn
from torch.optim import Optimizer


def _no_decay(name: str, p) -> bool:
    return name.endswith(".layer_scale") or name.endswith("registers")


def _no_muon(name: str, p) -> bool:
    return p.ndim < 2


def build_optimizer(
    model: nn.Module,
    learning_rate: float,
    weight_decay: float,
    momentum: float,
    nesterov: bool,
    algorithm: Literal["adam", "muon", "adamuon"] = "muon",
) -> Optimizer:
    decay_muon, no_decay_muon, decay_adam, no_decay_adam = [], [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if _no_muon(name, p) or algorithm == "adam":
            (no_decay_adam if _no_decay(name, p) else decay_adam).append(p)
        else:
            (no_decay_muon if _no_decay(name, p) else decay_muon).append(p)

    param_groups = [
        {"params": decay_muon, "weight_decay": weight_decay, "use_fallback": False},
        {"params": no_decay_muon, "weight_decay": 0.0, "use_fallback": False},
        {"params": decay_adam, "weight_decay": weight_decay, "use_fallback": True},
        {"params": no_decay_adam, "weight_decay": 0.0, "use_fallback": True},
    ]
    return Muon(
        param_groups,
        lr=learning_rate,
        betas=(0.9, 0.95),
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
