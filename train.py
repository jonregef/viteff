from contextlib import redirect_stdout
from functools import partial
from glob import glob
import io
import logging

from pydantic_settings import CliApp
from rich.logging import RichHandler
from rich.text import Text
from torch import Tensor, nn
import torch
import trackio

logging.basicConfig(level=logging.DEBUG, format="%(message)s", handlers=[RichHandler()])
for _ in ["cutlass", "torchao", "httpcore", "spdl", "filelock", "asyncio", "PIL"]:
    logging.getLogger(_).setLevel(logging.WARNING)

from src.config import RunConfig
from src.curriculum import BatchSizeCurriculum
from src.dataloader import build_webdataset_dataloader, train_augs
from src.hooks import CheckpointHook, Hook, LoggingHook, TrainerState, ValidationHook
from src.models import build_model
from src.optimization import build_optimizer, build_scheduler
from src.utils import seed_everything


@torch.inference_mode()
def validate(
    model: nn.Module, *, urls: list[str], batch_size: int, threads: int, seed: int
) -> dict[str, float]:
    dataloader = build_webdataset_dataloader(
        urls, batch_size=batch_size, threads=threads, augs=None, train=False, seed=seed
    )
    was_training = model.training
    model.eval()
    totals: dict[str, float] = {}
    n = 0
    try:
        for images, labels in dataloader:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                batch = model.forward_with_target(images, labels)  # type: ignore
            batch_size = labels.size(0)
            for k, v in batch.items():
                totals[k] = totals.get(k, 0.0) + v.item() * batch_size
            n += batch_size
    finally:
        model.train(was_training)
        del dataloader
        torch.cuda.empty_cache()
    return {f"val/{k}": v / n for k, v in totals.items()}


def train(config: RunConfig) -> None:
    logging.debug(config)
    seed_everything(config.seed)
    f = io.StringIO()
    with redirect_stdout(f):
        trackio.init(
            name=config.id,
            project=config.logging.project,
            group=config.logging.group,
            resume="allow",
            auto_log_gpu=True,
            gpu_log_interval=60,
            config=config.model_dump(),
        )
    logging.info(Text.from_ansi(f.getvalue().strip()))
    model = build_model(
        max_seq_len=config.data.max_seq_len,
        use_fp8=config.precision == "fp8",
        config=config.model,
    )
    model.to(device="cuda")
    logging.debug(model)
    model_ema = None
    if config.ema.enabled:
        model_ema = torch.optim.swa_utils.AveragedModel(
            model,
            multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(config.ema.decay),
            use_buffers=True,
        )
    compiled = torch.compile(model.forward_with_target, dynamic=True)  # type: ignore

    train_urls = sorted(glob(config.data.train_shards))
    if not train_urls:
        raise FileNotFoundError(f"No shards match {config.data.train_shards}")
    valid_urls = sorted(glob(config.data.valid_shards))
    if not valid_urls:
        raise FileNotFoundError(f"No shards match {config.data.valid_shards}")
    logging.debug(f"{len(train_urls)} training and {len(valid_urls)} validation shards")

    optimizer = build_optimizer(model, **config.optimizer.model_dump())
    scheduler = build_scheduler(optimizer, config.unit, **config.scheduler.model_dump())

    config.logging.directory.mkdir(parents=True, exist_ok=True)
    config.checkpoint.directory.mkdir(parents=True, exist_ok=True)

    state = TrainerState(
        unit=config.unit,
        model=model,
        model_ema=model_ema,
        optimizer=optimizer,
        scheduler=scheduler,
        curriculum=None,  # type: ignore[arg-type]
    )
    state.curriculum = BatchSizeCurriculum(
        config.data.batch_size, config.data.milestones, lambda: state.now
    )
    done = lambda: state.now >= config.scheduler.total  # noqa: E731

    dataloader = build_webdataset_dataloader(
        train_urls,
        train=True,
        batch_size=config.data.batch_size,  # type: ignore # FIXME
        threads=config.data.threads,
        augs=train_augs,
        seed=config.seed,
    )
    hooks: list[Hook] = [
        LoggingHook(frequency=config.logging.frequency),
        CheckpointHook(
            frequency=config.checkpoint.frequency,
            directory=config.checkpoint.directory,
            config_dump=config.model_dump(),
        ),
        ValidationHook(
            frequency=config.validation.frequency,
            validate_fn=partial(
                validate,
                urls=valid_urls,
                batch_size=config.validation.batch_size,
                threads=config.data.threads,
                seed=config.seed,
            )
        ),    
    ]
    for h in hooks:
        h.on_train_start(state)

    while not done():
        if config.unit == "epoch":
            scheduler.step(state.epoch)
        last_loss = torch.zeros((), device="cuda")
        optimizer.zero_grad(set_to_none=True)  # Needed?
        for images, labels in dataloader:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                batch_metrics = compiled(images, labels)  # type: ignore
            loss: Tensor = batch_metrics.pop("loss")
            if config.unit == "step":
                scheduler.step_update(num_updates=state.step)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            last_loss = loss.detach()
            if config.clip_gradients is not None:
                nn.utils.clip_grad_norm_(model.parameters(), config.clip_gradients)
            optimizer.step()
            if model_ema is not None:
                model_ema.update_parameters(model)
            state.samples_seen_delta += len(images)
            if config.unit == "step":
                state.last_loss = last_loss.item()
                for h in hooks: 
                    h.on_tick(state)
            state.step += 1
            if done():
                break
        if config.unit == "epoch":
            state.last_loss = last_loss.item()
            for h in hooks:
                h.on_tick(state)
        state.epoch += 1

    for h in hooks:
        h.on_train_end(state)


class _TrainingApp(RunConfig):
    def cli_cmd(self) -> None:
        train(self)


if __name__ == "__main__":
    CliApp.run(_TrainingApp)
