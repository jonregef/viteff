from contextlib import redirect_stdout
from glob import glob
import io
import logging
import time

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
from src.dataloader import build_webdataset_dataloader, train_augs, val_augs
from src.models import build_model
from src.optimization import build_optimizer, build_scheduler
from src.utils import seed_everything


@torch.inference_mode()
def validate(
    model: nn.Module, urls: list[str], batch_size: int, threads: int, seed: int
) -> dict[str, float]:
    dataloader = build_webdataset_dataloader(
        urls,
        batch_size=batch_size,
        threads=threads,
        augs=val_augs,
        train=False,
        seed=seed,
    )
    was_training = model.training
    model.eval()
    totals: dict[str, float] = {}
    n = 0
    try:
        for images, labels in dataloader:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                batch = model.forward_with_target(images, labels)  # type: ignore
            for k, v in batch.items():
                totals[k] = totals.get(k, 0.0) + v.item()
            n += labels.size(0)
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
    model.to(device="cuda", dtype=torch.bfloat16)
    logging.debug(model)
    model_ema = None
    if config.ema.enabled:
        model_ema = torch.optim.swa_utils.AveragedModel(
            model,
            multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(config.ema.decay),
            use_buffers=True,
        )
    compiled = torch.compile(model, dynamic=True)

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

    step, epoch = 0, 0
    now = (lambda: step) if config.unit == "step" else (lambda: epoch)
    done = lambda: now() >= config.scheduler.total  # noqa: E731

    curriculum = BatchSizeCurriculum(
        config.data.batch_size, config.data.milestones, now
    )
    dataloader = build_webdataset_dataloader(
        train_urls,
        train=True,
        batch_size=config.data.batch_size,  # type: ignore # FIXME
        threads=config.data.threads,
        augs=train_augs,
        seed=config.seed,
    )
    samples_seen, start_time = 0, time.perf_counter()

    def hooks(loss: float) -> None:
        nonlocal samples_seen, start_time
        current = now()
        if current % config.logging.frequency == 0:
            throughput = samples_seen / (time.perf_counter() - start_time)
            metrics = {
                "step": step,
                "epoch": epoch,
                "batch_size": curriculum.at(current),
                "throughput": throughput,
                "loss": loss,
            }
            logging.info(", ".join(f"{k}: {v:.4g}" for k, v in metrics.items()))
            metrics.pop("step", None)
            trackio.log(metrics)
            samples_seen, start_time = 0, time.perf_counter()
        if current % config.checkpoint.frequency == 0:
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "step": step,
                    "epoch": epoch,
                    "model_ema": model_ema.module.state_dict() if model_ema else None,
                    "config": config.model_dump(),
                },
                config.checkpoint.directory / f"{step:08d}.pt",
            )
            logging.info("saved")
        if current % config.validation.frequency == 0:
            eval_target = model_ema.module if model_ema is not None else model
            logging.info("validating...")
            metrics = validate(
                eval_target,
                valid_urls,
                batch_size=config.validation.batch_size,
                threads=config.data.threads,
                seed=config.seed,
            )
            logging.info(", ".join(f"{k}: {v:.4f}" for k, v in metrics.items()))
            trackio.log(metrics)

    while not done():
        if config.unit == "epoch":
            scheduler.step(epoch)
        last_loss = 0.0
        optimizer.zero_grad(set_to_none=True)  # Needed?
        for images, labels in dataloader:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                batch_metrics = compiled.forward_with_target(images, labels)  # type: ignore
            loss = batch_metrics.pop("loss")
            if config.unit == "step":
                scheduler.step_update(num_updates=step)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            last_loss = loss.item()
            if config.clip_gradients is not None:
                nn.utils.clip_grad_norm_(model.parameters(), config.clip_gradients)
            optimizer.step()
            if model_ema is not None:
                model_ema.update_parameters(model)
            samples_seen += len(images)
            if config.unit == "step":
                hooks(last_loss)
            step += 1
            if done():
                break
        if config.unit == "epoch":
            hooks(last_loss)
        epoch += 1


class _TrainingApp(RunConfig):
    def cli_cmd(self) -> None:
        train(self)


if __name__ == "__main__":
    CliApp.run(_TrainingApp)
