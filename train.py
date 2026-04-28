from glob import glob
import logging
import time

from pydantic_settings import CliApp
from rich.logging import RichHandler
from torch import Tensor, nn
import torch
import torch._dynamo
import trackio

logging.basicConfig(level=logging.DEBUG, format="%(message)s", handlers=[RichHandler()])
for _ in ["cutlass", "torchao", "httpcore", "spdl", "filelock", "asyncio"]:
    logging.getLogger(_).setLevel(logging.WARNING)

from src.config import RunConfig
from src.curriculum import BatchSizeCurriculum
from src.dataloader import build_imagenet_dataloader, train_transform, val_transform
from src.models import build_model
from src.optimization import build_optimizer, build_scheduler
from src.utils import seed_everything

torch._dynamo.config.verbose = True


@torch.inference_mode()
def validate(
    model: nn.Module,
    urls: list[str],
    batch_size: int,
    threads: int,
    seed: int,
) -> dict[str, float]:
    dataloader = build_imagenet_dataloader(
        urls,
        batch_size=batch_size,
        threads=threads,
        transform=val_transform,
        train=False,
        seed=seed,
    )
    was_training = model.training
    model.eval()
    totals: dict[str, float] = {}
    n = 0
    try:
        for images, labels in dataloader:
            images = [_.to(device="cuda", non_blocking=True) for _ in images]
            labels = labels.to(device="cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(images)
                batch = model.metrics(logits, labels)  # type: ignore
            for k, v in batch.items():
                totals[k] = totals.get(k, 0.0) + v.item()
            n += labels.size(0)
    finally:
        if was_training:
            model.train()
    return {f"val/{k}": v / n for k, v in totals.items()}


def train(config: RunConfig) -> None:
    logging.debug(config)
    seed_everything(config.seed)
    trackio.init(
        name=config.id,
        project=config.logging.project,
        group=config.logging.group,
        resume="allow",
        auto_log_gpu=True,
        gpu_log_interval=60,
        config=config.model_dump(),
    )
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
    dataloader = build_imagenet_dataloader(
        train_urls,
        train=True,
        batch_size=config.data.batch_size,  # type: ignore # FIXME
        threads=config.data.threads,
        transform=train_transform,
        seed=config.seed,
    )
    samples_seen, start_time, loss = 0, time.perf_counter(), torch.empty(0)

    def hooks() -> None:
        nonlocal samples_seen, start_time
        current = now()
        if current % config.logging.frequency == 0:
            throughput = samples_seen / (time.perf_counter() - start_time)
            metrics = {
                "step": step,
                "epoch": epoch,
                "batch_size": curriculum.at(current),
                "samples_seen": samples_seen,
                "elapsed": time.perf_counter() - start_time,
                "throughput": throughput,
                "loss": loss.item() if loss.numel() > 0 else None,
            }
            logging.info(", ".join(f"{k}: {v}" for k, v in metrics.items()))
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
        for images, labels in dataloader:
            images = [_.to(device="cuda", non_blocking=True) for _ in images]
            labels = labels.to(device="cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = compiled(images)
                loss: Tensor = model.loss(logits, labels)  # type: ignore
            if config.unit == "step":
                scheduler.step_update(num_updates=step)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if config.clip_gradients is not None:
                nn.utils.clip_grad_norm_(model.parameters(), config.clip_gradients)
            optimizer.step()
            if model_ema is not None:
                model_ema.update_parameters(model)
            samples_seen += len(images)
            step += 1
            if config.unit == "step":
                hooks()
            if done():
                break
        epoch += 1
        if config.unit == "epoch":
            hooks()


class _TrainingApp(RunConfig):
    def cli_cmd(self) -> None:
        train(self)


if __name__ == "__main__":
    CliApp.run(_TrainingApp)
