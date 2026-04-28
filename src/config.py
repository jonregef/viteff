from functools import partial
from pathlib import Path
from typing import Literal
import logging
import secrets
import sys

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    TomlConfigSettingsSource,
)


class DataConfig(BaseModel):
    train_shards: str = "./data/train-*.tar"
    valid_shards: str = "./data/val-*.tar"
    batch_size: int | list[int] = Field(
        256,
        description="Use a list (along with milestones) to enable curriculum learning (image resolution will adapt to keep the number of patches under but close to max_seq_len).",
    )
    milestones: list[int] | None = Field(
        None, description="Curriculum milestones in global unit (must begin with 0)"
    )
    max_seq_len: int = Field(65_536, description="Set to maximize GPU utilization.")
    threads: int = 16

    @model_validator(mode="after")
    def _validate_batch_size(self) -> "DataConfig":
        if isinstance(self.batch_size, int):
            if self.batch_size <= 0:
                raise ValueError("batch_size must be positive")
            if self.milestones is not None:
                logging.warning("milestones is ignored")
            return self
        sizes, milestones = self.batch_size, self.milestones
        if milestones is None:
            raise ValueError("milestones are required when batch_size is a list")
        if len(sizes) != len(milestones):
            raise ValueError(f"got {len(milestones)} milestones, expected {len(sizes)}")
        if any(s <= 0 for s in sizes):
            raise ValueError("batch sizes must be positive")
        if milestones[0] != 0:
            raise ValueError("first milestone must be 0")
        if any(b <= a for a, b in zip(milestones, milestones[1:])):
            raise ValueError("milestones must be strictly increasing")
        return self


class LoggingConfig(BaseModel):
    frequency: int = 1
    directory: Path = Path("./logs")
    project: str = "default"
    group: str = "default"


class CheckpointConfig(BaseModel):
    frequency: int = 1
    directory: Path = Path("./checkpoints")


class SegmentationConfig(BaseModel):
    task: Literal["segmentation"] = "segmentation"
    loss: Literal["dice"] = "dice"
    classes: int = 1000


class ClassificationConfig(BaseModel):
    task: Literal["classification"] = "classification"
    method: Literal["register", "avgpool", "attnpool"] = "register"
    loss: Literal["ce", "bce"] = "ce"
    classes: int = 1000
    smooth: float = 0.1


class SchedulerConfig(BaseModel):
    total: int = 300
    warmup: int = 5


class OptimizerConfig(BaseModel):
    learning_rate: float = 0.02
    weight_decay: float = 0.05
    momentum: float = 0.95
    nesterov: bool = False
    algorithm: Literal["adam", "muon", "adamuon"] = "muon"


class EmaConfig(BaseModel):
    enabled: bool = True
    decay: float = 0.9999


class ModelConfig(BaseModel):
    size: Literal["s", "b", "l", "xl", None] = "b"
    dim: int | None = None
    layers: int | None = None
    atten_heads: int | None = None
    registers: int | None = None
    sparse: bool = False
    patch_method: Literal["resize", "drop", "random"] = "resize"
    with_ape: bool = False
    head: ClassificationConfig | SegmentationConfig = Field(
        default_factory=ClassificationConfig, discriminator="task"
    )

    def _set_size(self, dim: int, layers: int, atten_heads: int, registers: int):
        self.dim = self.dim or dim
        self.layers = self.layers or layers
        self.atten_heads = self.atten_heads or atten_heads
        self.registers = self.registers or registers

    def model_post_init(self, context) -> None:
        match self.size:
            case "s":
                self._set_size(384, 12, 6, 4)
            case "b":
                self._set_size(768, 12, 12, 4)
            case "l":
                self._set_size(1024, 24, 16, 4)
            case "xl":
                self._set_size(1152, 28, 16, 4)


class ValidationConfig(BaseModel):
    frequency: int = 1
    batch_size: int = 256


class RunConfig(BaseSettings, cli_parse_args=True, cli_implicit_flags=True):
    config: Path | None = Field(
        None, description="TOML config file (priority: CLI > env > TOML > defaults)"
    )
    id: str = Field(
        default_factory=partial(secrets.token_hex, nbytes=8),
        description="Run id, manually set to append logs",
    )
    seed: int = 42
    precision: Literal["bf16", "fp8"] = "bf16"
    unit: Literal["epoch", "step"] = "epoch"
    clip_gradients: float | None = 1.0

    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    scheduler: SchedulerConfig = SchedulerConfig()
    logging: LoggingConfig = LoggingConfig()
    checkpoint: CheckpointConfig = CheckpointConfig()
    validation: ValidationConfig = ValidationConfig()
    ema: EmaConfig = EmaConfig()

    def model_post_init(self, context):
        self.logging.directory /= self.id
        self.checkpoint.directory /= self.id

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type["RunConfig"],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        args = sys.argv[1:]
        for i, arg in enumerate(args):
            if arg == "--config" and i + 1 < len(args):
                toml_path = Path(args[i + 1])
                if not toml_path.exists():
                    raise FileNotFoundError(f"Config file not found: {toml_path}")
                return (
                    init_settings,
                    env_settings,
                    TomlConfigSettingsSource(settings_cls, toml_file=toml_path),
                )
        return (init_settings, env_settings)
