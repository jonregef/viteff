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


class AugmentationConfig(BaseModel):
    recipe: Literal["threeaug", "randaug", "trivialaug"] = "trivialaug"
    magnitude: float = Field(
        default=0.5, ge=0, le=1, description="0: identity, 0.5: default, 1: heavy"
    )
    with_flip: bool = True


class DataConfig(BaseModel):
    train_shards: str = "./data/train-*.tar"
    valid_shards: str = "./data/val-*.tar"
    batch_size: int | list[int] = Field(
        default=256,
        description="Use a list (along with milestones) to enable curriculum learning (image resolution will adapt to keep the number of patches under but close to max_seq_len).",
    )
    milestones: list[int] | None = Field(
        default=None,
        description="Curriculum milestones in global unit (must begin with 0).",
    )
    max_seq_len: int = Field(
        default=65_536, description="Set to maximize GPU utilization."
    )
    threads: int = 16
    resolution_cap: int = 256

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
    nesterov: bool = True
    algorithm: Literal["adam", "muon", "adamuon"] = "muon"


class EmaConfig(BaseModel):
    enabled: bool = True
    decay: float = 0.9999


class ModelConfig(BaseModel):
    size: Literal["s", "b", "l", "xl", "so150m", "so400m", None] = "b"
    dim: int = 512
    layers: int = 16
    atten_heads: int = 16
    registers: int = 4
    mlp_ratio: float = 4.0
    sparse: bool = False
    patch_method: Literal["resize", "drop", "random"] = "resize"
    with_ape: bool = False
    drop_path: float = 0.0
    layerscale: float | None = None
    activation: Literal["gelu", "relu2", "derf"] = "gelu"
    head: ClassificationConfig | SegmentationConfig = Field(
        default_factory=ClassificationConfig, discriminator="task"
    )

    def _maybe_overwrite(self, name: str, value) -> None:
        if name not in self.model_fields_set:
            setattr(self, name, value)

    def _set_size(
        self, dim: int, layers: int, atten_heads: int, registers: int, mlp_ratio: float
    ):
        self._maybe_overwrite("dim", dim)
        self._maybe_overwrite("layers", layers)
        self._maybe_overwrite("atten_heads", atten_heads)
        self._maybe_overwrite("registers", registers)
        self._maybe_overwrite("mlp_ratio", mlp_ratio)

    def model_post_init(self, context) -> None:
        match self.size:
            case "s":
                self._set_size(384, 12, 6, 4, 4.0)
            case "b":
                self._set_size(768, 12, 12, 4, 4.0)
            case "l":
                self._set_size(1024, 24, 16, 4, 4.0)
            case "xl":
                self._set_size(1152, 28, 16, 4, 4.0)
            # https://arxiv.org/abs/2305.13035
            case "so150m":
                self._set_size(880, 18, 16, 4, 2320 / 880)
            case "so400m":
                self._set_size(1152, 27, 16, 4, 4304 / 1152)


class ValidationConfig(BaseModel):
    frequency: int = 1
    batch_size: int = 256
    resolution_cap: int = 256


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

    augmentation: AugmentationConfig = AugmentationConfig()
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
        settings_cls: type[BaseSettings],
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
