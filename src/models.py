from typing import Literal

from torch import nn, Tensor
from torchao.float8 import convert_to_float8_training
import torch

from .attention import VarlenBlock
from .config import ModelConfig, ClassificationConfig
from .heads.classification import (
    RegisterClassifier,
    AveragePoolingClassifier,
    AttentionPoolingClassifier,
)
from .patchifier import PatchifierOutput, VarlenPatchifier


class Preprocessor(nn.Module):
    def __init__(
        self, mean: list[float] | None = None, std: list[float] | None = None
    ) -> None:
        super().__init__()
        self.register_buffer(
            "mean", torch.tensor(mean or [0.5], dtype=torch.float32).view(-1, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor(std or [0.5], dtype=torch.float32).view(-1, 1, 1)
        )

    @torch.compiler.disable
    def forward(self, images: list[Tensor]) -> list[Tensor]:  # type: ignore
        return [
            (image.to(self.mean.dtype) / 255.0 - self.mean) / self.std  # type: ignore
            for image in images
        ]


class VarlenVisionTransformer(nn.Module):
    def __init__(
        self,
        dim: int,
        layers: int,
        atten_heads: int,
        registers: int,
        max_seq_len: int,
        sparse: bool = False,
        in_channels: int = 3,
        patch_size: int = 16,
        patch_method: Literal["resize", "drop", "random"] = "resize",
        with_ape: bool = False,
        proj_drop: float = 0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.registers = registers
        self.preprocessor = Preprocessor()
        self.patchifier = VarlenPatchifier(
            in_channels=in_channels,
            patch_size=patch_size,
            embed_dim=dim,
            num_heads=atten_heads,
            max_seq_len=max_seq_len,
            method=patch_method,
            num_registers=registers,
            with_ape=with_ape,
        )
        self.blocks = nn.ModuleList(
            VarlenBlock(dim, num_heads=atten_heads, sparse=sparse, proj_drop=proj_drop)
            for _ in range(layers)
        )
        self.out_norm = nn.RMSNorm(dim)

    def forward(self, images: list[Tensor]) -> tuple[Tensor, PatchifierOutput]:
        p: PatchifierOutput = self.patchifier(self.preprocessor(images))
        x = p.tokens
        del p.tokens
        for block in self.blocks:
            x = block(x, p.cu_seqlens, p.max_seqlen, p.rope_cos, p.rope_sin)
        x = self.out_norm(x)
        return x, p


class ClassificationViT(nn.Module):
    def __init__(
        self,
        backbone: VarlenVisionTransformer,
        num_classes: int,
        method: Literal["register", "avgpool", "attnpool"],
        loss: Literal["ce", "bce"],
        smooth: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.backbone = backbone
        match method:
            case "register":
                if backbone.registers < 1:
                    raise ValueError("RegisterClassifier requires registers >= 1")
                self.head = RegisterClassifier(backbone.dim, num_classes)
            case "avgpool":
                self.head = AveragePoolingClassifier(backbone.dim, num_classes)
            case "attnpool":
                self.head = AttentionPoolingClassifier(backbone.dim, num_classes)
        match loss:
            case "ce":
                self.loss = nn.CrossEntropyLoss(label_smoothing=smooth)
            case "bce":
                self.loss = nn.BCEWithLogitsLoss()

    def forward(self, images: list[Tensor]) -> Tensor:
        return self.head(*self.backbone(images))

    def forward_with_target(
        self, images: list[Tensor], labels: Tensor
    ) -> dict[str, Tensor]:
        logits = self.forward(images)
        if isinstance(self.loss, nn.BCEWithLogitsLoss) and len(labels.shape) == 1:
            onehot = torch.nn.functional.one_hot(labels, self.num_classes).to(logits)
            loss = self.loss(logits, onehot)
        else:
            loss = self.loss(logits, labels)
        top5_pred = logits.topk(min(5, logits.size(-1)), dim=-1).indices
        return {
            "loss": loss,
            "top1": (top5_pred[:, 0] == labels).float().mean(),
            "top5": (top5_pred == labels.unsqueeze(-1)).any(-1).float().mean(),
        }


def build_model(max_seq_len: int, use_fp8: bool, config: ModelConfig) -> nn.Module:
    assert config.dim and config.layers and config.atten_heads and config.registers
    backbone = VarlenVisionTransformer(
        dim=config.dim,
        layers=config.layers,
        atten_heads=config.atten_heads,
        registers=config.registers,
        max_seq_len=max_seq_len,
        sparse=config.sparse,
        patch_method=config.patch_method,
        proj_drop=config.proj_drop,
    )
    match config.head:
        case ClassificationConfig():
            model = ClassificationViT(
                backbone,
                num_classes=config.head.classes,
                method=config.head.method,
                loss=config.head.loss,
                smooth=config.head.smooth,
            )
        case _:
            raise NotImplementedError()

    if use_fp8:
        is_linear = lambda m, _: type(m) is nn.Linear  # noqa: E731
        model = convert_to_float8_training(model, module_filter_fn=is_linear)
    return model
