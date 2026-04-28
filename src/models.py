from typing import Literal

from torch import nn, Tensor
from torchao.float8 import convert_to_float8_training

from .attention import VarlenBlock
from .config import ModelConfig, ClassificationConfig
from .heads.classification import (
    RegisterClassifier,
    AveragePoolingClassifier,
    AttentionPoolingClassifier,
)
from .patchifier import PatchifierOutput, VarlenPatchifier


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
    ) -> None:
        super().__init__()
        self.dim = dim
        self.registers = registers
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
            VarlenBlock(dim=dim, num_heads=atten_heads, sparse=sparse)
            for _ in range(layers)
        )
        self.out_norm = nn.RMSNorm(dim)

    def forward(self, images: list[Tensor]) -> tuple[Tensor, PatchifierOutput]:
        p: PatchifierOutput = self.patchifier(images)
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

    def metrics(self, logits: Tensor, labels: Tensor) -> dict[str, Tensor]:
        top5_pred = logits.topk(min(5, logits.size(-1)), dim=-1).indices
        return {
            "loss": self.loss(logits, labels),
            "top1": (top5_pred[:, 0] == labels).float().mean(),
            "top5": (top5_pred == labels.unsqueeze(-1)).any(-1).float().mean(),
        }


def build_model(max_seq_len: int, use_fp8: bool, config: ModelConfig) -> nn.Module:
    backbone = VarlenVisionTransformer(
        dim=config.dim,
        layers=config.layers,
        atten_heads=config.atten_heads,
        registers=config.registers,
        max_seq_len=max_seq_len,
        sparse=config.sparse,
        patch_method=config.patch_method,
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

    model.to("cuda")
    if use_fp8:
        model = convert_to_float8_training(
            model, module_filter_fn=lambda m: type(m) is nn.Linear
        )
    return model
