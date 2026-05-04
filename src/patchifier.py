from math import sqrt
from random import choice
from typing import Literal

from einops import rearrange
from pydantic import BaseModel
from torch import nn, Tensor
import torch

from .encoding import RoPE, APE


def resize(image: Tensor, size: tuple[int, int]) -> Tensor:
    return torch.nn.functional.interpolate(
        image[None, ...], size, mode="bilinear", antialias=True, align_corners=False
    )[0]


class PatchifierOutput(BaseModel, arbitrary_types_allowed=True):
    tokens: Tensor  # (sum_N, embed_dim)
    cu_seqlens: Tensor  # (B+1,) int32
    max_seqlen: int
    patch_hw: list[tuple[int, int]]
    patch_coords: Tensor  # (sum_N, 2) long
    rope_cos: Tensor  # (sum_N, head_dim)
    rope_sin: Tensor
    is_patch: Tensor  # (sum_N,) bool


class VarlenPatchifier(nn.Module):
    methods = ("resize", "drop")

    def __init__(
        self,
        *,
        in_channels: int = 3,
        patch_size: int = 16,
        embed_dim: int = 768,
        num_heads: int = 12,
        max_seq_len: int = 65_536,
        method: Literal["resize", "drop", "random"] = "resize",
        num_registers: int = 0,  # e.g. 1 for CLS, 5 for CLS + 4 registers
        with_ape: bool = False,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim {embed_dim} not divisible by num_heads {num_heads}"
            )
        self.min_patches_per_side = 2
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.max_seq_len = max_seq_len
        self.method = method
        self.num_registers = num_registers
        self.with_ape = with_ape

        self.proj = nn.Linear(in_channels * patch_size * patch_size, embed_dim)
        self.rope = RoPE(2, self.head_dim, base=10_000.0)
        self.ape = APE(2, embed_dim, base=10_000.0) if with_ape else None
        if num_registers > 0:
            self.registers = nn.Parameter(torch.empty(num_registers, embed_dim))
            nn.init.trunc_normal_(self.registers, std=0.02)
            # https://arxiv.org/abs/2602.08071
            self.register_rope = RoPE(2, self.head_dim, base=100_000.0)
            prefix_pos = torch.arange(num_registers)
            prefix_pos = torch.stack([prefix_pos, prefix_pos], dim=-1)
            self.register_buffer("register_positions", prefix_pos)

    def _shrink_grid(self, height: int, width: int, budget: int) -> tuple[int, int]:
        Hp_nat = max(1, height // self.patch_size)
        Wp_nat = max(1, width // self.patch_size)
        if Hp_nat * Wp_nat <= budget:
            return Hp_nat, Wp_nat
        Wp = max(self.min_patches_per_side, int(sqrt(budget * width / height)))
        Hp = max(self.min_patches_per_side, int(Wp * height / width))
        while Hp * Wp > budget and Wp > self.min_patches_per_side:
            Wp -= 1
            Hp = max(self.min_patches_per_side, int(Wp * height / width))
        return Hp, Wp

    def _patchify(self, img: Tensor, Hp: int, Wp: int) -> tuple[Tensor, Tensor]:
        p = self.patch_size
        patches = rearrange(img, "c (hp p1) (wp p2) -> (hp wp) (c p1 p2)", p1=p, p2=p)
        ys, xs = torch.meshgrid(
            torch.arange(Hp, device=img.device),
            torch.arange(Wp, device=img.device),
            indexing="ij",
        )
        coords = rearrange(torch.stack([ys, xs], dim=-1), "hp wp two -> (hp wp) two")
        return patches, coords

    @torch.compiler.disable
    def forward(self, images: list[Tensor]) -> PatchifierOutput:  # type: ignore
        batch_size, device = len(images), images[0].device
        patch_budget = max(
            1, (self.max_seq_len - batch_size * self.num_registers) // batch_size
        )
        raw_patches_list, patch_coords_list, patch_hw, patch_seqlens = [], [], [], []
        patches, coords, Hp, Wp = None, None, None, None
        for img in images:
            _, height, width = img.shape
            match self.method if self.method != "random" else choice(self.methods):
                case "resize":
                    Hp, Wp = self._shrink_grid(height, width, patch_budget)
                    th, tw = Hp * self.patch_size, Wp * self.patch_size
                    if (height, width) != (th, tw):
                        img = resize(img, (th, tw))
                    patches, coords = self._patchify(img, Hp, Wp)

                case "drop":
                    Hp = max(1, height // self.patch_size)
                    Wp = max(1, width // self.patch_size)
                    img = img[:, : Hp * self.patch_size, : Wp * self.patch_size]
                    patches, coords = self._patchify(img, Hp, Wp)
                    if self.training and patches.shape[0] > patch_budget:
                        idx = torch.randperm(patches.shape[0])[:patch_budget]
                        patches, coords = patches[idx], coords[idx]

            raw_patches_list.append(patches)
            patch_coords_list.append(coords)
            patch_hw.append((Hp, Wp))
            patch_seqlens.append(patches.shape[0] if patches is not None else 0)

        projected = self.proj(torch.cat(raw_patches_list, dim=0))  # (sum_patches, D)
        all_patch_coords = torch.cat(patch_coords_list, dim=0)
        patch_cos, patch_sin = self.rope(all_patch_coords)
        if self.ape is not None:
            norm_parts = []
            for coords_i, (Hp, Wp) in zip(patch_coords_list, patch_hw):
                denom = torch.tensor(
                    [max(Hp - 1, 1), max(Wp - 1, 1)], dtype=torch.float32, device=device
                )
                norm_parts.append(coords_i.float() / denom)
            projected = projected + self.ape(torch.cat(norm_parts, dim=0))

        # Assemble final sequence: [register, patches] per image.
        if self.num_registers > 0:
            proj_pi = projected.split(patch_seqlens)
            pcos_pi = patch_cos.split(patch_seqlens)
            psin_pi = patch_sin.split(patch_seqlens)
            pref_cos, pref_sin = self.register_rope(self.register_positions)  # (P, D)
            tok, crd, cs, sn = [], [], [], []
            for p, c, pc, ps in zip(proj_pi, patch_coords_list, pcos_pi, psin_pi):
                tok += [self.registers, p]
                crd += [self.register_positions, c]
                cs += [pref_cos, pc]
                sn += [pref_sin, ps]
            tokens, patch_coords = torch.cat(tok, dim=0), torch.cat(crd, dim=0)
            rope_cos, rope_sin = torch.cat(cs, dim=0), torch.cat(sn, dim=0)
            seqlens = [self.num_registers + n for n in patch_seqlens]
        else:
            tokens, patch_coords = projected, all_patch_coords
            rope_cos, rope_sin = patch_cos, patch_sin
            seqlens = patch_seqlens

        cu = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
        cu[1:] = torch.tensor(seqlens, dtype=torch.int32, device=device).cumsum(0)

        is_patch = torch.ones(tokens.shape[0], dtype=torch.bool, device=device)
        if self.num_registers > 0:
            starts = cu[:-1]  # (B,)
            offsets = torch.arange(self.num_registers, device=device)
            prefix_idx = (starts[:, None] + offsets[None, :]).flatten()
            is_patch[prefix_idx] = False  # registers are prefixes

        # pool patch tokens: output.tokens[output.is_patch]
        # specific register slot r: output.tokens[output.cu_seqlens[:-1] + r]
        return PatchifierOutput(
            tokens=tokens,
            cu_seqlens=cu,
            max_seqlen=max(seqlens),
            patch_hw=patch_hw,
            patch_coords=patch_coords,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            is_patch=is_patch,
        )
