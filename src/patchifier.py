from itertools import accumulate
from math import sqrt
from random import choice
from typing import Literal, NamedTuple

from einops import rearrange
from torch import nn, Tensor
import torch

from .encoding import RoPE, APE


class PatchifierOutput(NamedTuple):
    tokens: Tensor  # (sum_N, embed_dim)
    cu_seqlens: Tensor  # (B+1,) int32, or (B+2,) when batch padding is appended
    max_seqlen: int
    patch_hw: list[tuple[int, int]]
    patch_coords: Tensor  # (sum_N, 2) long
    rope_cos: Tensor  # (sum_N, head_dim)
    rope_sin: Tensor
    is_patch: Tensor  # (sum_N,) bool, False for registers and trailing batch padding
    batch_size: int  # number of real images; cu_seqlens may have an extra pad segment


class _Pieces(NamedTuple):
    raw_patches: list[Tensor]  # unpadded per-image patch tensors
    coords: list[Tensor]  # unpadded per-image coord tensors
    seqlens: list[int]  # real patches per image (no registers, no padding)
    patch_hw: list[tuple[int, int]]
    cu: Tensor  # int32, B+1 or B+2 entries (extra entry when pad_rows > 0)
    max_seqlen: int
    denom: Tensor | None  # (B, 2) float32 for APE
    batch_size: int
    pad_rows: int  # 0..15, batch-level trailing padding for tensor-core alignment


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
        num_registers: int = 0,
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
        self.pad_to = 16  # for cuda tensor core alignment

        self.proj = nn.Linear(in_channels * patch_size * patch_size, embed_dim)
        self.rope = RoPE(2, self.head_dim, base=10_000.0)
        self.ape = APE(2, embed_dim, base=10_000.0) if with_ape else None
        if num_registers > 0:
            self.registers = nn.Parameter(torch.empty(num_registers, embed_dim))
            nn.init.trunc_normal_(self.registers, std=0.02)
            prefix_pos = torch.arange(num_registers)
            prefix_pos = torch.stack([prefix_pos, prefix_pos], dim=-1)
            self.register_buffer("register_positions", prefix_pos)
            # https://arxiv.org/abs/2602.08071
            register_rope = RoPE(2, self.head_dim, base=100_000.0)
            with torch.no_grad():
                cos, sin = register_rope(prefix_pos)
            self.register_buffer("register_rope_cos", cos.contiguous())
            self.register_buffer("register_rope_sin", sin.contiguous())

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
    def _get_pieces(self, images: list[Tensor]) -> _Pieces:
        batch_size, device = len(images), images[0].device
        patch_budget = max(
            1, (self.max_seq_len - batch_size * self.num_registers) // batch_size
        )
        raw, coords_list, hw, seqlens = [], [], [], []
        patches, coords, Hp, Wp = torch.zeros(()), torch.zeros(()), 0, 0
        for img in images:
            _, height, width = img.shape
            match self.method if self.method != "random" else choice(self.methods):
                case "resize":
                    Hp, Wp = self._shrink_grid(height, width, patch_budget)
                    th, tw = Hp * self.patch_size, Wp * self.patch_size
                    if (height, width) != (th, tw):
                        img = torch.nn.functional.interpolate(
                            img.unsqueeze(0),
                            (th, tw),
                            mode="bilinear",
                            antialias=True,
                            align_corners=False,
                        ).squeeze(0)
                    patches, coords = self._patchify(img, Hp, Wp)

                case "drop":
                    Hp = max(1, height // self.patch_size)
                    Wp = max(1, width // self.patch_size)
                    img = img[:, : Hp * self.patch_size, : Wp * self.patch_size]
                    patches, coords = self._patchify(img, Hp, Wp)
                    num_patches = patches.shape[0]
                    if self.training and num_patches > patch_budget:
                        idx = torch.randperm(num_patches, device=device)[:patch_budget]
                        patches, coords = patches[idx], coords[idx]

            raw.append(patches)
            coords_list.append(coords)
            hw.append((Hp, Wp))
            seqlens.append(patches.shape[0])

        # Pad the entire packed batch (registers + patches) to a multiple of
        # self.pad_to once, instead of padding each image individually.
        # This introduces at most pad_to-1 padding tokens per batch.
        seqlens_with_reg = [self.num_registers + n for n in seqlens]
        total = sum(seqlens_with_reg)
        mask = self.pad_to - 1
        pad_rows = ((total + mask) & ~mask) - total

        cu_entries = [0, *accumulate(seqlens_with_reg)]
        if pad_rows > 0:
            # Append a trailing pad-only segment. FlashAttention varlen
            # processes segments independently, so this segment is isolated
            # from the real images and cannot contaminate their attention.
            cu_entries.append(cu_entries[-1] + pad_rows)
        cu = torch.tensor(cu_entries, dtype=torch.int32, device=device)

        denom: Tensor | None = None
        if self.ape is not None:
            denom = torch.tensor(
                [[max(Hp - 1, 1), max(Wp - 1, 1)] for Hp, Wp in hw],
                dtype=torch.float,
                device=device,
            )
        max_seqlen = max(max(seqlens_with_reg, default=0), pad_rows)
        return _Pieces(
            raw_patches=raw,
            coords=coords_list,
            seqlens=seqlens,
            patch_hw=hw,
            cu=cu,
            max_seqlen=max_seqlen,
            denom=denom,
            batch_size=batch_size,
            pad_rows=pad_rows,
        )

    @torch.compiler.disable
    def forward(self, images: list[Tensor]) -> PatchifierOutput:  # type: ignore
        pieces: _Pieces = self._get_pieces(images)  # type: ignore
        device = images[0].device
        projected = self.proj(torch.cat(pieces.raw_patches, 0))  # (sum_real_patches, D)
        all_patch_coords = torch.cat(pieces.coords, 0)
        patch_cos, patch_sin = self.rope(all_patch_coords)
        if pieces.denom is not None and self.ape is not None:
            seqlens_t = torch.tensor(pieces.seqlens, device=device, dtype=torch.int64)
            denom_per_token = pieces.denom.repeat_interleave(seqlens_t, dim=0)
            projected = projected + self.ape(all_patch_coords / denom_per_token)

        # Assemble interleaved [reg_i, patches_i, ...] across real images, unpadded.
        if self.num_registers > 0:
            proj_pi = projected.split(pieces.seqlens)
            pcos_pi = patch_cos.split(pieces.seqlens)
            psin_pi = patch_sin.split(pieces.seqlens)
            pref_cos, pref_sin = self.register_rope_cos, self.register_rope_sin
            tok, crd, cs, sn = [], [], [], []
            for i, c, pc, ps in zip(proj_pi, pieces.coords, pcos_pi, psin_pi):
                tok += [self.registers, i]
                crd += [self.register_positions, c]
                cs += [pref_cos, pc]
                sn += [pref_sin, ps]
            tokens, patch_coords = torch.cat(tok, dim=0), torch.cat(crd, dim=0)
            rope_cos, rope_sin = torch.cat(cs, dim=0), torch.cat(sn, dim=0)
        else:
            tokens, patch_coords = projected, all_patch_coords
            rope_cos, rope_sin = patch_cos, patch_sin

        # Append a single trailing pad block for the whole batch. RoPE for these
        # rows is set to identity rotation (cos=1, sin=0) — they only attend to
        # themselves via cu_seqlens segmentation, so the values do not matter.
        if pieces.pad_rows > 0:
            n = pieces.pad_rows
            tokens = torch.cat([tokens, tokens.new_zeros(n, tokens.shape[1])], dim=0)
            patch_coords = torch.cat(
                [patch_coords, patch_coords.new_zeros(n, patch_coords.shape[1])], dim=0
            )
            rope_cos = torch.cat(
                [rope_cos, rope_cos.new_ones(n, rope_cos.shape[1])], dim=0
            )
            rope_sin = torch.cat(
                [rope_sin, rope_sin.new_zeros(n, rope_sin.shape[1])], dim=0
            )

        is_patch = torch.ones(tokens.shape[0], dtype=torch.bool, device=device)

        if self.num_registers > 0:
            offsets = torch.arange(self.num_registers, device=device)
            # Per-image register positions; cu[batch_size:] may include the
            # trailing pad segment, which we deliberately skip here.
            prefix_idx = (
                pieces.cu[: pieces.batch_size].long()[:, None] + offsets[None, :]
            ).flatten()
            is_patch = is_patch.index_fill(0, prefix_idx, False)

        if pieces.pad_rows > 0:
            pad_start = tokens.shape[0] - pieces.pad_rows
            pad_idx = torch.arange(pieces.pad_rows, device=device) + pad_start
            is_patch = is_patch.index_fill(0, pad_idx, False)

        # pool patch tokens: output.tokens[output.is_patch]
        # specific register slot r: output.tokens[output.cu_seqlens[:output.batch_size] + r]
        return PatchifierOutput(
            tokens=tokens,
            cu_seqlens=pieces.cu,
            max_seqlen=pieces.max_seqlen,
            patch_hw=pieces.patch_hw,
            patch_coords=patch_coords,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            is_patch=is_patch,
            batch_size=pieces.batch_size,
        )
