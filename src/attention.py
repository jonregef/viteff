import logging
from typing import Literal

from einops import rearrange
from torch import nn, Tensor
from torch.nn.attention.varlen import varlen_attn
from torchao.sparsity.training import SemiSparseActivationLinear
import torch
import torch.nn.attention as attn

available = attn.list_flash_attention_impls()
if available:
    try:
        major, minor = torch.cuda.get_device_capability()
        attn.activate_flash_attention_impl(available[-1 if major > 8 else 0])
        logging.debug(f"Activated {attn.current_flash_attention_impl()} backend")
    except ModuleNotFoundError:
        logging.warning("Failed to set flash attention backend, using fallback")
        pass


class VarlenAttention(nn.Module):
    """Varlen multi-head self-attention for packed token sequences.

    Operates on packed tensors of shape (total_tokens, dim), where all images
    in the batch are concatenated and segmented by `cu_seqlens`. RoPE tables
    (from VarlenPatchifier) are applied to Q and K per-head. Uses FlashAttention-4
    varlen kernel. No padding, no mask — segmentation is done via cu_seqlens.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        *,
        qk_norm: bool = True,
        softmax_scale: float | None = None,
    ) -> None:
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} not divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads  # should be 64 or 128 for fast path
        self.softmax_scale = softmax_scale or self.head_dim**-0.5

        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = nn.RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim, bias=False)

    @staticmethod
    def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        """x: (..., N, D). cos/sin: (N, D) or any shape broadcastable to x."""
        x1, x2 = x.chunk(2, dim=-1)
        return x * cos + torch.cat([-x2, x1], dim=-1) * sin

    def forward(
        self,
        x: Tensor,  # (total_tokens, dim)
        cu_seqlens: Tensor,  # (B + 1,) int32
        max_seqlen: int,
        rope_cos: Tensor | None = None,  # (total_tokens, head_dim)
        rope_sin: Tensor | None = None,
    ) -> Tensor:
        qkv = rearrange(
            self.qkv(x),
            "t (three h d) -> three t h d",
            **dict(three=3, h=self.num_heads, d=self.head_dim),
        )
        q, k, v = qkv.unbind(0)  # each (total_tokens, h, d)
        q, k = self.q_norm(q.float()), self.k_norm(k.float())

        if rope_cos is not None and rope_sin is not None:
            # Broadcast RoPE over head dim: (total_tokens, 1, head_dim)
            cos, sin = rope_cos.unsqueeze(1), rope_sin.unsqueeze(1)
            q, k = self.apply_rope(q, cos, sin), self.apply_rope(k, cos, sin)

        q, k = q.to(v.dtype), k.to(v.dtype)
        out = varlen_attn(
            *(q, k, v),
            cu_seq_q=cu_seqlens,
            cu_seq_k=cu_seqlens,
            max_q=max_seqlen,
            max_k=max_seqlen,
            scale=self.softmax_scale,
            window_size=(-1, -1),
        )
        return self.proj(rearrange(out, "t h d -> t (h d)"))


class Derf(nn.Module):
    """https://arxiv.org/abs/2512.10938"""

    def __init__(self, dim: int, alpha: float = 0.5, shift: float = 0.0) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.full((dim,), alpha))
        self.shift = nn.Parameter(torch.full((dim,), shift))
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x: Tensor) -> Tensor:
        return self.gamma * torch.erf(self.alpha * x + self.shift) + self.beta


class SquaredReLU(nn.Module):
    """https://arxiv.org/abs/2402.03804"""

    def __init__(self, cap: float | None = None) -> None:
        super().__init__()
        self.cap = cap

    def forward(self, x: Tensor) -> Tensor:
        x = x.clamp(0, self.cap)
        return x * x


class LayerScale(nn.Module):
    def __init__(self, dim: int, init: float) -> None:
        super().__init__()
        self.layerscale = nn.Parameter(torch.full((dim,), init))

    def forward(self, x: Tensor) -> Tensor:
        return x * self.layerscale


class DropPath(nn.Module):
    def __init__(self, p: float = 0.0) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: Tensor, cu_seqlens: Tensor) -> Tensor:
        if self.p == 0.0 or not self.training:
            return x
        num_samples = cu_seqlens.numel() - 1
        keep = (torch.rand(num_samples, device=x.device) > self.p).to(x.dtype)
        keep = keep / (1.0 - self.p)
        seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.int64)
        keep_per_token = keep.repeat_interleave(seqlens).unsqueeze(-1)
        return x * keep_per_token


class VarlenBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        layerscale: float | None = 1e-4,
        sparse: bool = False,
        drop_path: float = 0.0,
        activation: Literal["gelu", "relu2", "derf"] = "gelu",
    ) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(dim)
        self.attn = VarlenAttention(dim, num_heads)
        self.norm2 = nn.RMSNorm(dim)
        hidden = round(dim * mlp_ratio)

        if layerscale:
            self.ls1 = LayerScale(dim, layerscale)
            self.ls2 = LayerScale(dim, layerscale)
        else:
            self.ls1, self.ls2 = nn.Identity(), nn.Identity()

        self.drop_path = DropPath(drop_path)
        linear1 = nn.Linear(dim, hidden, bias=False)
        match activation:
            case "gelu":
                act = nn.GELU()
            case "relu2":
                act = SquaredReLU()
            case "derf":
                act = Derf(dim)
        self.mlp = nn.Sequential(linear1, act, nn.Linear(hidden, dim, bias=False))

        if sparse and hidden % 128 == 0:
            # FIXME: couldn't get to work.
            # Sequence-length needs to be padded to a multiple of 128
            # but even then, backwards pass get shapes wrong and panics
            # Also currently incompatible with fp8 training
            self.mlp = nn.Sequential(
                linear1,
                SquaredReLU(),
                SemiSparseActivationLinear(hidden, dim, bias=False),
            )

    def forward(
        self,
        x: Tensor,  # (total_tokens, dim)
        cu_seqlens: Tensor,  # (B+1,) int32
        max_seqlen: int,
        rope_cos: Tensor | None = None,  # (total_tokens, head_dim)
        rope_sin: Tensor | None = None,  # (total_tokens, head_dim)
    ) -> Tensor:
        a = self.attn(self.norm1(x), cu_seqlens, max_seqlen, rope_cos, rope_sin)
        x = x + self.drop_path(self.ls1(a), cu_seqlens)
        x = x + self.drop_path(self.ls2(self.mlp(self.norm2(x))), cu_seqlens)
        return x
