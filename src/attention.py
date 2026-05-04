import logging

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


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """x: (..., N, D). cos/sin: (N, D) or any shape broadcastable to x."""
    x1, x2 = x.chunk(2, dim=-1)
    return x * cos + torch.cat([-x2, x1], dim=-1) * sin


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
        proj_drop: float = 0.0,
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
        self.proj_drop = nn.Dropout(proj_drop) if proj_drop > 0 else nn.Identity()

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
        q, k = self.q_norm(q), self.k_norm(k)

        if rope_cos is not None and rope_sin is not None:
            # Broadcast RoPE over head dim: (total_tokens, 1, head_dim)
            cos, sin = rope_cos.unsqueeze(1), rope_sin.unsqueeze(1)
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        out = varlen_attn(
            *(q, k, v),
            cu_seq_q=cu_seqlens,
            cu_seq_k=cu_seqlens,
            max_q=max_seqlen,
            max_k=max_seqlen,
            scale=self.softmax_scale,
            window_size=(-1, -1),
        )
        return self.proj_drop(self.proj(rearrange(out, "t h d -> t (h d)")))


class SquaredReLU(nn.Module):
    def __init__(self, cap: float | None = None) -> None:
        super().__init__()
        self.cap = cap

    def forward(self, x: Tensor) -> Tensor:
        x = x.clamp(0, self.cap)
        return x * x


class LayerScale(nn.Module):
    def __init__(self, dim: int, init: float) -> None:
        super().__init__()
        self.layer_scale = nn.Parameter(torch.full((dim,), init))

    def forward(self, x: Tensor) -> Tensor:
        return x * self.layer_scale


class VarlenBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        layerscale: float | None = 1e-4,
        sparse: bool = False,
        proj_drop: float = 0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(dim)
        self.attn = VarlenAttention(dim, num_heads, proj_drop=proj_drop)
        self.norm2 = nn.RMSNorm(dim)
        hidden = int(dim * mlp_ratio)

        if layerscale:
            self.ls1 = LayerScale(dim, layerscale)
            self.ls2 = LayerScale(dim, layerscale)
        else:
            self.ls1, self.ls2 = nn.Identity(), nn.Identity()

        if sparse and hidden % 16 == 0:
            self.mlp = nn.Sequential(
                nn.Linear(dim, hidden, bias=False),
                SquaredReLU(),  # 2402.03804
                SemiSparseActivationLinear(hidden, dim, bias=False),
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(dim, hidden, bias=False),
                nn.GELU(),
                nn.Linear(hidden, dim, bias=False),
            )

    def forward(
        self,
        x: Tensor,  # (total_tokens, dim)
        cu_seqlens: Tensor,  # (B+1,) int32
        max_seqlen: int,
        rope_cos: Tensor | None = None,  # (total_tokens, head_dim)
        rope_sin: Tensor | None = None,  # (total_tokens, head_dim)
    ) -> Tensor:
        x = x + self.ls1(
            self.attn(self.norm1(x), cu_seqlens, max_seqlen, rope_cos, rope_sin)
        )
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x
