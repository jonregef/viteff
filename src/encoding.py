from einops import rearrange, repeat
from torch import nn, Tensor
import torch


class APE(nn.Module):
    """N-D sinusoidal Absolute Positional Encoding."""

    def __init__(self, axes: int, dim: int, base: float = 10_000.0) -> None:
        super().__init__()
        if dim % (2 * axes) != 0:
            raise ValueError(f"dim must be divisible by {2 * axes}, got {dim}")
        self.axes = axes
        freq_dim = dim // (2 * axes)
        omega = 2 * torch.pi / (base ** (torch.arange(freq_dim) / freq_dim))
        self.register_buffer("omega", omega)

    def forward(self, coords: Tensor) -> Tensor:
        assert coords.shape[-1] == self.axes  # coords: (N, n_axes)
        ang = coords.unsqueeze(-1) * self.omega  # (N, n_axes, freq_dim)
        enc = rearrange(
            torch.stack([ang.sin(), ang.cos()], dim=-1),  # (N, n_axes, freq_dim, 2)
            "n ax d sc -> n (ax d sc)",  # axis-blocked, sin before cos
        )
        return enc


class RoPE(nn.Module):
    """N-D Rotary Positional Encoding."""

    def __init__(self, axes: int, dim: int, base: float = 10_000.0) -> None:
        super().__init__()
        if dim % (2 * axes) != 0:
            raise ValueError(f"dim must be divisible by {2 * axes}, got {dim}")
        self.axes = axes
        freq_dim = dim // (2 * axes)
        inv_freq = 1.0 / (base ** (torch.arange(freq_dim).float() / freq_dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, coords: Tensor) -> tuple[Tensor, Tensor]:
        assert coords.shape[-1] == self.axes  # coords: (N, n_axes)
        ang = coords.unsqueeze(-1) * self.inv_freq  # (N, axes, freq_dim)
        # (N, freq_dim*axes), axis-major interleave
        half = rearrange(ang, "n ax d -> n (d ax)")
        cos = repeat(half.cos(), "n d -> n (r d)", r=2)
        sin = repeat(half.sin(), "n d -> n (r d)", r=2)
        return cos, sin
