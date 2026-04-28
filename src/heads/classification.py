from torch import nn, Tensor
import torch

from ..patchifier import PatchifierOutput


class RegisterClassifier(nn.Module):
    def __init__(self, dim: int, num_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(dim, num_classes, bias=True)

    def forward(self, tokens: Tensor, metadata: PatchifierOutput) -> Tensor:
        # (B, dim), first token ([CLS] token) of each segment
        cls = tokens[metadata.cu_seqlens[:-1]]
        return self.fc(cls)


class AveragePoolingClassifier(nn.Module):
    def __init__(self, dim: int, num_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(dim, num_classes, bias=True)

    def forward(self, tokens: Tensor, metadata: PatchifierOutput) -> Tensor:
        device, dtype = tokens.device, tokens.dtype
        batch_size = metadata.cu_seqlens.shape[0] - 1
        seg_ids = torch.repeat_interleave(
            torch.arange(batch_size, device=device),
            (metadata.cu_seqlens[1:] - metadata.cu_seqlens[:-1]),
        )
        patch_tokens, patch_seg = tokens[metadata.is_patch], seg_ids[metadata.is_patch]
        summed = torch.zeros(batch_size, tokens.shape[-1], device=device, dtype=dtype)
        summed.index_add_(0, patch_seg, patch_tokens)
        counts = torch.zeros(batch_size, device=device, dtype=dtype)
        counts.index_add_(0, patch_seg, torch.ones_like(patch_seg, dtype=dtype))
        pooled = summed / counts.clamp_min(1).unsqueeze(-1)
        return self.fc(pooled)


class AttentionPoolingClassifier(nn.Module):
    def __init__(self, dim: int, num_classes: int) -> None:
        super().__init__()
        self.fc = nn.Linear(dim, num_classes, bias=True)
        self.query = nn.Parameter(torch.zeros(dim))
        nn.init.trunc_normal_(self.query, std=0.02)
        self.kv = nn.Linear(dim, 2 * dim, bias=False)
        self.scale = dim**-0.5

    def forward(self, tokens: Tensor, metadata: PatchifierOutput) -> Tensor:
        device, dtype = tokens.device, tokens.dtype
        batch_size = metadata.cu_seqlens.shape[0] - 1
        seg_ids = torch.repeat_interleave(
            torch.arange(batch_size, device=tokens.device),
            (metadata.cu_seqlens[1:] - metadata.cu_seqlens[:-1]),
        )
        patch_tokens = tokens[metadata.is_patch]
        patch_seg = seg_ids[metadata.is_patch]
        # attention — per-image softmax over patches against a single learned query
        k, v = self.kv(patch_tokens).chunk(2, dim=-1)
        logits = (k * self.query).sum(-1) * self.scale  # (sum_patches,)
        # segment-softmax: subtract per-segment max for stability
        max_per = torch.full((batch_size,), float("-inf"), device=device, dtype=dtype)
        max_per = max_per.scatter_reduce(
            0, patch_seg, logits, reduce="amax", include_self=True
        )
        ex = (logits - max_per[patch_seg]).exp()
        denom = torch.zeros(batch_size, device=device, dtype=dtype)
        denom.index_add_(0, patch_seg, ex)
        weights = ex / denom[patch_seg]
        pooled = torch.zeros(batch_size, tokens.shape[-1], device=device, dtype=dtype)
        pooled.index_add_(0, patch_seg, v * weights.unsqueeze(-1))
        return self.fc(pooled)
