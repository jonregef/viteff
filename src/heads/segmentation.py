from torch import nn, Tensor

from ..patchifier import PatchifierOutput


class PlainMaskSegmenter(nn.Module):
    """work in progress! (https://arxiv.org/abs/2603.25398)"""

    def __init__(self, dim: int, num_classes: int) -> None:
        super().__init__()
        # TODO from official implementation: https://github.com/tue-mps/pmt

    def forward(self, tokens: Tensor, metadata: PatchifierOutput) -> Tensor:
        raise NotImplementedError()
