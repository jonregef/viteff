from bisect import bisect_right
from typing import Callable

from spdl.pipeline.defs import Aggregator


class BatchSizeCurriculum(Aggregator):
    def __init__(
        self,
        batch_sizes: list[int] | int,
        milestones: list[int] | None,
        now: Callable[[], int],
    ) -> None:
        if isinstance(batch_sizes, int):
            self.sizes, self.milestones = [batch_sizes], [0]
        else:
            assert milestones is not None
            self.sizes, self.milestones = batch_sizes, milestones
        self.now = now
        self._buffer: list = []

    def at(self, idx: int) -> int:
        i = bisect_right(self.milestones, idx) - 1
        return self.sizes[max(i, 0)]

    def accumulate(self, item):
        self._buffer.append(item)
        if len(self._buffer) >= self.at(self.now()):
            batch, self._buffer = self._buffer, []
            return batch
        return None

    def flush(self):
        if self._buffer:
            batch, self._buffer = self._buffer, []
            return batch
        return None
