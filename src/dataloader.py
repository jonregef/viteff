from pathlib import Path
from typing import cast

from spdl.pipeline import Pipeline, PipelineBuilder
from spdl.pipeline.defs import Aggregator
from torch import nn, Tensor
from torchvision.transforms import v2 as T
from torchvision.transforms.v2 import functional as TF
import spdl.io
import torch
import webdataset as wds


class CapLongestEdge(nn.Module):
    def __init__(self, max_size: int) -> None:
        super().__init__()
        self.max_size = max_size

    def forward(self, img: Tensor) -> Tensor:
        _, h, w = img.shape
        m = max(h, w)
        if m <= self.max_size:
            return img
        s = self.max_size / m
        return TF.resize(img, [round(h * s), round(w * s)], antialias=True)


train_transform = T.Compose(
    [
        CapLongestEdge(max_size=1024),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.05),
        T.RandomGrayscale(p=0.1),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

val_transform = T.Compose(
    [
        CapLongestEdge(max_size=1024),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


_rgb = spdl.io.get_video_filter_desc(pix_fmt="rgb24")


def _decode_jpeg(jpg: bytes) -> Tensor:
    packets = spdl.io.demux_image(jpg)
    frames = spdl.io.decode_packets(packets, filter_desc=_rgb)
    tensor = spdl.io.to_torch(spdl.io.convert_frames(frames))
    return tensor.permute(2, 0, 1).contiguous()


def _make_decode(transform: T.Compose):
    def decode_one(item: tuple[bytes, int]) -> tuple[Tensor, int]:
        jpg, label = item
        return transform(_decode_jpeg(jpg)), int(label)

    return decode_one


def _collate(batch: list[tuple[Tensor, int]]) -> tuple[list[Tensor], Tensor]:
    # Pin in the SPDL thread so the training loop's .to(..., non_blocking=True) is truly async.
    # Keeping .to("cuda") out of this thread avoids forcing CUDA init on non-main threads.
    imgs, labels = zip(*batch)
    imgs = [cast(Tensor, img).pin_memory() for img in imgs]
    labels = torch.tensor(labels, dtype=torch.long).pin_memory()
    return imgs, labels


def build_pipeline(
    urls: list[str] | list[Path],
    batch_size: int | Aggregator,
    threads: int,
    transform: T.Transform,
    train: bool,
    seed: int,
    distributed: bool = False,
) -> Pipeline:
    dataset = wds.WebDataset(
        urls,
        shardshuffle=256 if train else False,
        nodesplitter=wds.split_by_node if distributed else None,
        handler=wds.warn_and_continue,
        detshuffle=True,
        seed=seed,
    )
    if train:
        dataset = dataset.shuffle(5000)
    pipeline = (
        PipelineBuilder()
        .add_source(iter(dataset.to_tuple("jpg", "cls")))
        .pipe(_make_decode(transform), concurrency=threads)
        .aggregate(batch_size)
        .pipe(_collate)
        .add_sink(4)
        .build(num_threads=threads)
    )
    return pipeline
