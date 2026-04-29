"""https://facebookresearch.github.io/spdl/main/generated/imagenet_classification.html"""

# from spdl.dataloader import DataLoader  # TODO
# import spdl.io
from functools import partial

from torch import nn, Tensor
from torchvision.io import decode_image, ImageReadMode
from torchvision.transforms import v2 as T
import torchvision.transforms.v2.functional as TF
from webdataset.compat import WebDataset
from webdataset.handlers import warn_and_continue
import torch


class CapLongestEdge(nn.Module):
    def __init__(self, max_size: int) -> None:
        super().__init__()
        self.max_size = max_size

    def forward(self, img: Tensor) -> Tensor:
        _, h, w = img.shape
        s = min(self.max_size / max(h, w), 1.0)
        return TF.resize(img, [round(h * s), round(w * s)], antialias=True)


train_transform = T.Compose(
    [
        CapLongestEdge(max_size=1024),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomGrayscale(p=0.1),
        # T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=None),
    ]
)

val_transform = T.Compose([CapLongestEdge(max_size=1024)])


def decode(sample: tuple[bytes, bytes], transform: T.Transform) -> tuple[Tensor, int]:
    img_bytes, label = sample
    img_tensor = decode_image(
        torch.frombuffer(bytearray(img_bytes), dtype=torch.uint8),
        mode=ImageReadMode.RGB,
    )
    return transform(img_tensor), int(label)


def collate(batch: list[tuple[Tensor, int]]):
    images, labels = zip(*batch)
    return [img.contiguous() for img in images], torch.tensor(labels, dtype=torch.int64)


def build_imagenet_dataloader(
    paths: list[str],
    train: bool,
    batch_size: int,
    transform: T.Transform,
    threads: int = 8,
    seed: int = 42,
):
    dataset = WebDataset(
        paths,
        shardshuffle=256 if train else False,
        handler=warn_and_continue,
        detshuffle=True,
        seed=seed,
    )
    dataset = dataset.to_tuple("jpg", "cls").map(
        partial(decode, transform=transform), handler=warn_and_continue
    )

    if train:
        dataset = dataset.shuffle(1024, seed=seed)
    return torch.utils.data.DataLoader(
        dataset,  # type:ignore
        batch_size=batch_size,
        num_workers=threads,
        collate_fn=collate,
        pin_memory=True,
        prefetch_factor=4,
        drop_last=train,
        # webdataset handles shuffling internally; persistent workers
        # avoid re-initializing the shard iterator each epoch
        persistent_workers=threads > 0,
    )
