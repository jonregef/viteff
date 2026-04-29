"""https://facebookresearch.github.io/spdl/main/generated/imagenet_classification.html"""

# from spdl.dataloader import DataLoader  # TODO
from functools import partial
from typing import cast

from torch import nn, Tensor
from torchvision.io import decode_image, ImageReadMode
from torchvision.transforms import v2 as T
from torchvision.transforms.v2 import functional as TF
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
        torch.frombuffer(bytearray(img_bytes), dtype=torch.uint8), ImageReadMode.RGB
    )
    img_tensor = cast(Tensor, transform(img_tensor)).contiguous()
    return img_tensor, int(label)


def collate(batch: list[tuple[Tensor, int]]):
    images, labels = zip(*batch)
    return list(images), torch.tensor(labels, dtype=torch.int64)


class CudaPrefetcher:
    """Async copy of variable-size image batches to GPU."""

    def __init__(
        self,
        loader: torch.utils.data.DataLoader,
        device: torch.device = torch.device("cuda"),
    ) -> None:
        self.loader = loader
        self.device = device
        self.stream = torch.cuda.Stream(device)

    def __iter__(self):
        next_images, next_labels = None, None

        def preload(images, labels):
            with torch.cuda.stream(self.stream):
                return (
                    [
                        img.pin_memory().to(self.device, non_blocking=True)
                        for img in images
                    ],
                    labels.pin_memory().to(self.device, non_blocking=True),
                )

        it = iter(self.loader)
        try:
            images, labels = next(it)
            next_images, next_labels = preload(images, labels)
        except StopIteration:
            return

        for images, labels in it:
            # Start transferring current batch while we prep next
            torch.cuda.current_stream(self.device).wait_stream(self.stream)
            cur_images, cur_labels = next_images, next_labels
            next_images, next_labels = preload(images, labels)
            yield cur_images, cur_labels

        # Last batch
        torch.cuda.current_stream(self.device).wait_stream(self.stream)
        yield next_images, next_labels

    def __len__(self):
        return len(self.loader)


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
    return CudaPrefetcher(
        torch.utils.data.DataLoader(
            dataset,  # type:ignore
            batch_size=batch_size,
            collate_fn=collate,
            drop_last=train,
            pin_memory=False,  # FIXME: pinning with variable-size images OOMs
            num_workers=threads,
            prefetch_factor=4,
            persistent_workers=threads > 0,
            multiprocessing_context="forkserver",
        )
    )
