from typing import Any
import warnings

from torch import nn, Tensor
from torch.profiler import record_function
from torchvision.io import decode_image, ImageReadMode
from torchvision.transforms import v2 as T
from torchvision.transforms.v2 import functional as TF
import torch

warnings.filterwarnings(
    "ignore",
    message=r".*The PyTorch API of nested tensors is in prototype stage.*",
    category=UserWarning,
)


@record_function("decoding")
def decode_and_cap(
    sample: tuple[bytes, bytes], max_size: int = 224
) -> tuple[Tensor, int]:
    jpg, cls = sample
    img = decode_image(
        torch.frombuffer(bytearray(jpg), dtype=torch.uint8), ImageReadMode.RGB
    )
    _, h, w = img.shape
    s = max_size / max(h, w)
    if s >= 1.0:
        return img, int(cls)
    return TF.resize(img, [round(h * s), round(w * s)], antialias=True), int(cls)


class CudaPrefetcher:
    def __init__(self, loader: torch.utils.data.DataLoader) -> None:
        self.loader = loader
        self.stream = torch.cuda.Stream("cuda")

    @record_function("host_to_device")
    def _load(self, nested: Tensor, targets: Tensor):
        with torch.cuda.stream(self.stream):
            nested = nested.to("cuda", non_blocking=True)
            targets = targets.to("cuda", non_blocking=True)
        return nested, targets

    def __iter__(self):
        compute = torch.cuda.current_stream()
        iterator = iter(self.loader)
        try:
            nested, targets = next(iterator)
        except StopIteration:
            return

        prefetched = self._load(nested, targets)
        for nested, labels in iterator:
            next_prefetched = self._load(nested, labels)
            compute.wait_stream(self.stream)
            yield prefetched[0].unbind(), prefetched[1]
            prefetched = next_prefetched
        compute.wait_stream(self.stream)
        yield prefetched[0].unbind(), prefetched[1]

    def __len__(self):
        return len(self.loader)


train_augs = torch.jit.script(
    torch.nn.Sequential(
        T.RandomHorizontalFlip(p=0.5),
        T.RandomGrayscale(p=0.1),
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.05),
    )
)


def collate(batch: list[tuple[Tensor, int]]) -> tuple[Tensor, Tensor]:
    images, labels = zip(*batch)
    nested = torch.nested.nested_tensor(
        list(images), layout=torch.strided, pin_memory=True
    )
    return nested, torch.tensor(labels, dtype=torch.int64)


class PicklableAugs:
    def __init__(self, augs: nn.Module) -> None:
        self.augs = augs

    def __call__(self, sample: tuple[Tensor, Any]) -> tuple[Tensor, Any]:
        image, target = sample
        return self.augs(image), target


def build_webdataset_dataloader(
    paths: list[str],
    train: bool,
    batch_size: int,
    augs: nn.Module | None,
    threads: int,
    seed: int = 42,
) -> CudaPrefetcher:
    from webdataset.compat import WebDataset
    from webdataset.handlers import warn_and_continue

    dataset = (
        WebDataset(
            paths,
            shardshuffle=256 if train else False,
            handler=warn_and_continue,
            detshuffle=True,
            seed=seed,
        )
        .shuffle(2048 if train else 0, seed=seed)
        .to_tuple("jpg", "cls")
        .map(decode_and_cap, handler=warn_and_continue)
    )
    if augs is not None:
        dataset = dataset.map(PicklableAugs(augs), handler=warn_and_continue)

    loader = torch.utils.data.DataLoader(
        dataset,  # type: ignore
        batch_size=batch_size,
        collate_fn=collate,
        drop_last=train,
        pin_memory=True,
        num_workers=threads,
        prefetch_factor=4,
        persistent_workers=train and threads > 0,
        multiprocessing_context="fork",
    )
    return CudaPrefetcher(loader)
