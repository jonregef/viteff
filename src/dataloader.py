from torch import nn, Tensor
from torch.profiler import profile, ProfilerActivity, record_function
from torchvision.io import decode_image, ImageReadMode
from torchvision.transforms import v2 as T
from torchvision.transforms.v2 import functional as TF
import torch


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
    def __init__(self, loader: torch.utils.data.DataLoader, augs: nn.Module) -> None:
        self.loader = loader
        self.augs = augs.to("cuda")  # FIXME: can't compile due to randomness
        self.streams = (torch.cuda.Stream("cuda"), torch.cuda.Stream("cuda"))

    @record_function("augmentation")
    def _load(self, images: list[Tensor], labels: Tensor, stream: torch.cuda.Stream):
        with torch.cuda.stream(stream):
            # Augment on this stream — overlaps with compute on default stream
            # keep in uint8, the model performs casting and division internally
            imgs = [
                self.augs(img.pin_memory().to("cuda", non_blocking=True))
                for img in images
            ]
            lbls = labels.to("cuda", non_blocking=True)
        return imgs, lbls, stream

    def __iter__(self):
        compute = torch.cuda.current_stream()
        it = iter(self.loader)
        try:
            (images, labels) = next(it)
        except StopIteration:
            return

        prefetched = self._load(images, labels, self.streams[0])
        for i, (images, labels) in enumerate(it):
            next_stream = self.streams[(i + 1) % 2]
            next_prefetched = self._load(images, labels, next_stream)
            compute.wait_stream(prefetched[2])
            yield prefetched[0], prefetched[1]
            prefetched = next_prefetched

        compute.wait_stream(prefetched[2])
        yield prefetched[0], prefetched[1]

    def __len__(self):
        return len(self.loader)


val_augs = nn.Identity()
train_augs = torch.jit.script(
    torch.nn.Sequential(
        T.RandomHorizontalFlip(p=0.5),
        T.RandomGrayscale(p=0.1),
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.05),
    )
)


def collate(
    batch: list[tuple[Tensor, int]], patch_size: int = 16
) -> tuple[list[Tensor], Tensor]:
    images, labels = zip(*batch)
    return list(images), torch.tensor(labels, dtype=torch.int64)


def build_webdataset_dataloader(
    paths: list[str],
    train: bool,
    batch_size: int,
    augs: nn.Module,
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

    loader = torch.utils.data.DataLoader(
        dataset,  # type: ignore
        batch_size=batch_size,
        collate_fn=collate,
        drop_last=train,
        pin_memory=False,  # we pin manually in CudaPrefetcher
        num_workers=threads,
        prefetch_factor=4,
        persistent_workers=train and threads > 0,
        multiprocessing_context="spawn",
    )
    return CudaPrefetcher(loader, augs=augs)
