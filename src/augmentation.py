from typing import Literal

import torch
from torchvision.transforms import InterpolationMode, v2 as T


def three_augment(magnitude: float = 0.5) -> T.Transform:
    """https://arxiv.org/abs/2204.07118"""

    m = max(0.0, min(1.0, magnitude))
    if m == 0.0:
        return T.Identity()
    return T.Compose(
        [
            T.RandomApply(
                [
                    T.RandomChoice(
                        [
                            T.Grayscale(num_output_channels=3),
                            T.RandomSolarize(threshold=round(256 * (1 - m)), p=1.0),
                            T.GaussianBlur(5, sigma=(0.1, max(0.11, 4.0 * m))),
                        ]
                    )
                ],
                p=min(1.0, 2.0 * m),
            ),
            T.ColorJitter(brightness=0.6 * m, contrast=0.6 * m, saturation=0.6 * m),
        ]
    )


def rand_augment(magnitude: float = 0.5) -> T.Transform:
    """https://arxiv.org/abs/1909.13719"""

    m = max(0.0, min(1.0, magnitude))
    if m == 0.0:
        return T.Identity()
    return T.RandAugment(
        num_ops=2,
        magnitude=max(1, round(m * 18)),
        interpolation=InterpolationMode.BILINEAR,
    )


def trivial_augment(magnitude: float = 0.5) -> T.Transform:
    """https://arxiv.org/abs/2103.10158"""

    m = max(0.0, min(1.0, magnitude))
    if m == 0.0:
        return T.Identity()
    aug = T.TrivialAugmentWide()
    # sliding magnitude window (0.5: full range)
    nb = aug.num_magnitude_bins
    lo = min(nb - 1, round(nb * (max(0.0, 2 * m - 1))))
    hi = max(lo + 1, round(nb * (min(1.0, 2 * m))))

    def window(fn):
        def windowed(num_bins: int, h: int, w: int):
            full = fn(num_bins, h, w)
            if full is None:
                return None
            return torch.linspace(float(full[lo]), float(full[hi - 1]), num_bins)

        return windowed

    aug._AUGMENTATION_SPACE = {
        op: (window(fn), signed) for op, (fn, signed) in aug._AUGMENTATION_SPACE.items()
    }
    return aug


def build_augmentation(
    name: Literal["threeaug", "randaug", "trivialaug"] = "trivialaug",
    magnitude: float = 0.5,
    with_flip: bool = True,
) -> T.Transform:
    augs: list[T.Transform] = [T.RandomHorizontalFlip()] if with_flip else []
    match name:
        case "threeaug":
            augs.append(three_augment(magnitude))
        case "randaug":
            augs.append(rand_augment(magnitude))
        case "trivialaug":
            augs.append(trivial_augment(magnitude))
    return T.Compose(augs)
