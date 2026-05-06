from torchvision.transforms import v2 as T

ThreeAugment = T.Compose(
    [
        T.RandomChoice(
            [
                T.Grayscale(num_output_channels=3),
                T.RandomSolarize(threshold=128, p=1.0),
                T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
            ]
        ),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    ]
)
