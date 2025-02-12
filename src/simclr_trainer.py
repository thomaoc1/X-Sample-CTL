from torchvision import transforms

from abstract_trainer import ClrTrainer


class SimClrTrainer(ClrTrainer):
    @staticmethod
    def image_augmentation_fn(size: int):
        kernel_size = max(1, int(0.1 * size))  # Ensure it's at least 1
        kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size  # Make it odd
        color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)

        return transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([color_jitter], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.GaussianBlur(kernel_size=kernel_size),
                transforms.Lambda(lambd=lambda x: x / 255.0)
            ]
        )

    def __init__(
            self,
            dataset_path: str,
            batch_size: int,
            device: str,
            encoder_checkpoint_path: str,
            head_out_features: int = 128,
            num_workers_dl: int = 4,
            epochs: int = 100,
            encoder_load_path: str | None = None,
            size: int = 224
    ):
        initial_transform = transforms.Compose(
            [
                transforms.PILToTensor(),
                transforms.RandomResizedCrop(size),
            ]
        )

        super().__init__(
            dataset_path=dataset_path,
            batch_size=batch_size,
            device=device,
            encoder_checkpoint_path=encoder_checkpoint_path,
            head_out_features=head_out_features,
            num_worker_dl=num_workers_dl,
            epochs=epochs,
            encoder_load_path=encoder_load_path,
            initial_transform=initial_transform,
            image_augmentation_transform=SimClrTrainer.image_augmentation_fn(size),
        )


