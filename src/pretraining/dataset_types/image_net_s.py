from typing import Literal
from torchvision import transforms
from torchvision.transforms import autoaugment

from src.pretraining.dataset_types.image_folder_dataset import ImageFolderDataset


_VALID_ALG = Literal['simclr', 'xclr']
class ImageNetS(ImageFolderDataset):
    def __init__(
            self,
            image_folder_path: str,
            num_workers: int,
            batch_size: int,
            alg: _VALID_ALG,
    ):
        self._alg = alg
        self._resize = 224

        super().__init__(
            image_folder_path=image_folder_path,
            num_workers=num_workers,
            batch_size=batch_size,
            cpu_transform=self._associated_cpu_transform(),
            gpu_transform=self._associated_gpu_transform()

        )

    def _associated_cpu_transform(self):
        if self._alg == 'simclr':
            return transforms.Compose(
                [
                    transforms.PILToTensor(),
                    transforms.RandomResizedCrop(self._resize, antialias=True),
                ]
            )
        else:
            return transforms.Compose(
            [
                transforms.Resize((self._resize, self._resize)),
                transforms.PILToTensor(),
            ]
        )

    def _associated_gpu_transform(self):
        if self._alg == 'simclr':
            kernel_size = max(1, int(0.1 * self._resize))  # Ensure it's at least 1
            kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size  # Make it odd
            color_jitter = transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)

            return transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomApply([color_jitter], p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.GaussianBlur(kernel_size=kernel_size),
                    transforms.Lambda(lambd=lambda x: x / 255.0),
                ]
            )
        else:
            return transforms.Compose(
                [
                    autoaugment.AutoAugment(policy=autoaugment.AutoAugmentPolicy.IMAGENET),
                    transforms.Lambda(lambd=lambda x: x / 255.0),
                ]
            )