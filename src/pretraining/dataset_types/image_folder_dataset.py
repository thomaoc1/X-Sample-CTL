from torch import nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms as transforms
from torchvision.datasets import ImageFolder

from src.pretraining.dataset_types.dataset import ValidClrDataset


class ImageFolderDataset(ValidClrDataset):

    def __init__(
            self,
            image_folder_path: str,
            cpu_transform: nn.Module | transforms.Compose,
            num_workers: int,
            batch_size: int,
            gpu_transform: nn.Module | transforms.Compose
    ):
        dataset = ImageFolder(
            root=image_folder_path,
            transform=cpu_transform,
        )

        self.label_range = len(dataset.classes)

        self._dataloader = DataLoader(
            dataset=dataset,
            num_workers=num_workers,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
        )

        self._gpu_transform = gpu_transform


    def get_gpu_augmentations(self) -> nn.Module | transforms.Compose:
        return self._gpu_transform

    def get_dataloader(self) -> DataLoader:
        return self._dataloader
