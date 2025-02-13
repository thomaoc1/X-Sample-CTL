from abc import abstractmethod, ABC

import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import DataLoader


class ValidClrDataset(ABC):
    @abstractmethod
    def get_dataloader(self) -> DataLoader:
        pass

    @abstractmethod
    def get_gpu_augmentations(self) -> nn.Module | transforms.Compose:
        pass