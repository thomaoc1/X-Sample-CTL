import torch.nn as nn
from torchvision.models import resnet50
import torch.utils.checkpoint as checkpoint

class ImageEncoder(nn.Module):
    def __init__(self, out_features):
        super().__init__()

        self._backbone = nn.Sequential(*list(resnet50().children())[:-1])
        self._head = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, out_features),
        )

    def forward(self, x):
        x = checkpoint.checkpoint(self._backbone, x, use_reentrant=False)
        x = x.flatten(1)
        return self._head(x)
