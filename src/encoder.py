import torch.nn as nn
import torchvision.models as models

class ResNetEncoder(nn.Module):
    def __init__(self, out_dim: int | None = 128, detach_head=False):
        super().__init__()
        if not detach_head and not out_dim:
            raise ValueError("If in detached is not detached you must have an output dimension")

        self._is_head_detached = detach_head
        self._backbone = models.resnet50()
        self._backbone.fc = nn.Identity()
        self._head = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, out_dim),
        )

    def forward(self, x):
        x = self._backbone(x)
        if not self._is_head_detached:
            x = self._head(x)
        return x