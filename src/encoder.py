import torch.nn as nn
import torchvision.models as models

class ResNetEncoder(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.backbone = models.resnet50(num_classes=out_dim)
        dim_mlp = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.backbone.fc)

    def forward(self, x):
        return self.backbone(x)