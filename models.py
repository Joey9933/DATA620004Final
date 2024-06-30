import torch.nn as nn
from torchvision import models

class SimCLR(nn.Module):
    def __init__(self, out_dim):
        super(SimCLR, self).__init__()
        self.backbone = models.resnet18(weights=None, num_classes=out_dim)
        dim_features = self.backbone.fc.in_features  # 512 for ResNet-18

        # add mlp projection head
        self.backbone.fc = nn.Sequential(
            nn.Linear(dim_features, dim_features),
            nn.ReLU(),
            self.backbone.fc,
        )
    def forward(self, x):
        return self.backbone(x)

class ResNet(nn.Module):

    def __init__(self, out_dim) -> None:
        super().__init__()
        self.backbone = models.resnet18(weights=None, num_classes=out_dim)

    def forward(self, x):
        return self.backbone(x)