import torch
from torch import nn
from torchvision.models import resnet50

class res50(nn.Module):
    def __init__(self, hidden_dim=256):
            super().__init__()
            self.backbone = resnet50()
            del self.backbone.fc

            # create conversion layer
            self.conv = nn.Conv2d(2048, hidden_dim, 1)
    def forward(self, inputs):
        # propagate inputs through ResNet-50 up to avg-pool layer
        x = self.backbone.conv1(inputs)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        # convert from 2048 to 256 feature planes for the transformer
        h = self.conv(x)
        return h #B,C,W,H