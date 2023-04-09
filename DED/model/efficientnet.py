

import torch
from torch import nn
import torchvision.models as models


class EfficientNet_b0(nn.Module):
    def __init__(self, class_num, pretrained=True):
        super(EfficientNet_b0, self).__init__()
        efficientnet_b0 = models.efficientnet_b0(pretrained=pretrained)
        self.feature = efficientnet_b0.features
        self.pool = efficientnet_b0.avgpool
        self.classifier = efficientnet_b0.classifier
        self.classifier[1] = nn.Linear(1280, class_num)

    def forward(self, x):
        x = self.feature(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x



class EfficientNet_b0_PK(nn.Module):
    def __init__(self, input_channel, class_num, pretrained=True):
        super(EfficientNet_b0_PK, self).__init__()
        self.pre_process = nn.Sequential(
            nn.Conv2d(input_channel, 3, (1, 1)),
            nn.ReLU(),
        )
        efficientnet_b0 = models.efficientnet_b0(pretrained=pretrained)
        self.feature = efficientnet_b0.features
        self.pool = efficientnet_b0.avgpool
        self.classifier = efficientnet_b0.classifier
        self.classifier[1] = nn.Linear(1280, class_num)

    def forward(self, x):
        x = self.pre_process(x)
        x = self.feature(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
