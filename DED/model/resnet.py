

import torch
from torch import nn
import torchvision.models as models



class ResNet18(nn.Module):
    def __init__(self, class_num, pretrained=True):
        super(ResNet18, self).__init__()
        resnet18 = models.resnet18(pretrained=pretrained)
        self.conv1 = resnet18.conv1
        self.bn1 = resnet18.bn1
        self.relu = resnet18.relu
        self.maxpool = resnet18.maxpool
        self.layer1 = resnet18.layer1
        self.layer2 = resnet18.layer2
        self.layer3 = resnet18.layer3
        self.layer4 = resnet18.layer4
        self.avgpool = resnet18.avgpool
        self.fc = nn.Linear(512, class_num)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x



class Res18_PK(nn.Module):
    def __init__(self, input_channel, class_num, pretrained=True):
        super(Res18_PK, self).__init__()
        resnet18 = models.resnet18(pretrained=pretrained)
        self.pre_process = nn.Sequential(
            nn.Conv2d(input_channel, 3, (1, 1)),
            nn.ReLU(),
        )

        self.conv1 = resnet18.conv1
        self.bn1 = resnet18.bn1
        self.relu = resnet18.relu
        self.maxpool = resnet18.maxpool
        self.layer1 = resnet18.layer1
        self.layer2 = resnet18.layer2
        self.layer3 = resnet18.layer3
        self.layer4 = resnet18.layer4
        self.avgpool = resnet18.avgpool
        self.fc = nn.Linear(512, class_num)

    def forward(self, x):
        x = self.pre_process(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class ResNet34(nn.Module):
    def __init__(self, class_num, pretrained=True):
        super(ResNet34, self).__init__()
        resnet34 = models.resnet34(pretrained=pretrained)
        self.conv1 = resnet34.conv1
        self.bn1 = resnet34.bn1
        self.relu = resnet34.relu
        self.maxpool = resnet34.maxpool
        self.layer1 = resnet34.layer1
        self.layer2 = resnet34.layer2
        self.layer3 = resnet34.layer3
        self.layer4 = resnet34.layer4
        self.avgpool = resnet34.avgpool
        self.fc = nn.Linear(512, class_num)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class Res34_PK(nn.Module):
    def __init__(self, input_channel, class_num, pretrained=True):
        super(Res34_PK, self).__init__()
        self.pre_process = nn.Sequential(
            nn.Conv2d(input_channel, 3, (1, 1)),
            nn.ReLU(),
        )

        resnet34 = models.resnet34(pretrained=pretrained)
        self.conv1 = resnet34.conv1
        self.bn1 = resnet34.bn1
        self.relu = resnet34.relu
        self.maxpool = resnet34.maxpool
        self.layer1 = resnet34.layer1
        self.layer2 = resnet34.layer2
        self.layer3 = resnet34.layer3
        self.layer4 = resnet34.layer4
        self.avgpool = resnet34.avgpool
        self.fc = nn.Linear(512, class_num)

    def forward(self, x):
        x = self.pre_process(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class ResNet50(nn.Module):
    def __init__(self, class_num, pretrained=True):
        super(ResNet50, self).__init__()
        resnet50 = models.resnet50(pretrained=pretrained)
        self.conv1 = resnet50.conv1
        self.bn1 = resnet50.bn1
        self.relu = resnet50.relu
        self.maxpool = resnet50.maxpool
        self.layer1 = resnet50.layer1
        self.layer2 = resnet50.layer2
        self.layer3 = resnet50.layer3
        self.layer4 = resnet50.layer4
        self.avgpool = resnet50.avgpool
        self.fc = nn.Linear(2048, class_num)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
