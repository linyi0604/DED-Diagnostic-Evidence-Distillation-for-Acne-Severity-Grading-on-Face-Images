import torch
from torch import nn
import torchvision.models as models

class VGG11(nn.Module):
    def __init__(self, class_num, pretrained=True):
        super(VGG11, self).__init__()
        vgg11 = models.vgg11(pretrained=pretrained)
        self.feature = vgg11.features
        self.pool = vgg11.avgpool
        self.classifier = vgg11.classifier
        self.classifier[6] = nn.Linear(4096, class_num)

    def forward(self, x):
        x = self.feature(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class Vgg11_PK(nn.Module):
    def __init__(self, input_channel, class_num, pretrained=True):
        super(Vgg11_PK, self).__init__()

        self.pre_process = nn.Sequential(
            nn.Conv2d(input_channel, 3, (1, 1)),
            nn.ReLU(),
        )

        vgg11 = models.vgg11(pretrained=pretrained)
        self.feature = vgg11.features
        self.pool = vgg11.avgpool
        self.classifier = vgg11.classifier
        self.classifier[6] = nn.Linear(4096, class_num)

    def forward(self, x):
        x = self.pre_process(x)
        x = self.feature(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class VGG13(nn.Module):
    def __init__(self, class_num, pretrained=True):
        super(VGG13, self).__init__()
        vgg13 = models.vgg13(pretrained=pretrained)
        self.feature = vgg13.features
        self.pool = vgg13.avgpool
        self.classifier = vgg13.classifier
        self.classifier[6] = nn.Linear(4096, class_num)

    def forward(self, x):
        x = self.feature(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x



class Vgg13_PK(nn.Module):
    def __init__(self, input_channel, class_num, pretrained=True):
        super(Vgg13_PK, self).__init__()

        self.pre_process = nn.Sequential(
            nn.Conv2d(input_channel, 3, (1, 1)),
            nn.ReLU(),
        )

        vgg13 = models.vgg13(pretrained=pretrained)
        self.feature = vgg13.features
        self.pool = vgg13.avgpool
        self.classifier = vgg13.classifier
        self.classifier[6] = nn.Linear(4096, class_num)

    def forward(self, x):
        x = self.pre_process(x)
        x = self.feature(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x




class VGG16(nn.Module):
    def __init__(self, class_num, pretrained=True):
        super(VGG16, self).__init__()
        vgg16 = models.vgg16(pretrained=pretrained)
        self.feature = vgg16.features
        self.pool = vgg16.avgpool
        self.classifier = vgg16.classifier
        self.classifier[6] = nn.Linear(4096, class_num)

    def forward(self, x):
        x = self.feature(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x



class VGG16_image_mask(nn.Module):
    def __init__(self, input_channel, class_num, pretrained=True):
        super(VGG16_image_mask, self).__init__()
        vgg16 = models.vgg16(pretrained=pretrained)
        self.pre_process = nn.Sequential(
            nn.Conv2d(input_channel, 3, (1, 1)),
            nn.ReLU(),
        )
        self.feature = vgg16.features
        self.pool = vgg16.avgpool
        self.classifier = vgg16.classifier
        self.classifier[6] = nn.Linear(4096, class_num)

    def forward(self, x):
        x = self.pre_process(x)
        x = self.feature(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x



class VGG16_image_mask_to_level_couting_regression(nn.Module):
    def __init__(self, input_channel, class_num, max_distribution_num, pretrained=True):
        super(VGG16_image_mask_to_level_couting_regression, self).__init__()
        vgg16 = models.vgg16(pretrained=pretrained)
        self.pre_process = nn.Sequential(
            nn.Conv2d(input_channel, 3, (1, 1)),
            nn.ReLU(),
        )
        self.feature = vgg16.features
        self.pool = vgg16.avgpool
        self.linear = vgg16.classifier[:-2]
        self.classifier = nn.Linear(4096, class_num)
        self.regressor = nn.Linear(4096, max_distribution_num)

    def forward(self, x):
        x = self.pre_process(x)
        x = self.feature(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        classification = self.classifier(x)
        regression = self.regressor(x)
        return classification, regression


class Teacher(nn.Module):
    def __init__(self, input_channel, class_num, max_distribution_num, pretrained=True):
        super(Teacher, self).__init__()
        vgg16 = models.vgg16(pretrained=pretrained)
        self.pre_process = nn.Sequential(
            nn.Conv2d(input_channel, 3, (1, 1)),
            nn.ReLU(),
        )
        self.feature = vgg16.features
        self.pool = vgg16.avgpool
        self.linear = vgg16.classifier[:-2]
        self.classifier = nn.Linear(4096, class_num)
        self.regressor = nn.Linear(4096, max_distribution_num)

    def forward(self, x):
        with torch.no_grad():
            x = self.pre_process(x)
            x = self.feature(x)
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            distribution = self.linear(x)
            classification = self.classifier(distribution)
            regression = self.regressor(distribution)
        return distribution, classification, regression


class Student(nn.Module):
    def __init__(self, class_num, pretrained=True):
        super(Student, self).__init__()
        vgg16 = models.vgg16(pretrained=pretrained)

        self.feature = vgg16.features
        self.pool = vgg16.avgpool
        self.linear = vgg16.classifier[:-2]
        self.classifier = nn.Linear(4096, class_num)

    def forward(self, x):
        x = self.feature(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        distribution = self.linear(x)
        classification = self.classifier(distribution)
        return distribution, classification



