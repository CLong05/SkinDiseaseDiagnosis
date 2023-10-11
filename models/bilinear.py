import copy
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional


class BCNN(nn.Module):
    def __init__(self):
        super(BCNN, self).__init__()
        features = torchvision.models.resnet18(pretrained=True, progress=True)
        # Remove the pooling layer and full connection layer
        self.model1 = nn.Sequential(*list(features.children())[:-2])
        self.model2 = copy.deepcopy(self.model1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        feature1 = self.model1(x)
        feature2 = self.model2(x)
        feature1_ = self.avgpool(feature1).view(feature1.size(0), -1)
        feature2_ = self.avgpool(feature2).view(feature2.size(0), -1)
        # Cross product operation
        feature1 = feature1.view(feature1.size(0), 512, 7 * 7)
        feature2 = feature2.view(feature2.size(0), 512, 7 * 7)
        feature2_T = torch.transpose(feature2, 1, 2)
        features = torch.bmm(feature1, feature2_T) / (7 * 7)
        features = features.view(features.size(0), 512 * 512)
        # The signed square root
        features = torch.sign(features) * torch.sqrt(torch.abs(features) + 1e-12)
        # L2 regularization
        features = torch.nn.functional.normalize(features)
        return feature1_, feature2_, features

