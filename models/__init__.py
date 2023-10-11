import torch
import torch.nn as nn

from torchvision.models import *

from .vit import VisionTransformer


class CNN(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.model_name = backbone
        if backbone == 'resnet18':
            self.backbone = resnet18(pretrained=True, progress=True)
            self.backbone.fc = nn.Identity()
            self.fc = nn.Linear(512, 40)
        elif backbone == 'resnet50':
            self.backbone = resnet50(pretrained=True, progress=True)
            self.backbone.fc = nn.Identity()
            self.fc = nn.Linear(2048, 40)
        elif backbone == 'resnet101':
            self.backbone = resnet101(pretrained=True, progress=True)
            self.backbone.fc = nn.Identity()
            self.fc = nn.Linear(2048, 40)
        elif backbone == 'vit':
            self.backbone = VisionTransformer()
            state_dict = torch.load('/Users/lurenjie/Documents/pretrained/vit_base_p16_224.pth')
            self.backbone.load_state_dict(state_dict)
            self.fc = nn.Linear(768, 40)
        else:
            raise RuntimeError('未定义模型')
    
    def forward(self, x):
        if self.model_name == 'vit':
            features = self.backbone.forward_features()
            features = features[:, 0, :]  # select class token
        else:
            features = self.backbone(x)
        score = self.fc(features)
        return features, score

def make_model(model_name):
    '''
    model_name = [resnet18, resnet50]
    cache dir = /Users/lurenjie/.cache/torch/hub/checkpoints
    '''
    model = CNN(model_name)
    return model


if __name__ == '__main__':
    net = make_model('resnet18')
    print(net)

