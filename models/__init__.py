import torch.nn as nn

from torchvision.models import resnet18, resnet50
from .cnn import *


def make_model(model_name):
    '''
    model_name = [resnet18, resnet50, cnn1]
    cache dir = /Users/lurenjie/.cache/torch/hub/checkpoints
    '''
    if model_name == 'resnet18':
        model = resnet18(pretrained=True, progress=True)
        model.fc = nn.Linear(512, 40)
    elif model_name == 'resnet50':
        model = resnet50(pretrained=True, progress=True)
        model.fc = nn.Linear(2048, 40)
    elif model_name == 'cnn1':
        model = CNN()
    return model


if __name__ == '__main__':
    net = make_model('resnet18')
    print(net)
