import torch.nn as nn

from torchvision.models import resnet18, resnet50

class CNN(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        if backbone == 'resnet18':
            self.backbone = resnet18(pretrained=True, progress=True)
            self.backbone.fc = nn.Identity()
            self.fc = nn.Linear(512, 40)
        elif backbone == 'resnet50':
            self.backbone = resnet50(pretrained=True, progress=True)
            self.backbone.fc = nn.Identity()
            self.fc = nn.Linear(2048, 40)
        else:
            raise RuntimeError('未定义模型')
    
    def forward(self, x):
        feature = self.backbone(x)
        score = self.fc(feature)
        return feature, score


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
