from torchvision import  models


def make_model():
    '''
    cache dir = /Users/lurenjie/.cache/torch/hub/checkpoints/resnet18-5c106cde.pth
    '''
    resnet18 = models.resnet18(pretrained=True, progress=True)
    for name, param in resnet18.named_parameters():
        if 'fc' not in name:
            param.requires_grad = False
    return resnet18


if __name__ == '__main__':
    net = make_model()
    for layer in net.named_parameters():
        print(layer[0])
