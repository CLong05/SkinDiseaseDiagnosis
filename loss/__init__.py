import torch
import torch.nn as nn

from .triplet_loss import TripletLoss


class Loss(nn.Module):
    def __init__(self, weight=1.0, margin=0.3):
        super().__init__()
        assert(weight >=0 and weight <= 1)
        self.weight = weight
        self.loss_fn_1 = nn.CrossEntropyLoss()
        self.loss_fn_2 = TripletLoss(margin)

    def forward(self, scores, features, labels):
        '''
        features: 一个batch的图像特征
        scores: 全连层的输出, 用于求交叉熵loss
        labels: 标签
        '''
        loss1 = self.loss_fn_1(scores, labels)
        loss2, _, _ = self.loss_fn_2(features, labels)

        loss = (1 - self.weight) * loss1 + self.weight * loss2
        return loss


if __name__ == '__main__':
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    features = torch.rand(6, 10)
    loss_fn = Loss(0.5)
    loss_fn(features, features, labels)
