import torch
import torch.nn as nn

from .triplet_loss import TripletLoss


class Loss(nn.Module):
    def __init__(self, weight=0.5, margin=0.3, smooth=0.1):
        super().__init__()
        assert(weight >=0 and weight <= 1)
        self.weight = weight
        self.smooth = smooth
        self.loss_fn_1 = nn.CrossEntropyLoss()
        self.loss_fn_2 = TripletLoss(margin, 0.0)

    def forward(self, scores, features, labels):
        '''
        features: 一个batch的图像特征
        scores: 全连层的输出, 用于求交叉熵loss
        labels: 标签
        '''
        # label smooth
        smooth_label = torch.zeros_like(scores).to(labels.device)
        smooth_label.fill_(self.smooth / 39)
        smooth_label.scatter_(1, labels.unsqueeze(1), 1 - self.smooth)

        loss1 = self.loss_fn_1(scores, labels)
        if self.weight != 0:
            loss2, _, _ = self.loss_fn_2(features, labels)
            print(loss1.item(), loss2.item())
            loss = (1 - self.weight) * loss1 + self.weight * loss2
        else:
            loss = loss1
        return loss
