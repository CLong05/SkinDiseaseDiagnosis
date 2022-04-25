import numpy as np
from torch.utils.data import Sampler


class RandomClassSampler(Sampler):
    '''
    在不同类别均匀的采样
    '''
    def __init__(self, train_idx, num_instances=4):
        self.interval = 48 // num_instances  # 每个类别48个样本，每个类别采样num_instances个
        self.train_idx = train_idx.numpy().reshape(40, 48)

    def __iter__(self):
        np.random.shuffle(self.train_idx)  # 打乱类别的顺序
        for i in range(40):  # 打乱每个类别内的顺序
            np.random.shuffle(self.train_idx[i])
        train_sample_idx = list()
        for i in range(self.interval):
            for c in range(40):  # 每个类别采样num_instances个
                train_sample_idx += list(self.train_idx[c][i::self.interval])
        return iter(train_sample_idx)

    def __len__(self):
        return self.length
