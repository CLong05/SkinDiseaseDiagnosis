from torch.utils.data import Sampler


class RandomClassSampler(Sampler):
    '''
    在不同类别均匀的采样
    '''
    def __init__(self, train_idx, num_instances=4):
        self.train_sample_idx = list()
        interval = 48 // num_instances  # 每个类别48个样本，每个类别采样num_instances个
        for i in range(interval):  # 假设每个类别按顺序排好
            self.train_sample_idx += list(train_idx[i::interval])

    def __iter__(self):
        return iter(self.train_sample_idx)

    def __len__(self):
        return self.length
