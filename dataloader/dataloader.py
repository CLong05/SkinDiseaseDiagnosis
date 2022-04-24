import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, SubsetRandomSampler

from .sampler import RandomClassSampler
from .dataset import MapDataset

def get_5fold_indices(total_len):
    indices = torch.arange(0, total_len, dtype=torch.long)
    folds = []
    for fold in range(5):
        val_idx = indices[indices % 5 == fold]  # 每五个采样一个作为验证集样本
        train_idx = indices[indices % 5 != fold]
        folds.append([train_idx, val_idx])
    return folds


def make_dataloader(path, size, train_batch_size, test_batch_size):
    origin_dataset = ImageFolder(path, transform=None)
    dataloaders = list()
    for fold, (train_idx, val_idx) in enumerate(get_5fold_indices(len(origin_dataset))):
        dataset = MapDataset(origin_dataset, size, fold)
        train_sampler = RandomClassSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        train_loader = DataLoader(dataset, train_batch_size, sampler=train_sampler, num_workers=2)
        val_loader= DataLoader(dataset, test_batch_size, sampler=val_sampler, num_workers=2)
        dataloaders.append([train_loader, val_loader])
    return dataloaders
