import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, SubsetRandomSampler


def get_transform(size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(size),
        transforms.Pad(10, 10),
        transforms.CenterCrop(size),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])])
    return transform


def get_indices():
    '''
    每个类别前10个图片组成测试集, 其余的作为训练集
    '''
    indices = torch.arange(0, 2400, dtype=torch.int)
    test_idices = indices[indices % 60 < 10]
    train_idices = indices[indices % 60 >= 10]
    train_idices = train_idices[torch.randperm(len(train_idices))]
    return train_idices, test_idices


def get_5fold_indices(total_len):
    indices = torch.arange(0, total_len, dtype=torch.long)
    folds = []
    for i in range(5):
        val_idx = indices[indices % 5 == i]  # 每五个采样一个作为验证集样本
        train_idx = indices[indices % 5 != i]
        folds.append([train_idx, val_idx])
    return folds


def make_dataloader(path, size, train_batch_size, test_batch_size):
    '''
    一共2400张图片, 400张是训练集, 400张测试集, 1600张训练集
    '''
    dataset = ImageFolder(path, transform=get_transform(size))
    train_indices, test_indices = get_indices()
    test_sampler = SubsetRandomSampler(test_indices)
    test_loader = DataLoader(dataset, test_batch_size, sampler=test_sampler)

    dataloaders = list()
    for train_idx, val_idx in get_5fold_indices(len(train_indices)):
        val_sampler = SubsetRandomSampler(train_indices[train_idx])
        train_sampler = SubsetRandomSampler(train_indices[val_idx])
        val_loader= DataLoader(dataset, train_batch_size, sampler=val_sampler)
        train_loader = DataLoader(dataset, train_batch_size, sampler=train_sampler)
        dataloaders.append([train_loader, val_loader, test_loader])
    return dataloaders
