import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, SubsetRandomSampler


def get_transform(size):
    transform = transforms.Compose([
        transforms.ToTensor(),  # 图片都是72 * 72的
        transforms.RandomHorizontalFlip(),
        transforms.Pad(14, 14),
        transforms.RandomResizedCrop(size, scale=(0.25, 1)),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])])
    return transform


def get_indices():
    '''
    每个类别前10个图片组成测试集, 其余的作为训练集
    '''
    indices = torch.arange(0, 2400, dtype=torch.int)
    test_indices = indices[indices % 60 < 10]
    train_indices = indices[indices % 60 >= 10]
    train_indices = train_indices[torch.randperm(len(train_indices))]
    return train_indices, test_indices


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

if __name__ == '__main__':
    path = '~/Documents/dataset/Skin40'
    size = (100, 100)
    train_batch_size = 160
    test_batch_size = 160
    dataloaders = make_dataloader(path, size, train_batch_size, test_batch_size)
    for train_loader, val_loader, test_loader in dataloaders:
        print('train_loader')
        for imgs, labels in train_loader:
            label_num = [0] * 40  # 检查每个batch的标签数目是否均衡
            for label in labels:
                label_num[label] += 1
            print(label_num)
        print('test_loader')
        label_num = [0] * 40  # 验证每个类别数量是否相同
        for imgs, labels in test_loader:
            for label in labels:
                label_num[label] += 1
        print(label_num)
