import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler


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
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        transforms.Pad(14, 14),
        transforms.RandomResizedCrop(size, scale=(0.25, 1)),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])])
    dataset = ImageFolder(path, transform=transform)
    dataloaders = list()
    for train_idx, val_idx in get_5fold_indices(len(dataset)):
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)
        train_loader = DataLoader(dataset, train_batch_size, sampler=train_sampler, num_workers=2)
        val_loader= DataLoader(dataset, test_batch_size, sampler=val_sampler, num_workers=2)
        dataloaders.append([train_loader, val_loader])
    return dataloaders

if __name__ == '__main__':
    path = '~/Documents/dataset/Skin40'
    size = (100, 100)
    train_batch_size = 32
    test_batch_size = 128
    dataloaders = make_dataloader(path, size, train_batch_size, test_batch_size)
    for train_loader, val_loader in dataloaders:
        print('train_loader')
        for imgs, labels in train_loader:
            label_num = [0] * 40  # 检查每个batch的标签数目是否均衡
            for label in labels:
                label_num[label] += 1
            print(label_num)
        print('val_loader')
        label_num = [0] * 40  # 验证每个类别数量是否相同
        for imgs, labels in val_loader:
            for label in labels:
                label_num[label] += 1
        print(label_num)
