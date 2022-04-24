from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset


class MapDataset(Dataset):
    def __init__(self, dataset, size, fold):
        self.dataset = dataset
        self.fold = fold
        self.train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.Pad(14, 14),
            transforms.RandomResizedCrop(size, scale=(0.25, 1)),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])])
        self.val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])])

    def __getitem__(self, index):
        img, label = self.dataset[index]
        if index % 5 == self.fold:
            img = self.val_transform(img)
        else:
            img = self.train_transform(img)
        return img, label
    
    def __len__(self):
        return len(self.dataset)
