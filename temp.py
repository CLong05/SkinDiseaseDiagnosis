from dataloader import *
import matplotlib.pyplot as plt

if __name__ == '__main__':
    dataloaders = make_dataloader('~/Documents/dataset/Skin40', (224, 224), 32, 128)
    for train_loader, _ in dataloaders:
        for imgs, labels in train_loader:
            for i in range(len(imgs)):
                plt.imshow(imgs[i].permute(1, 2, 0).numpy())
                plt.title(labels[i])
                plt.show()
