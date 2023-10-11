import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import ImageFolder


from dataloader import *
from models import make_model


def find_wrong():
    dataloaders = make_dataloader(
        '~/Documents/dataset/Skin40', (224, 224), 32, 80)
    for fold, (_, val_loader) in enumerate(dataloaders):
        fold += 1
        model = make_model('resnet101')
        state_dict = torch.load(
            './logs/model_fold1_59.pth', map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)

        for ids, imgs, labels in val_loader:
            _, prediction = model(imgs)
            prediction = prediction.argmax(dim=1)
            wrong = prediction != labels
            print('wrong_id', ids[wrong])
            print('pred', prediction[wrong])
        exit()


def predict(model, img):
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])])
    img = val_transform(img).unsqueeze(0)
    _, predict = model(img)
    return predict.argmax(dim=1).item()


def main():
    wrong_id = [700, 1000,  380, 1795, 1260,  370,  450, 1435, 1380,  435,  915, 1995,
        1755, 1690, 1180,   15,  270, 2110,  245,  835,  260, 1395,  390,  460,
        2345, 1965, 1355,  795, 1515, 1745, 1290, 2145, 2170, 2200, 2030,  660, 1250,  445, 1925, 1460,
        1470, 1320,  725,  105,  410,  415,   10,  805, 1210, 1440, 1235,  265,  825, 2180, 1195,  255,  365,  355, 1130, 1680, 1330,  275,
        1735,  715, 1375, 1705,  360, 1475,  615,  535, 1245, 2070, 1975, 1340,
         935,  495, 1650,  810, 285, 1420, 1620,  720,  910, 1450,  905, 1505, 1425,  820, 1030,  900,
          65, 1165, 1345, 1555,  785, 1570,  490,  520, 2060, 1185, 1125,   75,
        1155, 530, 1930,  250, 1225,  510, 1045, 1940, 1090,  240,  925, 1365,  170,
         500,  480, 1640,  280, 1720,  290, 1635,  920,  295,  505, 540,  860, 1805, 1020, 1115, 1025, 1715,  870,  840,  515, 1190,  625,
         420,  395, 1955,  575, 1255, 70, 1665]
    pred = [6, 37,  1, 20, 12, 11, 20,  0, 34,  6, 30, 32, 28, 22, 14,  4, 36, 24,
        17,  8, 28, 39, 26,  8, 24, 31, 34, 20, 6,  8, 10,  4,  4, 28, 31, 20,  6, 28, 31, 12, 38,  0, 37,  8,  1, 25,
        28,  6, 13, 38, 9, 34,  9, 34, 13, 28, 20, 36, 14, 36, 34, 14, 17,  6,  6,  5, 27, 12,
        26, 20, 13, 28, 31,  0, 11, 20, 26, 20, 34,  0, 28, 37, 23,  4, 18,  4, 39, 20, 26,  1, 10, 13, 34, 11, 20, 36,
        20, 20,  0,  8, 10, 12, 20, 19, 31, 36,  7, 20, 24, 31,  6, 21,  7,  0, 17, 20, 13, 15,  0, 17,  6,
        26, 28, 14, 20, 13, 37,  6,  4, 15, 38,  5, 31, 37, 16, 20,  1, 19, 20, 38, 19, 17, 29,
        26]
    print('错误个数:', len(wrong_id), len(pred))
    assert(len(pred) == len(wrong_id))

    model = make_model('resnet101')
    state_dict = torch.load(
        './logs/model_fold1_59.pth', map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    dataset = ImageFolder('~/Documents/dataset/Skin40', transform=None)
    for i, idx in enumerate(wrong_id):
        img, label = dataset[idx]
        plt.imshow(img)
        plt.title(f'true label: {label}, wrong label: {pred[i]}')
        plt.show()
        continue
        w, h = img.size
        img = img.crop([0.3 * w, 0.3 * h, 0.6 * w, 0.6 * h])
        plt.imshow(img)
        plt.title(f'true label: {label}, pred label: {predict(model, img)}')
        plt.show()


if __name__ == '__main__':
    main()
    # find_wrong()
