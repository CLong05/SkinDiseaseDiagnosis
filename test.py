import torch
'''
@torch.no_grad()
def test(model, test_loader, device=torch.device('cpu')):
    model.eval()
    correct_num, total_num = 0, 0
    recall = torch.zeros(40)  # 统计每个类别准确率, 测试集中每个类别12个样本
    for imgs, labels in test_loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        _, prediction = model(imgs)

        batch_size = imgs.shape[0]
        prediction = prediction.argmax(dim=1)
        correct_num += (prediction == labels).sum()
        for i in range(batch_size):
            if prediction[i] == labels[i]:
                recall[labels[i].item()] += 1
        total_num += batch_size
    return correct_num.item() / total_num, recall / 12
'''

@torch.no_grad()
def test(model, test_loader, device=torch.device('cpu')):
    model.eval()
    correct_num, total_num = 0, 0
    confusion_matrix = torch.zeros(40, 40)
    for imgs, labels in test_loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        _, prediction = model(imgs)

        batch_size = imgs.shape[0]
        prediction = prediction.argmax(dim=1)
        correct_num += (prediction == labels).sum()
        for i in range(batch_size):
            confusion_matrix[labels[i].item()][prediction[i].item()] += 1
        total_num += batch_size
    return correct_num.item(), total_num, confusion_matrix

