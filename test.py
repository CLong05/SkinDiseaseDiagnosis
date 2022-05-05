import torch

@torch.no_grad()
def test(model, test_loader, device=torch.device('cpu')):
    model.eval()
    correct_num, total_num = 0, 0
    recall = torch.zeros(40)
    for imgs, labels in test_loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        _, prediction = model(imgs)

        batch_size = imgs.shape[0]
        prediction = prediction.argmax(dim=1)
        correct_num += (prediction == labels).sum()
        for i in prediction.cpu():
            recall[i] += 1
        total_num += batch_size
    return correct_num.item() / total_num, recall / 12
