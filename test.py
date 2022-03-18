import torch

def test(model, test_loader, device=torch.device('cpu')):
    model.eval()
    correct_num, total_num = 0, 0
    for imgs, labels in test_loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        prediction = model(imgs)

        batch_size = imgs.shape[0]
        correct_num += (prediction.argmax(dim=1) == labels).sum()
        total_num += batch_size
    return correct_num.cpu().detach().numpy() / total_num
