import torch

def trainer(model, train_loader, loss_fn, optimizer, device=torch.device('cpu')):
    model.train()
    correct_num, total_num = 0, 0
    for imgs, labels in train_loader:
        optimizer.zero_grad()

        imgs = imgs.to(device)
        labels = labels.to(device)

        prediction = model(imgs)
        loss = loss_fn(prediction, labels)
        loss.backward()
        optimizer.step()

        batch_size = imgs.shape[0]
        correct_num += (prediction.argmax(dim=1) == labels).sum()
        total_num += batch_size
    return loss.cpu().detach().numpy(), correct_num.cpu().detach().numpy() / total_num
