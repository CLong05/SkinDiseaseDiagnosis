import torch
import torch.nn as nn


def trainer(model, teachers, train_loader, loss_fn, optimizer, device=torch.device('cpu'), T=1):
    model.train()
    correct_num, total_num = 0, 0
    
    for imgs, labels in train_loader:   
        optimizer.zero_grad()

        imgs = imgs.to(device)
        labels = labels.to(device)

        softmax = nn.Softmax(dim=1)
        class2teacher = [0, 0, 0, 0, 2, 0, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 2, 1, 0, 2, 1, 1, 0, 2, 0, 2, 1, 1, 1, 2, 0]
        features, prediction = model(imgs)
        with torch.no_grad():
            teacher_prediction = torch.zeros_like(prediction)
            teacher_predictions = [None] * 3
            _, teacher_predictions[0] = teachers[0](imgs)
            _, teacher_predictions[1] = teachers[1](imgs)
            _, teacher_predictions[2] = teachers[2](imgs)
            for i, label in enumerate(labels):
                teacher_prediction[i] = teacher_predictions[class2teacher[label]][i] 
            teacher_prediction = softmax(teacher_prediction / T)
        loss1 = loss_fn(prediction, features, labels)
        loss2 = loss_fn(prediction, features, teacher_prediction)
        weight = 0.9
        print('CE: ', (1 - weight) * loss1.item(), 'distill: ', weight * loss2.item())
        loss = (1 - weight) * loss1 + weight * loss2
    
        loss.backward()
        optimizer.step()

        batch_size = imgs.shape[0]
        correct_num += (prediction.argmax(dim=1) == labels).sum()
        total_num += batch_size
    return loss.item(), correct_num.cpu().detach().numpy() / total_num
