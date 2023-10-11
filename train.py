import torch
import numpy as np


# CutMix的参数
beta = 0  # 1
cutmix_prob = 0.5


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def trainer(model, train_loader, loss_fn, optimizer, device=torch.device('cpu')):
    model.train()
    correct_num, total_num = 0, 0
    
    # CutMix
    for i, (imgs, labels) in enumerate(train_loader):   
        optimizer.zero_grad()

        imgs = imgs.to(device)
        labels = labels.to(device)

        r = np.random.rand(1)
        if beta > 0 and r < cutmix_prob:
            # generate mixed sample
            lam = np.random.beta(beta, beta)  # 通过lam决定裁剪叠加块的大小，并在后面计算loss时作为权重
            rand_index = torch.randperm(imgs.size()[0]).to(device)
            target_a = labels
            target_b = labels[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(imgs.size(), lam)
            imgs[:, :, bbx1:bbx2, bby1:bby2] = imgs[rand_index, :, bbx1:bbx2, bby1:bby2]  # 进行裁剪替换操作
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (imgs.size()[-1] * imgs.size()[-2]))
            # compute output
            features, prediction = model(imgs)
            loss = loss_fn(prediction, features, target_a) * lam + loss_fn(prediction, features, target_b) * (1. - lam)  # 以lam作为权重
        else:
            # compute output
            features, prediction = model(imgs)
            loss = loss_fn(prediction, features, labels)
    
        loss.backward()
        optimizer.step()

        batch_size = imgs.shape[0]
        correct_num += (prediction.argmax(dim=1) == labels).sum()
        total_num += batch_size
    return loss.cpu().detach().numpy(), correct_num.cpu().detach().numpy() / total_num
