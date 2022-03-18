import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim

from dataloader import make_dataloader
from model import make_model
from train import trainer
from test import test
from logger import make_logger
from utils.set_seed import setup_seed

setup_seed(0)

class CONFIG:
    DATASET_PATH = '/Users/lurenjie/Documents/dataset/Skin40'
    SIZE = (224, 224)
    # model
    # train
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    LR = 0.0001
    TRAIN_BATCH_SIZE = 32
    EPOCH_NUM = 100
    EVALUATE_INTERVAL = 10
    SAVE_INTERVAL = 100
    # test
    TEST_BATCH_SIZE = 100


cfg = CONFIG
logger = make_logger("train_log", "logs", train=True)
train_data = pd.DataFrame(  # 保存训练数据，用于绘制曲线
    torch.zeros(cfg.EPOCH_NUM, 7).numpy(),
    columns=['loss', 'train_accuracy', 'fold1', 'fold2', 'fold3', 'fold4', 'fold5'])


def train():
    dataloaders = make_dataloader(
        cfg.DATASET_PATH,
        cfg.SIZE,
        cfg.TRAIN_BATCH_SIZE, cfg.TEST_BATCH_SIZE)
    for fold, (train_loader, val_loader, test_loader) in enumerate(dataloaders):
        fold += 1
        model = make_model().to(cfg.DEVICE)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.fc.parameters(), lr=cfg.LR)

        logger.info(f'fold{fold} start trainning')
        for epoch in range(cfg.EPOCH_NUM):
            loss, train_accuracy = trainer(model, train_loader, loss_fn, optimizer, cfg.DEVICE)

            train_data.loc[epoch]['loss'] = loss
            train_data.loc[epoch]['train_accuracy'] = train_accuracy

            logger.info(
                'fold[%d/5] epoch[%d/%d] loss: %f train accuracy: %f' % (
                fold, epoch + 1, cfg.EPOCH_NUM, loss, train_accuracy))

            if (epoch + 1) % cfg.EVALUATE_INTERVAL == 0:
                val_accuracy = test(model, val_loader, cfg.DEVICE)
                train_data.loc[epoch][f'fold{fold}']=val_accuracy
                logger.info('val accuracy: %f' % (val_accuracy))

            if (epoch + 1) % cfg.SAVE_INTERVAL == 0:
                torch.save(model.state_dict(), f'logs/model_fold{fold}_{epoch}.pth')
                train_data.to_csv(f'train_data_{epoch}.csv')
        # 测试该fold下的模型准确率
        test_accuracy = test(model, test_loader, cfg.DEVICE)
        logger.info('-------------------------')
        logger.info('test accuracy: %f' % (test_accuracy))
    # finished trainning
    logger.info('finished trainning')


if __name__ == '__main__':
    train()
