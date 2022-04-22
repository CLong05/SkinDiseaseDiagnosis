import time
import torch
import argparse
import torch.nn as nn
import torch.optim as optim

from dataloader import make_dataloader
from models import make_model
from train import trainer
from test import test
from utils.logger import Logger
from utils.set_seed import setup_seed

setup_seed(0)

class CONFIG:
    def __init__(self):
        # dataset
        self.DATASET_PATH = '~/Documents/dataset/Skin40'
        self.SIZE = (224, 224)
        # model
        self.MODEL = 'resnet18'
        # train
        self.DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.LR = 0.001
        self.TRAIN_BATCH_SIZE = 32
        self.TEST_BATCH_SIZE = 128
        self.WEIGHT_DECAY = 1e-4
        self.EPOCH_NUM = 50
        self.EVALUATE_INTERVAL = self.EPOCH_NUM / 10
        self.SAVE_INTERVAL = self.EPOCH_NUM
        # log
        filename = time.strftime("%Y-%m-%d_%H:%M", time.localtime())
        self.LOG_DIR = f'logs/{filename}'

    def get_all_configs(self):
        return vars(self).items()


def train(cfg, logger):
    dataloaders = make_dataloader(
        cfg.DATASET_PATH,
        cfg.SIZE,
        cfg.TRAIN_BATCH_SIZE, cfg.TEST_BATCH_SIZE)
    for fold, (train_loader, val_loader) in enumerate(dataloaders):
        fold += 1
        model = make_model(cfg.MODEL).to(cfg.DEVICE)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)

        logger.info(f'fold{fold} start trainning')
        for epoch in range(cfg.EPOCH_NUM):
            loss, train_accuracy = trainer(model, train_loader, loss_fn, optimizer, cfg.DEVICE)

            logger.save_train_data(epoch, fold, loss, train_accuracy)
            logger.info(
                'fold[%d/5] epoch[%d/%d] loss: %f train accuracy: %f' % (
                fold, epoch + 1, cfg.EPOCH_NUM, loss, train_accuracy))

            if (epoch + 1) % cfg.EVALUATE_INTERVAL == 0:
                val_accuracy = test(model, val_loader, cfg.DEVICE)
                logger.save_evaluate_data(epoch, fold, val_accuracy)
                logger.info('val accuracy: %f' % (val_accuracy))

            if (epoch + 1) % cfg.SAVE_INTERVAL == 0:
                torch.save(model.state_dict(), cfg.LOG_DIR + f'/model_fold{fold}_{epoch}.pth')
    # finished trainning


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--log',
        action='store',
        required=True,
        choices=['true', 'false'],
        help='是否记录log')
    args = parser.parse_args()

    cfg = CONFIG()
    logger = Logger('train_log',cfg.LOG_DIR if args.log == 'true' else None, cfg.EPOCH_NUM)
    logger.info('------------------configs------------------')
    for name, value in cfg.get_all_configs():
        logger.info(f'{name}: {value}')
    logger.info('--------------start trainning--------------')
    train(cfg, logger)
    logger.info('finished trainning')
