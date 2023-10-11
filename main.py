import time
import torch
import argparse
import torch.optim as optim

from dataloader import make_dataloader
from models import make_model
from train import trainer
from test import test
from utils.logger import Logger
from utils.set_seed import setup_seed
from loss import Loss

setup_seed(0)


parser = argparse.ArgumentParser()
parser.add_argument(
    '--log',
    action='store',
    required=True,
    choices=['true', 'false'],
    help='是否记录log')
args = parser.parse_args()


class CONFIG:
    def __init__(self):
        # description
        self.DESCRIPTION = '试试vit'
        # dataset
        self.DATASET_PATH = '~/origin_skin40'
        self.SIZE = (224, 224)
        # model
        self.MODEL = 'vit'
        # train
        self.DEVICE = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
        self.LR = 0.0001
        self.STEP_SIZE = 2
        self.TRAIN_BATCH_SIZE = 32
        self.TEST_BATCH_SIZE = 128
        self.WEIGHT_DECAY = 1e-3
        self.EPOCH_NUM = 60
        self.MARGIN = 0.3 
        self.WEIGHT = 0.0  # triplet loss的占比
        self.SMOOTH = 0.0 
        self.EVALUATE_INTERVAL = self.EPOCH_NUM // 20
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
        loss_fn = Loss(cfg.WEIGHT, cfg.MARGIN, cfg.SMOOTH)
        optimizer = optim.Adam(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, cfg.STEP_SIZE, gamma=0.8)

        logger.info(f'-----------------fold{fold} start trainning-----------------')
        best_accuracy, best_epoch = 0.7, -1
        for epoch in range(cfg.EPOCH_NUM):
            loss, train_accuracy = trainer(model, train_loader, loss_fn, optimizer, cfg.DEVICE)

            logger.save_train_data(epoch, fold, loss, train_accuracy)
            logger.info(
                'fold[%d/5] epoch[%d/%d] loss: %f train accuracy: %f, lr:%e' % (
                fold, epoch + 1, cfg.EPOCH_NUM, loss, train_accuracy, optimizer.param_groups[0]['lr']))

            if (epoch + 1) % cfg.EVALUATE_INTERVAL == 0:
                correct_num, total_num, confusion_matrix = test(model, val_loader, cfg.DEVICE)
                val_accuracy = correct_num / total_num
                logger.save_evaluate_data(epoch, fold, val_accuracy, confusion_matrix)
                logger.info(f'val accuracy: {val_accuracy:.4f}, correct_num: {correct_num}, total_num: {total_num}')
                if args.log == 'true' and val_accuracy > best_accuracy:
                    torch.save(model.state_dict(), cfg.LOG_DIR + f'/model_fold{fold}.pth')
                    torch.save(confusion_matrix, cfg.LOG_DIR + f'/confusion_matrix{fold}.pth')
                if val_accuracy > best_accuracy:
                    best_accuracy = val_accuracy
                    best_epoch = epoch
            lr_scheduler.step()
        logger.info(f'best_epoch: {best_epoch}, best_accuracy: {best_accuracy}')
    # finished trainning


if __name__ == '__main__':

    cfg = CONFIG()
    logger = Logger('train_log',cfg.LOG_DIR if args.log == 'true' else None, cfg.EPOCH_NUM)
    logger.info('------------------configs------------------')
    for name, value in cfg.get_all_configs():
        logger.info(f'{name}: {value}')
    logger.info('--------------start trainning--------------')
    train(cfg, logger)
    logger.info('finished trainning')
