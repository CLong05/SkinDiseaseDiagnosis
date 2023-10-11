import time
import torch
import argparse
import os.path as osp
import torch.nn as nn
import torch.optim as optim

from dataloader import make_dataloader_subset, make_dataloader
from models import make_model
from train import trainer
from test import test
from utils.logger import Logger
from utils.set_seed import setup_seed
from loss import Loss


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
        self.DESCRIPTION = '使用前面训练的三个网络进行蒸馏'
        # dataset
        self.DATASET_PATH = '../Skin40'
        self.SIZE = (224, 224)
        # model
        self.MODEL = 'resnet101'
        # train
        self.DEVICE = torch.device('cuda:7') if torch.cuda.is_available() else torch.device('cpu')
        self.LR = 0.0001
        self.STEP_SIZE = 2
        self.TRAIN_BATCH_SIZE = 32
        self.TEST_BATCH_SIZE = 128
        self.WEIGHT_DECAY = 1e-3
        self.EPOCH_NUM = 60
        self.MARGIN = 0.3 
        self.WEIGHT = 0.0  # triplet loss的占比
        self.SMOOTH = 0.0
        self.T = 3
        self.EVALUATE_INTERVAL = self.EPOCH_NUM // 20 if self.EPOCH_NUM > 20 else 1
        # log
        filename = time.strftime("%Y-%m-%d_%H:%M", time.localtime())
        self.LOG_DIR = f'logs/{filename}'

    def get_all_configs(self):
        return vars(self).items()


class TempModel(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = models
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        p = []
        vote = torch.zeros(x.shape[0], 40).to(x.device)
        for idx, model in enumerate(self.models):
            s = self.softmax(model(x)[1])
            temp = s.argmax(dim=1)
            for i in range(x.shape[0]):
                vote[i, temp[i]] += 1
            p.append(s / s.max(dim=1).values.view(-1, 1))
            p.append(s)
        p = sum(p)
        for i in range(x.shape[0]):
            if vote[i, :].max() == 1:
                vote[i, :] = s[i, :]
        return None, vote


def train(cfg, logger):
    dataloaders = make_dataloader_subset(
        cfg.DATASET_PATH,
        cfg.SIZE,
        cfg.TRAIN_BATCH_SIZE, cfg.TEST_BATCH_SIZE)
    for fold, (_, val_loader) in enumerate(dataloaders):
        fold += 1
        # model = make_model(cfg.MODEL).to(cfg.DEVICE)

        teacher1 = make_model(cfg.MODEL).to(cfg.DEVICE)        
        teacher2 = make_model(cfg.MODEL).to(cfg.DEVICE)
        teacher3 = make_model(cfg.MODEL).to(cfg.DEVICE)
        teacher4 = make_model(cfg.MODEL).to(cfg.DEVICE)
        teacher5 = make_model('vit').to(cfg.DEVICE)
        teacher1.load_state_dict(torch.load(f'logs/2022-06-07_09:20/model_fold{fold}.pth'))
        teacher2.load_state_dict(torch.load(f'logs/2022-06-08_22:59/model_fold{fold}.pth'))
        teacher3.load_state_dict(torch.load(f'logs/2022-06-10_10:08/model_fold{fold}.pth'))
        teacher4.load_state_dict(torch.load(f'logs/2022-06-10_21:39/model_fold{fold}.pth'))
        teacher5.load_state_dict(torch.load(f'model_fold{fold}.pth'))
        teachers = [teacher1, teacher2, teacher3, teacher4, teacher5]
        model = TempModel(teachers)
        correct_num, total_num, _ = test(model, val_loader, cfg.DEVICE)
        print(f'fold:{fold}, {correct_num}/{total_num}({correct_num/total_num:.4f})')


if __name__ == '__main__':
    setup_seed(0)
    cfg = CONFIG()
    logger = Logger('train_log',cfg.LOG_DIR if args.log == 'true' else None, cfg.EPOCH_NUM)
    logger.info('------------------configs------------------')
    for name, value in cfg.get_all_configs():
        logger.info(f'{name}: {value}')
    logger.info('--------------start trainning--------------')
    train(cfg, logger)
    logger.info('finished trainning')
