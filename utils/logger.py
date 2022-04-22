import logging
import pandas as pd
import torch
import os
import sys
import os.path as osp


class Logger:
    def __init__(self, name, save_dir, epoch_num):
        self.save_dir = save_dir

        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)

        formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        if save_dir:
            if not osp.exists(save_dir):
                os.makedirs(save_dir)
            fh = logging.FileHandler(os.path.join(save_dir, "train_log.txt"), mode='w')
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)
        if save_dir:
            self.train_data = pd.DataFrame(  # 保存训练数据，用于绘制曲线
                torch.zeros(epoch_num, 7).numpy(),
                columns=['loss', 'train_accuracy', 'fold1', 'fold2', 'fold3', 'fold4', 'fold5'])

    def info(self, string):
        self.logger.info(string)

    def train_data(self, epoch, loss, accuracy):
        if self.save_dir:
            self.train_data.loc[epoch]['loss'] = loss
            self.train_data.loc[epoch]['train_accuracy'] = accuracy
    
    def evaluate_data(self, epoch, fold, accuracy):
        if self.save_dir:
            self.train_data.loc[epoch][f'fold{fold}']=accuracy
            self.train_data.to_csv(self.save_dir + f'/train_data_{epoch}.csv')
