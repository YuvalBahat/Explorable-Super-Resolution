import os
import sys
from utils.util import get_timestamp
import numpy as np

# print to file and std_out simultaneously
class PrintLogger(object):
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log = open(os.path.join(log_path, 'print_log.txt'), 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


class Logger(object):
    def __init__(self, opt,tb_logger_suffix=''):
        self.exp_name = opt['name']
        self.use_tb_logger = opt['use_tb_logger']
        self.opt = opt['logger']
        self.log_dir = opt['path']['log']
        if not os.path.isdir(self.log_dir):
            os.mkdir(self.log_dir)
        # loss log file
        self.loss_log_path = os.path.join(self.log_dir, 'loss_log.txt')
        with open(self.loss_log_path, 'a') as log_file:
            log_file.write('=============== Time: ' + get_timestamp() + ' =============\n')
            log_file.write('================ Training Losses ================\n')
        # val results log file
        self.val_log_path = os.path.join(self.log_dir, 'val_log.txt')
        with open(self.val_log_path, 'a') as log_file:
            log_file.write('================ Time: ' + get_timestamp() + ' ===============\n')
            log_file.write('================ Validation Results ================\n')
        if self.use_tb_logger:# and 'debug' not in self.exp_name:
            from tensorboard_logger import Logger as TensorboardLogger
            logger_dir_num = 0
            tb_logger_dir = self.log_dir.replace('experiments', 'logs')
            if not os.path.isdir(tb_logger_dir):
                os.mkdir(tb_logger_dir)
            existing_dirs = sorted([dir.split('_')[0] for dir in os.listdir(tb_logger_dir) if os.path.isdir(os.path.join(tb_logger_dir,dir))],key=lambda x:int(x.split('_')[0]))
            if len(existing_dirs)>0:
                logger_dir_num = int(existing_dirs[-1])+1
            self.tb_logger = TensorboardLogger(os.path.join(tb_logger_dir,str(logger_dir_num)+tb_logger_suffix))

    def print_format_results(self, mode, rlt,dont_print=False,keys_ignore_list=[]):
        epoch = rlt.pop('epoch')
        iters = rlt.pop('iters')
        time = rlt.pop('time')
        model = rlt.pop('model')
        if 'lr' in rlt:
            lr = rlt.pop('lr')
            message = '<epoch:{:3d}, iter:{:8,d}, time:{:.2f}, lr:{:.1e}> '.format(
                epoch, iters, time, lr)
        else:
            message = '<epoch:{:3d}, iter:{:8,d}, time:{:.2f}> '.format(epoch, iters, time)

        for label, value in rlt.items():
            if label in keys_ignore_list or any([prefix in label for prefix in ['GT_','quantized_']]):
                continue
            # if mode == 'train':
            if np.abs(value)>1e-1:
                message += '{:s}: {:.3f} '.format(label, value)
            else:
                message += '{:s}: {:.4e} '.format(label, value)
            # elif mode == 'val':
            #     message += '{:s}: {:.4e} '.format(label, value)
            # tensorboard logger
            if self.use_tb_logger:# and 'debug' not in self.exp_name:
                self.tb_logger.log_value(label, value, iters)

        # print in console
        if not dont_print:
            print(message)
        # write in log file
        if mode == 'train':
            with open(self.loss_log_path, 'a') as log_file:
                log_file.write(message + '\n')
        elif mode == 'val':
            with open(self.val_log_path, 'a') as log_file:
                log_file.write(message + '\n')
