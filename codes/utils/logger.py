import os
import sys
from utils.util import get_timestamp


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
    def __init__(self, opt):
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
            tb_logger_dir = self.log_dir.replace('results', 'logs')
            if not os.path.isdir(tb_logger_dir):
                os.mkdir(tb_logger_dir)
            existing_dirs = sorted([dir for dir in os.listdir(tb_logger_dir) if os.path.isdir(os.path.join(tb_logger_dir,dir))],key=lambda x:int(x))
            if len(existing_dirs)>0:
                logger_dir_num = int(existing_dirs[-1])+1
            self.tb_logger = TensorboardLogger(os.path.join(tb_logger_dir,str(logger_dir_num)))

    def print_format_results(self, mode, rlt,dont_print=False):
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
            if label in ['l_d_real','l_d_fake','D_real','D_fake','psnr_val','LR_decrease','Correctly_distinguished','l_g_range','D_loss_STD','l_g_pix']:
                continue
            if mode == 'train':
                message += '{:s}: {:.2e} '.format(label, value)
            elif mode == 'val':
                message += '{:s}: {:.4e} '.format(label, value)
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
