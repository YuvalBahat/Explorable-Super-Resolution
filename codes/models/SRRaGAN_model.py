import os
from collections import OrderedDict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import re
import models.networks as networks
from .base_model import BaseModel
from models.modules.loss import GANLoss, GradientPenaltyLoss,CreateRangeLoss
from torch.nn import Upsample
import DTE.DTEnet as DTEnet
import numpy as np


class SRRaGANModel(BaseModel):
    def __init__(self, opt):
        super(SRRaGANModel, self).__init__(opt)
        train_opt = opt['train']
        self.log_path = opt['path']['log']
        # define networks and load pretrained models
        self.DTE_net = None
        self.DTE_arch = opt['network_G']['DTE_arch']
        if self.DTE_arch or opt['train']['DTE_exp']:
            assert self.opt['train']['pixel_domain']=='HR' or not self.DTE_arch,'Why should I use DTE_arch AND penalize MSE in the LR domain?'
            self.DTE_net = DTEnet.DTEnet(DTEnet.Get_DTE_Conf(opt['scale']))
            if not self.DTE_arch:
                self.DTE_net.WrapArchitecture_PyTorch(only_padders=True)
        self.netG = networks.define_G(opt,DTE=self.DTE_net).to(self.device)  # G
        self.DTE_arch = opt['network_G']['DTE_arch']
        logs_2_keep = ['l_g_pix', 'l_g_fea', 'l_g_range', 'l_g_gan', 'l_d_real', 'l_d_fake', 'D_real', 'D_fake','D_logits_diff','psnr_val']
        self.log_dict = OrderedDict(zip(logs_2_keep, [[] for i in logs_2_keep]))

        if self.is_train:
            self.netD = networks.define_D(opt,DTE=self.DTE_net).to(self.device)  # D
            self.netG.train()
            self.netD.train()
        self.load()  # load G and D if needed

        # define losses, optimizer and scheduler
        if self.is_train:
            # G pixel loss
            if train_opt['pixel_weight'] > 0:
                l_pix_type = train_opt['pixel_criterion']
                if l_pix_type == 'l1':
                    self.cri_pix = nn.L1Loss().to(self.device)
                elif l_pix_type == 'l2':
                    self.cri_pix = nn.MSELoss().to(self.device)
                else:
                    raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_pix_type))
                self.l_pix_w = train_opt['pixel_weight']
            else:
                print('Remove pixel loss.')
                self.cri_pix = None

            # G feature loss
            if train_opt['feature_weight'] > 0:
                l_fea_type = train_opt['feature_criterion']
                if l_fea_type == 'l1':
                    self.cri_fea = nn.L1Loss().to(self.device)
                elif l_fea_type == 'l2':
                    self.cri_fea = nn.MSELoss().to(self.device)
                else:
                    raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_fea_type))
                self.l_fea_w = train_opt['feature_weight']
            else:
                print('Remove feature loss.')
                self.cri_fea = None
            if self.cri_fea:  # load VGG perceptual loss
                self.netF = networks.define_F(opt, use_bn=False).to(self.device)

            # Range limiting loss:
            if train_opt['range_weight'] > 0:
                self.cri_range = CreateRangeLoss(opt['range'])
                self.l_range_w = train_opt['range_weight']
            else:
                print('Remove range loss.')
                self.cri_range = None

            # GD gan loss
            self.cri_gan = GANLoss(train_opt['gan_type'], 1.0, 0.0).to(self.device)
            self.l_gan_w = train_opt['gan_weight']
            # D_update_ratio and D_init_iters are for WGAN
            self.D_update_ratio = train_opt['D_update_ratio'] if train_opt['D_update_ratio'] else 1
            self.D_init_iters = train_opt['D_init_iters'] if train_opt['D_init_iters'] else 0

            if train_opt['gan_type'] == 'wgan-gp':
                self.random_pt = torch.Tensor(1, 1, 1, 1).to(self.device)
                # gradient penalty loss
                self.cri_gp = GradientPenaltyLoss(device=self.device).to(self.device)
                self.l_gp_w = train_opt['gp_weigth']

            # optimizers
            # G
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            optim_params = []
            for k, v in self.netG.named_parameters():  # can optimize for a part of the model
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    print('WARNING: params [{:s}] will not optimize.'.format(k))
            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'], \
                weight_decay=wd_G, betas=(train_opt['beta1_G'], 0.999))
            self.optimizers.append(self.optimizer_G)
            # D
            wd_D = train_opt['weight_decay_D'] if train_opt['weight_decay_D'] else 0
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=train_opt['lr_D'], \
                weight_decay=wd_D, betas=(train_opt['beta1_D'], 0.999))
            self.optimizers.append(self.optimizer_D)

            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(lr_scheduler.MultiStepLR(optimizer, \
                        train_opt['lr_steps'], train_opt['lr_gamma']))
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

        print('---------- Model initialized ------------------')
        self.print_network()
        print('-----------------------------------------------')

    def feed_data(self, data, need_HR=True):
        # LR
        self.var_L = data['LR'].to(self.device)

        if need_HR:  # train or val
            self.var_H = data['HR'].to(self.device)

            input_ref = data['ref'] if 'ref' in data else data['HR']
            self.var_ref = input_ref.to(self.device)
    def Convert_2_LR(self,size):
        return Upsample(size=size,mode='bilinear')
    def optimize_parameters(self, step):
        # G
        for p in self.netD.parameters():
            p.requires_grad = False

        self.optimizer_G.zero_grad()

        self.fake_H = self.netG(self.var_L)
        if self.DTE_net is not None:
            # self.fake_H,self.var_H = self.DTE_net.Mask_Invalid_Regions_PyTorch(self.fake_H,self.var_H)
            self.fake_H,self.var_H,self.var_ref = self.DTE_net.HR_unpadder(self.fake_H),self.DTE_net.HR_unpadder(self.var_H),self.DTE_net.HR_unpadder(self.var_ref)
        l_g_total = 0#torch.zeros(size=[],requires_grad=True).type(torch.cuda.FloatTensor)
        if (step) % self.D_update_ratio == 0 and step > self.D_init_iters:
            if self.cri_pix:  # pixel loss
                if 'pixel_domain' in self.opt['train'] and self.opt['train']['pixel_domain']=='LR':
                    LR_size = list(self.var_L.size()[-2:])
                    l_g_pix = self.l_pix_w * self.cri_pix(self.Convert_2_LR(LR_size)(self.fake_H), self.Convert_2_LR(LR_size)(self.var_H))
                else:
                    l_g_pix = self.l_pix_w * self.cri_pix(self.fake_H, self.var_H)
                l_g_total += l_g_pix
            if self.cri_fea:  # feature loss
                if 'feature_domain' in self.opt['train'] and self.opt['train']['feature_domain']=='LR':
                    LR_size = list(self.var_L.size()[-2:])
                    real_fea = self.netF(self.Convert_2_LR(LR_size)(self.var_H)).detach()
                    fake_fea = self.netF(self.Convert_2_LR(LR_size)(self.fake_H))
                else:
                    real_fea = self.netF(self.var_H).detach()
                    fake_fea = self.netF(self.fake_H)
                l_g_fea = self.l_fea_w * self.cri_fea(fake_fea, real_fea)
                l_g_total += l_g_fea
            if self.cri_range: #range loss
                l_g_range = self.l_range_w * self.cri_range(self.fake_H)
                l_g_total += l_g_range
            # G gan + cls loss
            pred_g_fake = self.netD(self.fake_H)
            pred_d_real = self.netD(self.var_ref).detach()

            l_g_gan = self.l_gan_w * (self.cri_gan(pred_d_real - torch.mean(pred_g_fake), False) +
                                      self.cri_gan(pred_g_fake - torch.mean(pred_d_real), True)) / 2
            l_g_total += l_g_gan

            l_g_total.backward()
            self.optimizer_G.step()

        # D
        for p in self.netD.parameters():
            p.requires_grad = True

        self.optimizer_D.zero_grad()
        l_d_total = 0
        pred_d_real = self.netD(self.var_ref)
        pred_d_fake = self.netD(self.fake_H.detach())  # detach to avoid BP to G
        l_d_real = self.cri_gan(pred_d_real - torch.mean(pred_d_fake), True)
        l_d_fake = self.cri_gan(pred_d_fake - torch.mean(pred_d_real), False)

        l_d_total = (l_d_real + l_d_fake) / 2

        if self.opt['train']['gan_type'] == 'wgan-gp':
            batch_size = self.var_ref.size(0)
            if self.random_pt.size(0) != batch_size:
                self.random_pt.resize_(batch_size, 1, 1, 1)
            self.random_pt.uniform_()  # Draw random interpolation points
            interp = self.random_pt * self.fake_H.detach() + (1 - self.random_pt) * self.var_ref
            interp.requires_grad = True
            interp_crit, _ = self.netD(interp)
            l_d_gp = self.l_gp_w * self.cri_gp(interp, interp_crit)  # maybe wrong in cls?
            l_d_total += l_d_gp

        l_d_total.backward()
        self.optimizer_D.step()

        # set log
        if step % self.D_update_ratio == 0 and step > self.D_init_iters:
            # G
            if self.cri_pix:
                self.log_dict['l_g_pix'].append(l_g_pix.item())
            if self.cri_fea:
                self.log_dict['l_g_fea'].append(l_g_fea.item())
            if self.cri_range:
                self.log_dict['l_g_range'].append(l_g_range.item())
            self.log_dict['l_g_gan'].append(l_g_gan.item())
        # D
        self.log_dict['l_d_real'].append(l_d_real.item())
        self.log_dict['l_d_fake'].append(l_d_fake.item())

        if self.opt['train']['gan_type'] == 'wgan-gp':
            self.log_dict['l_d_gp'].append(l_d_gp.item())
        # D outputs
        self.log_dict['D_real'].append(torch.mean(pred_d_real.detach()))
        self.log_dict['D_fake'].append(torch.mean(pred_d_fake.detach()))
        self.log_dict['D_logits_diff'].append(torch.mean(pred_d_real.detach()-pred_d_fake.detach()))

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            if self.DTE_arch:
                self.fake_H = self.netG(self.var_L,pre_pad=True)
            else:
                self.fake_H = self.netG(self.var_L)
        self.netG.train()

    def get_current_log(self):
        dict_2_return = OrderedDict()
        for key in self.log_dict:
            if len(self.log_dict[key])>0 and not isinstance(self.log_dict[key][-1],tuple):
                dict_2_return[key] = self.log_dict[key][-1]
        return dict_2_return
    def save_log(self):
        np.savez(os.path.join(self.log_path,'logs.npz'), ** self.log_dict)
    def load_log(self):
        loaded_log = np.load(os.path.join(self.log_path,'logs.npz'))
        self.log_dict = OrderedDict()
        for key in loaded_log.files:
            if key=='psnr_val':
                self.log_dict[key] = [tuple(val) for val in loaded_log[key]]
            else:
                self.log_dict[key] = list(loaded_log[key])
    def display_log_figure(self):
        # keys_2_display = ['l_g_pix', 'l_g_fea', 'l_g_range', 'l_g_gan', 'l_d_real', 'l_d_fake', 'D_real', 'D_fake','D_logits_diff','psnr_val']
        keys_2_display = ['l_g_gan','D_logits_diff', 'psnr_val']
        fig = plt.figure()
        legend_strings = []
        for key in keys_2_display:
            if key in self.log_dict.keys() and len(self.log_dict[key])>0:
                if isinstance(self.log_dict[key][0],tuple):
                    plt.plot([val[0] for val in self.log_dict[key]],[val[1] for val in self.log_dict[key]])
                    series_avg = np.mean([val[1] for val in self.log_dict[key]])
                else:
                    plt.plot(self.log_dict[key])
                    if isinstance(self.log_dict[key][0],torch.Tensor):
                        series_avg = np.mean([val.data.cpu().numpy() for val in self.log_dict[key]])
                    else:
                        series_avg = np.mean(self.log_dict[key])
                legend_strings.append(key+' (%.2e)'%(series_avg))
        plt.legend(legend_strings,loc='best')
        plt.xlabel('Steps')
        plt.savefig(os.path.join(self.log_path,'logs.pdf'))
        plt.close(fig)


    def get_current_visuals(self, need_HR=True):
        out_dict = OrderedDict()
        out_dict['LR'] = self.var_L.detach()[0].float().cpu()
        out_dict['SR'] = self.fake_H.detach()[0].float().cpu()
        if need_HR:
            out_dict['HR'] = self.var_H.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        # Generator
        s, n = self.get_network_description(self.netG)
        print('Number of parameters in G: {:,d}'.format(n))
        if self.is_train:
            message = '-------------- Generator --------------\n' + s + '\n'
            network_path = os.path.join(self.save_dir, '../', 'network.txt')
            if not self.opt['train']['resume']:
                with open(network_path, 'w') as f:
                    f.write(message)

            # Discriminator
            s, n = self.get_network_description(self.netD)
            print('Number of parameters in D: {:,d}'.format(n))
            message = '\n\n\n-------------- Discriminator --------------\n' + s + '\n'
            if not self.opt['train']['resume']:
                with open(network_path, 'a') as f:
                    f.write(message)

            if self.cri_fea:  # F, Perceptual Network
                s, n = self.get_network_description(self.netF)
                print('Number of parameters in F: {:,d}'.format(n))
                message = '\n\n\n-------------- Perceptual Network --------------\n' + s + '\n'
                if not self.opt['train']['resume']:
                    with open(network_path, 'a') as f:
                        f.write(message)

    def load(self):
        resume_training = self.opt['train']['resume']
        load_path_G = self.opt['path']['pretrain_model_G']
        if resume_training is not None and resume_training:
            model_name = [name for name in os.listdir(self.opt['path']['models']) if '_G.pth' in name]
            model_name = sorted(model_name,key=lambda x: int(re.search('(\d)+(?=_G.pth)',x).group(0)))[-1]
            print('Resuming training with model for G [{:s}] ...'.format(os.path.join(self.opt['path']['models'],model_name)))
            self.load_network(os.path.join(self.opt['path']['models'],model_name), self.netG)
            self.load_log()
            if self.opt['is_train']:
                model_name = [name for name in os.listdir(self.opt['path']['models']) if '_D.pth' in name]
                model_name = sorted(model_name, key=lambda x: int(re.search('(\d)+(?=_D.pth)', x).group(0)))[-1]
                print('Resuming training with model for D [{:s}] ...'.format(os.path.join(self.opt['path']['models'],model_name)))
                self.load_network(os.path.join(self.opt['path']['models'],model_name), self.netD)

        else:
            if load_path_G is not None:
                print('loading model for G [{:s}] ...'.format(load_path_G))
                self.load_network(load_path_G, self.netG)
            load_path_D = self.opt['path']['pretrain_model_D']
            if self.opt['is_train'] and load_path_D is not None:
                print('loading model for D [{:s}] ...'.format(load_path_D))
                self.load_network(load_path_D, self.netD)

    def save(self, iter_label):
        self.save_network(self.save_dir, self.netG, 'G', iter_label)
        self.save_network(self.save_dir, self.netD, 'D', iter_label)
