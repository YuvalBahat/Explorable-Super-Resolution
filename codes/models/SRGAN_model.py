import os
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.optim import lr_scheduler

import models.networks as networks
from .base_model import BaseModel
from models.modules.loss import GANLoss, GradientPenaltyLoss

import tqdm
from utils import util
import numpy as np
import re

class SRGANModel(BaseModel):
    def __init__(self, opt):
        super(SRGANModel, self).__init__(opt)
        train_opt = opt['train']

        # define networks and load pretrained models
        self.netG = networks.define_G(opt,num_latent_channels=0).to(self.device)  # G
        if self.is_train:
            self.netD = networks.define_D(opt).to(self.device)  # D
            self.netG.train()
            self.netD.train()
        self.step = 0
        self.gradient_step_num = self.step
        self.log_path = opt['path']['log']
        self.generator_changed = True  # Initializing to true,to save the initial state```````

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
                self.reshuffle_netF_weights = False
                if 'feature_pooling' in train_opt or 'feature_model_arch' in train_opt:
                    if 'feature_model_arch' not in train_opt:
                        train_opt['feature_model_arch'] = 'vgg19'
                    elif 'feature_pooling' not in train_opt:
                        train_opt['feature_pooling'] = ''
                    self.reshuffle_netF_weights = 'shuffled' in train_opt['feature_pooling']
                    train_opt['feature_pooling'] = train_opt['feature_pooling'].replace('untrained_shuffled_','untrained_').replace('untrained_shuffled','untrained')
                    self.netF = networks.define_F(opt, use_bn=False,
                            state_dict=torch.load(train_opt['netF_checkpoint'])['state_dict'] if 'netF_checkpoint' in train_opt else None,
                                arch=train_opt['feature_model_arch'],arch_config=train_opt['feature_pooling']).to(self.device)
                else:
                    self.netF = networks.define_F(opt, use_bn=False).to(self.device)

            # GD gan loss
            self.cri_gan = GANLoss(train_opt['gan_type'], 1.0, 0.0).to(self.device)
            self.D_exists = self.cri_gan is not None
            self.l_gan_w = train_opt['gan_weight']
            # D_update_ratio and D_init_iters are for WGAN
            self.D_update_ratio = train_opt['D_update_ratio'] if train_opt['D_update_ratio'] else 1
            self.D_init_iters = train_opt['D_init_iters'] if train_opt['D_init_iters'] else 0

            if train_opt['gan_type'] == 'wgan-gp':
                self.random_pt = torch.Tensor(1, 1, 1, 1).to(self.device)
                # gradient penalty loss
                self.cri_gp = GradientPenaltyLoss(device=self.device).to(self.device)
                self.l_gp_w = train_opt['gp_weight']

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
            logs_2_keep = ['l_g_pix', 'l_g_fea', 'l_g_gan', 'l_d_real', 'l_d_fake',
                           'l_d_real_fake','D_real', 'D_fake', 'D_logits_diff', 'psnr_val', 'D_update_ratio', 'LR_decrease',
                           'Correctly_distinguished', 'l_d_gp']
            self.log_dict = OrderedDict(zip(logs_2_keep, [[] for i in logs_2_keep]))

            # self.log_dict = OrderedDict()
        self.load()  # load G and D if needed
        print('---------- Model initialized ------------------')
        self.print_network()
        print('-----------------------------------------------')

    def feed_data(self, data, need_GT=True):
        # LR
        self.var_L = data['LR'].to(self.device)
        if need_HR:  # train or val
            self.var_H = data['HR'].to(self.device)

            input_ref = data['ref'] if 'ref' in data else data['HR']
            self.var_ref = input_ref.to(self.device)

    def optimize_parameters(self):
        self.gradient_step_num = self.step
        # G
        self.optimizer_G.zero_grad()
        self.fake_H = self.netG(self.var_L)

        l_g_total = 0
        if self.step % self.D_update_ratio == 0 and self.step > self.D_init_iters:
            if self.cri_pix:  # pixel loss
                if 'pixel_domain' in self.opt['train'] and self.opt['train']['pixel_domain'] == 'LR':
                    LR_size = list(self.var_L.size()[-2:])
                    l_g_pix = self.cri_pix(self.Convert_2_LR(self.fake_H,LR_size),self.Convert_2_LR(self.var_H,LR_size))
                else:
                    l_g_pix = self.cri_pix(self.fake_H,self.var_H)
                l_g_pix = self.l_pix_w * l_g_pix
                l_g_total += l_g_pix
            if self.cri_fea:  # feature loss
                real_fea = self.netF(self.var_H).detach()
                fake_fea = self.netF(self.fake_H)
                l_g_fea = self.l_fea_w * self.cri_fea(fake_fea, real_fea)
                l_g_total += l_g_fea
            # G gan + cls loss
            pred_g_fake = self.netD(self.fake_H)
            l_g_gan = self.l_gan_w * self.cri_gan(pred_g_fake, True)
            l_g_total += l_g_gan

            l_g_total.backward()
            self.optimizer_G.step()
            self.generator_changed = True

        # D
        self.optimizer_D.zero_grad()
        l_d_total = 0
        # real data
        pred_d_real = self.netD(self.var_ref)
        l_d_real = self.cri_gan(pred_d_real, True)
        # fake data
        pred_d_fake = self.netD(self.fake_H.detach())  # detach to avoid BP to G
        l_d_fake = self.cri_gan(pred_d_fake, False)

        l_d_total = l_d_real + l_d_fake

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
        if self.step % self.D_update_ratio == 0 and self.step > self.D_init_iters:
            # G
            if self.cri_pix:
                self.log_dict['l_g_pix'].append((self.step,l_g_pix.item()))
            if self.cri_fea:
                self.log_dict['l_g_fea'].append((self.step,l_g_fea.item()))
                if self.reshuffle_netF_weights:
                    self.netF.module._initialize_weights()
            self.log_dict['l_g_gan'].append((self.step,l_g_gan.item()))
        # D
        self.log_dict['l_d_real'].append((self.step,l_d_real.item()))
        self.log_dict['l_d_fake'].append((self.step,l_d_fake.item()))

        if self.opt['train']['gan_type'] == 'wgan-gp':
            self.log_dict['l_d_gp'] = l_d_gp.item()
        # D outputs
        self.log_dict['D_real'].append((self.step,torch.mean(pred_d_real.detach())))
        self.log_dict['D_fake'].append((self.step,torch.mean(pred_d_fake.detach())))
        self.step += 1

    def perform_validation(self,data_loader,cur_Z,print_rlt,save_GT_HR,save_images):
        SAVE_IMAGE_COLLAGE = True
        avg_psnr = []
        idx = 0
        image_collage = []
        if save_images:
            num_val_images = len(data_loader.dataset)
            val_images_collage_rows = int(np.floor(np.sqrt(num_val_images)))
            while val_images_collage_rows > 1:
                if np.round(num_val_images / val_images_collage_rows) == num_val_images / val_images_collage_rows:
                    break
                val_images_collage_rows -= 1
            per_image_saved_patch = min([min(im['HR'].shape[1:]) for im in data_loader.dataset]) - 2
            GT_image_collage = []
        sr_images = []
        for val_data in tqdm.tqdm(data_loader):
            if save_images:
                if idx % val_images_collage_rows == 0:  image_collage.append([]);   GT_image_collage.append([]);
            idx += 1
            img_name = os.path.splitext(os.path.basename(val_data['HR_path'][0]))[0]
            val_data['Z'] = cur_Z
            self.feed_data(val_data)
            self.test()
            visuals = self.get_current_visuals()
            sr_img = 255*util.tensor2img(visuals['SR'], out_type=np.float32)  # float32
            sr_images.append(sr_img)
            gt_img = 255*util.tensor2img(visuals['HR'], out_type=np.float32)  # float32
            avg_psnr.append(util.calculate_psnr(sr_img, gt_img))
            if save_images:
                if SAVE_IMAGE_COLLAGE:
                    margins2crop = ((np.array(sr_img.shape[:2]) - per_image_saved_patch) / 2).astype(np.int32)
                    image_collage[-1].append(np.clip(sr_img[margins2crop[0]:-margins2crop[0], margins2crop[1]:-margins2crop[1], ...], 0,255).astype(np.uint8))
                    if save_GT_HR:  # Save GT HR images
                        GT_image_collage[-1].append(np.clip(gt_img[margins2crop[0]:-margins2crop[0], margins2crop[1]:-margins2crop[1], ...], 0,255).astype(np.uint8))
                else:
                    # Save SR images for reference
                    img_dir = os.path.join(self.opt['path']['val_images'], img_name)
                    util.mkdir(img_dir)
                    save_img_path = os.path.join(img_dir, '{:s}_{:d}.png'.format(img_name, self.step))
                    util.save_img(np.clip(sr_img, 0, 255).astype(np.uint8), save_img_path)
        if save_images:
            self.generator_changed = False
        avg_psnr = 1 * np.mean(avg_psnr)
        if SAVE_IMAGE_COLLAGE and save_images:
            save_img_path = os.path.join(os.path.join(self.opt['path']['val_images']),'{:d}_{}PSNR{:.3f}.png'.format(self.step,
                ('Z' + str(cur_Z)) if self.opt['network_G']['latent_input'] else '', avg_psnr))
            util.save_img(np.concatenate([np.concatenate(col, 0) for col in image_collage], 1), save_img_path)
            if save_GT_HR:  # Save GT HR images
                util.save_img(np.concatenate([np.concatenate(col, 0) for col in GT_image_collage], 1),os.path.join(os.path.join(self.opt['path']['val_images']), 'GT_HR.png'))
        print_rlt['psnr'] += avg_psnr
        return sr_images

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.fake_H = self.netG(self.var_L)
        self.netG.train()

    # def get_current_log(self):
    #     return self.log_dict
    def get_current_log(self):
        dict_2_return = OrderedDict()
        for key in self.log_dict:
            if len(self.log_dict[key])>0:
                if isinstance(self.log_dict[key][-1],tuple) or len(self.log_dict[key][-1])>1:
                    dict_2_return[key] = self.log_dict[key][-1][1]
                else:
                    dict_2_return[key] = self.log_dict[key][-1]
        return dict_2_return

    def save_log(self):
        np.savez(os.path.join(self.log_path,'logs.npz'), ** self.log_dict)

    def load_log(self,max_step=None):
        PREPEND_OLD_LOG = False
        loaded_log = np.load(os.path.join(self.log_path,'logs.npz'),allow_pickle=True)
        if PREPEND_OLD_LOG:
            old_log = np.load(os.path.join(self.log_path, 'old_logs.npz'))
        self.log_dict = OrderedDict([val for val in zip(self.log_dict.keys(),[[] for i in self.log_dict.keys()])])
        for key in loaded_log.files:
            if key=='psnr_val':
                self.log_dict[key] = ([tuple(val) for val in old_log[key]] if PREPEND_OLD_LOG else [])+[tuple(val) for val in loaded_log[key]]
            else:
                self.log_dict[key] = (list(old_log[key]) if PREPEND_OLD_LOG else [])+list(loaded_log[key])
                if len(self.log_dict[key])>0 and isinstance(self.log_dict[key][0][1],torch.Tensor):#Supporting old files where data was not converted from tensor - Causes slowness.
                    self.log_dict[key] = [[val[0],val[1].item()] for val in self.log_dict[key]]
            if max_step is not None:
                self.log_dict[key] = [pair for pair in self.log_dict[key] if pair[0]<=max_step]

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
            with open(network_path, 'w') as f:
                f.write(message)

            # Discriminator
            s, n,_ = self.get_network_description(self.netD)
            print('Number of parameters in D: {:,d}'.format(n))
            message = '\n\n\n-------------- Discriminator --------------\n' + s + '\n'
            with open(network_path, 'a') as f:
                f.write(message)

            if self.cri_fea:  # F, Perceptual Network
                s, n = self.get_network_description(self.netF)
                print('Number of parameters in F: {:,d}'.format(n))
                message = '\n\n\n-------------- Perceptual Network --------------\n' + s + '\n'
                with open(network_path, 'a') as f:
                    f.write(message)

    def load(self,max_step=None,resume_train=None):
        resume_training = resume_train if resume_train is not None else (self.opt['is_train'] and self.opt['train']['resume'])
        if max_step is not None or (resume_training is not None and resume_training) or not self.opt['is_train']:
            model_name = [name for name in os.listdir(self.opt['path']['models']) if '_G.pth' in name]
            model_name = sorted(model_name,key=lambda x: int(re.search('(\d)+(?=_G.pth)',x).group(0)))
            if max_step is not None:
                model_name = [model for model in model_name if int(re.search('(\d)+(?=_G.pth)',model).group(0))<=max_step]
            model_name = model_name[-1]
            loaded_model_step = int(re.search('(\d)+(?=_G.pth)',model_name).group(0))
            if self.opt['is_train']:
                self.step = (loaded_model_step+1)
                print('Resuming training with model for G [{:s}] ...'.format(os.path.join(self.opt['path']['models'],model_name)))
                self.load_network(os.path.join(self.opt['path']['models'],model_name), self.netG,optimizer=self.optimizer_G)
                self.load_log(max_step=loaded_model_step)
                if self.D_exists:
                    model_name = str(loaded_model_step)+'_D.pth'
                    print('Resuming training with model for D [{:s}] ...'.format(os.path.join(self.opt['path']['models'],model_name)))
                    self.load_network(os.path.join(self.opt['path']['models'],model_name), self.netD,optimizer=self.optimizer_D)
            else:
                print('Testing model for G [{:s}] ...'.format(os.path.join(self.opt['path']['models'],model_name)))
                self.load_network(os.path.join(self.opt['path']['models'],model_name), self.netG)
                if 'netD' in self.__dict__.keys(): #When running from GUI
                    model_name = model_name.replace('_G','_D')
                    print('Loading also model for D [{:s}] ...'.format(os.path.join(self.opt['path']['models'], model_name)))
                    self.load_network(os.path.join(self.opt['path']['models'], model_name), self.netD)
                self.gradient_step_num = loaded_model_step

        else:
            load_path_G = self.opt['path']['pretrain_model_G']
            if load_path_G is not None:
                print('loading model for G [{:s}] ...'.format(load_path_G))
                self.load_network(load_path_G, self.netG)
            load_path_D = self.opt['path']['pretrain_model_D']
            if self.opt['is_train'] and load_path_D is not None:
                print('loading model for D [{:s}] ...'.format(load_path_D))
                self.load_network(load_path_D, self.netD,optimizer=self.optimizer_D)

    # def load(self):
        # load_path_G = self.opt['path']['pretrain_model_G']
        # if load_path_G is not None:
        #     print('loading model for G [{:s}] ...'.format(load_path_G))
        #     self.load_network(load_path_G, self.netG)
        # load_path_D = self.opt['path']['pretrain_model_D']
        # if self.opt['is_train'] and load_path_D is not None:
        #     print('loading model for D [{:s}] ...'.format(load_path_D))
        #     self.load_network(load_path_D, self.netD)

    def save(self, iter_label):
        # self.save_network(self.save_dir, self.netG, 'G', iter_label,self.optimizer_G)
        # self.save_network(self.save_dir, self.netD, 'D', iter_label,self.optimizer_D)
        saving_path = self.save_network(self.save_dir, self.netG, 'G', iter_label,self.optimizer_G)
        if self.D_exists:
            self.save_network(self.save_dir, self.netD, 'D', iter_label,self.optimizer_D)
        return saving_path

