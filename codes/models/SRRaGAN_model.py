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
import h5py

class SRRaGANModel(BaseModel):
    def __init__(self, opt,accumulation_steps_per_batch):
        super(SRRaGANModel, self).__init__(opt)
        train_opt = opt['train']
        self.log_path = opt['path']['log']
        self.noise_input = opt['network_G']['noise_input'] if opt['network_G']['noise_input']!='None' else None
        self.relativistic_D = opt['network_D']['relativistic'] is None or bool(opt['network_D']['relativistic'])
        self.add_quantization_noise = bool(opt['network_D']['add_quantization_noise'])
        self.min_accumulation_steps = min(
            [opt['train']['grad_accumulation_steps_G'], opt['train']['grad_accumulation_steps_D']])
        self.max_accumulation_steps = accumulation_steps_per_batch
        self.grad_accumulation_steps_G = opt['train']['grad_accumulation_steps_G']
        self.grad_accumulation_steps_D = opt['train']['grad_accumulation_steps_D']
        # define networks and load pretrained models
        self.DTE_net = None
        self.DTE_arch = opt['network_G']['DTE_arch']
        self.decomposed_output = self.DTE_arch and bool(opt['network_D']['decomposed_input'])
        self.step = 0
        if self.DTE_arch or (opt['is_train'] and opt['train']['DTE_exp']):
            assert self.opt['train']['pixel_domain']=='HR' or not self.DTE_arch,'Why should I use DTE_arch AND penalize MSE in the LR domain?'
            DTE_conf = DTEnet.Get_DTE_Conf(opt['scale'])
            DTE_conf.sigmoid_range_limit = bool(opt['network_G']['sigmoid_range_limit'])
            DTE_conf.input_range = np.array(opt['range'])
            DTE_conf.decomposed_output = bool(opt['network_D']['decomposed_input'])
            self.DTE_net = DTEnet.DTEnet(DTE_conf)
            if not self.DTE_arch:
                self.DTE_net.WrapArchitecture_PyTorch(only_padders=True)
        self.netG = networks.define_G(opt,DTE=self.DTE_net).to(self.device)  # G
        logs_2_keep = ['l_g_pix', 'l_g_fea', 'l_g_range', 'l_g_gan', 'l_d_real', 'l_d_fake',
                       'D_real', 'D_fake','D_logits_diff','psnr_val','D_update_ratio','LR_decrease','Correctly_distinguished']
        self.log_dict = OrderedDict(zip(logs_2_keep, [[] for i in logs_2_keep]))
        self.debug = 'debug' in opt['path']['log']
        if self.is_train:
            self.netD = networks.define_D(opt,DTE=self.DTE_net).to(self.device)  # D
            self.netG.train()
            self.netD.train()
        self.load()  # load G and D if needed

        # define losses, optimizer and scheduler
        if self.is_train:
            # G pixel loss
            if train_opt['pixel_weight'] > 0 or self.debug:
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
            if train_opt['feature_weight'] > 0 or self.debug:
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
            if train_opt['range_weight'] > 0 or self.debug:
                self.cri_range = CreateRangeLoss(opt['range'])
                self.l_range_w = train_opt['range_weight']
            else:
                print('Remove range loss.')
                self.cri_range = None

            # GD gan loss
            self.cri_gan = GANLoss(train_opt['gan_type'], 1.0, 0.0).to(self.device)
            self.l_gan_w = train_opt['gan_weight']
            # D_update_ratio and D_init_iters are for WGAN
            self.global_D_update_ratio = train_opt['D_update_ratio'] if train_opt['D_update_ratio'] is not None else 1
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
            if os.path.isfile(os.path.join(self.log_path,'lr.npz')):
                lr_G = np.load(os.path.join(self.log_path,'lr.npz'))['lr_G']
                lr_D = np.load(os.path.join(self.log_path, 'lr.npz'))['lr_D']
            else:
                lr_G = train_opt['lr_G']
                lr_D = train_opt['lr_D']
            self.optimizer_G = torch.optim.Adam(optim_params, lr=lr_G, \
                weight_decay=wd_G, betas=(train_opt['beta1_G'], 0.999))
            self.optimizers.append(self.optimizer_G)
            # D
            wd_D = train_opt['weight_decay_D'] if train_opt['weight_decay_D'] else 0
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=lr_D, \
                weight_decay=wd_D, betas=(train_opt['beta1_D'], 0.999))
            self.optimizers.append(self.optimizer_D)

            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(lr_scheduler.MultiStepLR(optimizer, \
                        train_opt['lr_steps'], train_opt['lr_gamma']))
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')
            self.generator_step = False
            self.generator_changed = True#Initializing to true,to save the initial state```````

        print('---------- Model initialized ------------------')
        self.print_network()
        print('-----------------------------------------------')

    def feed_data(self, data, need_HR=True):
        # LR
        self.var_L = data['LR'].to(self.device)
        if self.noise_input is not None:
            if 'Z' in data.keys():
                cur_Z = data['Z']
            else:
                cur_Z = torch.normal(mean=torch.from_numpy(np.zeros(shape=[self.var_L.size(dim=0),1,1,1])).type(torch.FloatTensor),
                    std=torch.from_numpy(np.ones(shape=[self.var_L.size(dim=0),1,1,1])).type(torch.FloatTensor))
            self.var_L = torch.cat([(cur_Z*torch.ones(size=[1,1,self.var_L.size()[2],self.var_L.size()[3]])).type(self.var_L.type()),self.var_L],dim=1)
        if need_HR:  # train or val
            if self.add_quantization_noise:
                data['HR'] += (torch.rand_like(data['HR'])-0.5)/255 # Adding quantization noise to real images to avoid discriminating based on quantization differences between real and fake
            self.var_H = data['HR'].to(self.device)

            input_ref = data['ref'] if 'ref' in data else data['HR']
            self.var_ref = input_ref.to(self.device)
    def Convert_2_LR(self,size):
        return Upsample(size=size,mode='bilinear')
    def optimize_parameters(self):
        VERIFY_D_USING_PAST_PERFORMANCE = True
        self.gradient_step_num = self.step//self.max_accumulation_steps
        first_grad_accumulation_step_G = self.step%self.grad_accumulation_steps_G==0
        last_grad_accumulation_step_G = self.step % self.grad_accumulation_steps_G == (self.grad_accumulation_steps_G-1)
        first_grad_accumulation_step_D = self.step%self.grad_accumulation_steps_D==0
        last_grad_accumulation_step_D = self.step % self.grad_accumulation_steps_D == (self.grad_accumulation_steps_D-1)

        if first_grad_accumulation_step_D:
            if self.global_D_update_ratio>0:
                self.cur_D_update_ratio = self.global_D_update_ratio
            elif len(self.log_dict['D_logits_diff'])<self.opt['train']['D_valid_Steps_4_G_update']:
                self.cur_D_update_ratio = self.opt['train']['D_valid_Steps_4_G_update']
            else:#Varying update ratio:
                log_mean_D_diff = np.log(max(1e-5,np.mean([val[1] for val in self.log_dict['D_logits_diff'][-self.opt['train']['D_valid_Steps_4_G_update']:]])))
                if log_mean_D_diff<-2:
                    self.cur_D_update_ratio = int(-2*np.ceil((log_mean_D_diff+1)*2)/2)
                else:
                    self.cur_D_update_ratio = 1/max(1,int(np.floor((log_mean_D_diff+2)*20)))
                    # self.cur_D_update_ratio = max(1/50,np.floor(100*(log_mean_D_diff+1))/-100)
        # G
        if first_grad_accumulation_step_D or self.generator_step:
            G_grads_retained = True
            for p in self.netG.parameters():
                p.requires_grad = True
        else:
            G_grads_retained = False
            for p in self.netG.parameters():
                p.requires_grad = False
        self.fake_H = self.netG(self.var_L)
        if self.DTE_net is not None:
            if self.decomposed_output:
                self.fake_H = [self.DTE_net.HR_unpadder(self.fake_H[0]),self.DTE_net.HR_unpadder(self.fake_H[1])]
            else:
                self.fake_H = self.DTE_net.HR_unpadder(self.fake_H)
            self.var_H,self.var_ref = self.DTE_net.HR_unpadder(self.var_H),self.DTE_net.HR_unpadder(self.var_ref)

        # D
        if (self.gradient_step_num) % max([1,np.ceil(1/self.cur_D_update_ratio)]) == 0 and self.gradient_step_num > -self.D_init_iters:
            for p in self.netD.parameters():
                p.requires_grad = True
            for p in self.netG.parameters():
                p.requires_grad = False
            if first_grad_accumulation_step_D:
                self.optimizer_D.zero_grad()
                self.l_d_real_grad_step,self.l_d_fake_grad_step,self.D_real_grad_step,self.D_fake_grad_step,self.D_logits_diff_grad_step = [],[],[],[],[]
            l_d_total = 0
            # pred_d_real = self.netD(torch.cat([self.fake_H[0],self.var_ref-self.fake_H[0]],1) if self.decomposed_output else self.var_ref)
            # pred_d_fake = self.netD((torch.cat(self.fake_H,1) if self.decomposed_output else self.fake_H).detach())  # detach to avoid BP to G
            pred_d_real = self.netD([self.fake_H[0],self.var_ref-self.fake_H[0]] if self.decomposed_output else self.var_ref)
            pred_d_fake = self.netD([t.detach() for t in self.fake_H] if self.decomposed_output else self.fake_H.detach())  # detach to avoid BP to G
            if self.relativistic_D:
                l_d_real = self.cri_gan(pred_d_real - torch.mean(pred_d_fake), True)
                l_d_fake = self.cri_gan(pred_d_fake - torch.mean(pred_d_real), False)
            else:
                l_d_real = 2*self.cri_gan(pred_d_real, True)#Multiplying by 2 to be consistent with the SRGAN code, where losses are summed and not averaged.
                l_d_fake = 2*self.cri_gan(pred_d_fake, False)

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

            self.l_d_real_grad_step.append(l_d_real.item())
            self.l_d_fake_grad_step.append(l_d_fake.item())
            self.D_real_grad_step.append(torch.mean(pred_d_real.detach()).item())
            self.D_fake_grad_step.append(torch.mean(pred_d_fake.detach()).item())
            self.D_logits_diff_grad_step.append(list(torch.mean(pred_d_real.detach()-pred_d_fake.detach(),dim=(1,2,3)).data.cpu().numpy()))
            if first_grad_accumulation_step_D:
                self.generator_step = (self.gradient_step_num) % max(
                    [1, self.cur_D_update_ratio]) == 0 and self.gradient_step_num > self.D_init_iters
                # When D batch is larger than G batch, run G iter on final D iter steps, to avoid updating G in the middle of calculating D gradients.
                self.generator_step = self.generator_step and self.step % \
                                      self.grad_accumulation_steps_D >= self.grad_accumulation_steps_D - self.grad_accumulation_steps_G
                if VERIFY_D_USING_PAST_PERFORMANCE:
                    if self.generator_step and self.opt['train']['D_valid_Steps_4_G_update'] > 0:
                        self.generator_step = len(self.log_dict['D_logits_diff']) >= self.opt['train']['D_valid_Steps_4_G_update'] and \
                                              all([val[1] > np.log(self.opt['train']['min_D_prob_ratio_4_G']) for val in
                                              self.log_dict['D_logits_diff'][-self.opt['train']['D_valid_Steps_4_G_update']:]]) and \
                                              all([val[1] > np.log(self.opt['train']['min_mean_D_correct']) for val in
                                                   self.log_dict['Correctly_distinguished'][-self.opt['train']['D_valid_Steps_4_G_update']:]])

            if not VERIFY_D_USING_PAST_PERFORMANCE and self.generator_step:
                self.generator_step = all([val > 0 for val in self.D_logits_diff_grad_step[-1]]) \
                    and np.mean(self.D_logits_diff_grad_step[-1])>np.log(self.opt['train']['min_D_prob_ratio_4_G'])
            if G_grads_retained and not self.generator_step:# Freeing up the unnecessary gradients memory:
                    self.fake_H = [var.detach() for var in self.fake_H] if self.decomposed_output else self.fake_H.detach()
            l_d_total.backward(retain_graph=self.generator_step)
            if last_grad_accumulation_step_D:
                self.optimizer_D.step()
                # set log
                self.log_dict['l_d_real'].append((self.gradient_step_num,np.mean(self.l_d_real_grad_step)))
                self.log_dict['l_d_fake'].append((self.gradient_step_num,np.mean(self.l_d_fake_grad_step)))

                if self.opt['train']['gan_type'] == 'wgan-gp':
                    self.log_dict['l_d_gp'].append((self.gradient_step_num,l_d_gp.item()))
                # D outputs
                self.log_dict['D_real'].append((self.gradient_step_num,np.mean(self.D_real_grad_step)))
                self.log_dict['D_fake'].append((self.gradient_step_num,np.mean(self.D_fake_grad_step)))
                self.log_dict['D_logits_diff'].append((self.gradient_step_num,np.mean(self.D_logits_diff_grad_step)))
                self.log_dict['Correctly_distinguished'].append((self.gradient_step_num,np.mean([val0>0 for val1 in self.D_logits_diff_grad_step for val0 in val1])))
                self.log_dict['D_update_ratio'].append((self.gradient_step_num,self.cur_D_update_ratio))

        # G step:
        l_g_total = 0#torch.zeros(size=[],requires_grad=True).type(torch.cuda.FloatTensor)
        if self.generator_step:
            for p in self.netD.parameters():
                p.requires_grad = False
            for p in self.netG.parameters():
                p.requires_grad = True
            if first_grad_accumulation_step_G:
                self.optimizer_G.zero_grad()
                self.l_g_pix_grad_step,self.l_g_fea_grad_step,self.l_g_gan_grad_step,self.l_g_range_grad_step = [],[],[],[]
            if self.cri_pix:  # pixel loss
                if 'pixel_domain' in self.opt['train'] and self.opt['train']['pixel_domain']=='LR':
                    LR_size = list(self.var_L.size()[-2:])
                    l_g_pix = self.cri_pix(self.Convert_2_LR(LR_size)(self.fake_H), self.Convert_2_LR(LR_size)(self.var_H))
                else:
                    l_g_pix = self.cri_pix((self.fake_H[0]+self.fake_H[1]) if self.decomposed_output else self.fake_H, self.var_H)
                l_g_total += self.l_pix_w * l_g_pix
            if self.cri_fea:  # feature loss
                if 'feature_domain' in self.opt['train'] and self.opt['train']['feature_domain']=='LR':
                    LR_size = list(self.var_L.size()[-2:])
                    real_fea = self.netF(self.Convert_2_LR(LR_size)(self.var_H)).detach()
                    fake_fea = self.netF(self.Convert_2_LR(LR_size)(self.fake_H))
                else:
                    real_fea = self.netF(self.var_H).detach()
                    fake_fea = self.netF((self.fake_H[0]+self.fake_H[1]) if self.decomposed_output else self.fake_H)
                l_g_fea = self.cri_fea(fake_fea, real_fea)
                l_g_total += self.l_fea_w * l_g_fea
            if self.cri_range: #range loss
                l_g_range = self.cri_range((self.fake_H[0]+self.fake_H[1]) if self.decomposed_output else self.fake_H)
                l_g_total += self.l_range_w * l_g_range
            # G gan + cls loss
            # pred_g_fake = self.netD(torch.cat(self.fake_H,1) if self.decomposed_output else self.fake_H)
            # pred_d_real = self.netD(torch.cat([self.fake_H[0],self.var_ref-self.fake_H[0]],1) if self.decomposed_output else self.var_ref).detach()
            pred_g_fake = self.netD(self.fake_H)
            pred_d_real = self.netD([self.fake_H[0],self.var_ref-self.fake_H[0]] if self.decomposed_output else self.var_ref).detach()

            if self.relativistic_D:
                l_g_gan = self.l_gan_w * (self.cri_gan(pred_d_real - torch.mean(pred_g_fake), False) +
                                          self.cri_gan(pred_g_fake - torch.mean(pred_d_real), True)) / 2
            else:
                l_g_gan = self.l_gan_w * self.cri_gan(pred_g_fake, True)

            l_g_total += l_g_gan

            l_g_total.backward()
            self.l_g_pix_grad_step.append(l_g_pix.item())
            self.l_g_fea_grad_step.append(l_g_fea.item())
            self.l_g_gan_grad_step.append(l_g_gan.item())
            self.l_g_range_grad_step.append(l_g_range.item())
            if last_grad_accumulation_step_G:
                self.optimizer_G.step()
                self.generator_changed = True
                # set log
                if self.cri_pix:
                    self.log_dict['l_g_pix'].append((self.gradient_step_num,np.mean(self.l_g_pix_grad_step)))
                if self.cri_fea:
                    self.log_dict['l_g_fea'].append((self.gradient_step_num,np.mean(self.l_g_fea_grad_step)))
                if self.cri_range:
                    self.log_dict['l_g_range'].append((self.gradient_step_num,np.mean(self.l_g_range_grad_step)))
                self.log_dict['l_g_gan'].append((self.gradient_step_num,np.mean(self.l_g_gan_grad_step)))
        self.step += 1

        # set log
        # if step % self.global_D_update_ratio == 0 and step > self.D_init_iters:
            # G
            # if self.cri_pix:
            #     self.log_dict['l_g_pix'].append(l_g_pix.item())
            # if self.cri_fea:
            #     self.log_dict['l_g_fea'].append(l_g_fea.item())
            # if self.cri_range:
            #     self.log_dict['l_g_range'].append(l_g_range.item())
            # self.log_dict['l_g_gan'].append(l_g_gan.item())
        # D
        # self.log_dict['l_d_real'].append(l_d_real.item())
        # self.log_dict['l_d_fake'].append(l_d_fake.item())
        #
        # if self.opt['train']['gan_type'] == 'wgan-gp':
        #     self.log_dict['l_d_gp'].append(l_d_gp.item())
        # # D outputs
        # self.log_dict['D_real'].append(torch.mean(pred_d_real.detach()))
        # self.log_dict['D_fake'].append(torch.mean(pred_d_fake.detach()))
        # self.log_dict['D_logits_diff'].append(torch.mean(pred_d_real.detach()-pred_d_fake.detach()))

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            self.fake_H = self.netG(self.var_L)
        self.netG.train()

    def update_learning_rate(self,cur_step=None):
        #The returned value is LR_too_low
        SLOPE_BASED = False
        LOSS_BASED = True
        if len(self.log_dict['D_logits_diff'])<2 * self.opt['train']['steps_4_lr_std'] or self.log_dict['D_logits_diff'][0][0]>cur_step-self.opt['train']['steps_4_lr_std']:#Check after a minimal number of steps
            return False
        if os.path.isfile(os.path.join(self.log_path, 'lr.npz')):#Allow enough steps between checks
            if cur_step - np.load(os.path.join(self.log_path, 'lr.npz'))['step_num'] <= 2 * self.opt['train']['steps_4_lr_std']:
                return False
        if SLOPE_BASED:
            std,slope = 0,0
            for key in ['l_d_real','l_d_fake','l_g_gan']:
                relevant_loss_vals = [val[1] for val in self.log_dict[key] if val[0] >= cur_step - self.opt['train']['steps_4_lr_std']]
                [cur_slope, _], [[cur_var, _], _] = np.polyfit([i for i in range(len(relevant_loss_vals))],relevant_loss_vals,1, cov=True)
                # We take the the standard deviation as a measure
                std += 0.5*(0.5 if 'l_d' in key else 1.)*np.sqrt(cur_var)
                slope += 0.5*(0.5 if 'l_d' in key else 1.)*cur_slope
            reduce_lr = -self.opt['train']['lr_change_ratio']*slope<std
        elif LOSS_BASED:
            # win_length = 100
            # plt.clf()
            # plt.plot([self.log_dict['l_d_real'][i + win_length][0] for i in
            #           range(len(self.log_dict['l_d_real']) - win_length)], [np.std([(self.log_dict['l_d_real'][
            #                                                                          i:i + win_length][j][1] +
            #                                                                          self.log_dict['l_d_fake'][
            #                                                                          i:i + win_length][j][1]) / 2 for j
            #                                                                         in range(
            #         len(self.log_dict['l_d_real'][i:i + win_length]))]) for i in range(
            #     len(self.log_dict['l_d_real']) - win_length)]);
            # plt.savefig('D_loss_STD.pdf')
            relevant_loss_vals = [(val[1]+self.log_dict['l_d_fake'][i][1])/2 for i,val in enumerate(self.log_dict['l_d_real']) if val[0] >= cur_step - self.opt['train']['steps_4_lr_std']]
            reduce_lr = np.std(relevant_loss_vals)>self.opt['train']['std_4_lr_drop']
        else:
            relevant_D_logits_difs = [val[1] for val in self.log_dict['D_logits_diff'] if val[0] >= cur_step - self.opt['train']['steps_4_lr_std']]
            reduce_lr = np.std(relevant_D_logits_difs)>self.opt['train']['std_4_lr_drop']
        if reduce_lr:
            if SLOPE_BASED:
                print('slope: ', slope, 'STD: ', std)
            self.load(max_step=cur_step - self.opt['train']['steps_4_lr_std'])
            for optimizer in [self.optimizer_G,self.optimizer_D]:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= self.opt['train']['lr_gamma']
                    if param_group['lr']<1e-8:
                        return True
            np.savez(os.path.join(self.log_path,'lr.npz'),step_num=cur_step,lr_G =self.optimizer_G.param_groups[0]['lr'],lr_D =self.optimizer_D.param_groups[0]['lr'])
            self.log_dict['LR_decrease'].append([self.gradient_step_num,{'lr_G':self.optimizer_G.param_groups[0]['lr'],'lr_D':self.optimizer_D.param_groups[0]['lr']}])
            print('LR(D) reduced to %.2e, LR(G) reduced to %.2e.'%(self.optimizer_D.param_groups[0]['lr'],self.optimizer_G.param_groups[0]['lr']))
        return False

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
        loaded_log = np.load(os.path.join(self.log_path,'logs.npz'))
        self.log_dict = OrderedDict([val for val in zip(self.log_dict.keys(),[[] for i in self.log_dict.keys()])])
        for key in loaded_log.files:
            if key=='psnr_val':
                self.log_dict[key] = [tuple(val) for val in loaded_log[key]]
            else:
                self.log_dict[key] = list(loaded_log[key])
                if len(self.log_dict[key])>0 and isinstance(self.log_dict[key][0][1],torch.Tensor):#Supporting old files where data was not converted from tensor - Causes slowness.
                    self.log_dict[key] = [[val[0],val[1].item()] for val in self.log_dict[key]]
            if max_step is not None:
                self.log_dict[key] = [pair for pair in self.log_dict[key] if pair[0]<=max_step]
    def display_log_figure(self):
        # keys_2_display = ['l_g_pix', 'l_g_fea', 'l_g_range', 'l_g_gan', 'l_d_real', 'l_d_fake', 'D_real', 'D_fake','D_logits_diff','psnr_val']
        keys_2_display = ['l_g_gan','D_logits_diff', 'psnr_val','l_g_pix','l_g_fea','l_g_range','D_update_ratio']
        PER_KEY_FIGURE = True
        legend_strings = []
        plt.figure(2)
        plt.clf()
        min_global_val, max_global_val = np.finfo(np.float32).max,np.finfo(np.float32).min
        for key in keys_2_display:
            if key in self.log_dict.keys() and len(self.log_dict[key])>0:
                if PER_KEY_FIGURE:
                    plt.figure(1)
                    plt.clf()
                if isinstance(self.log_dict[key][0],tuple) or len(self.log_dict[key][0])==2:
                    cur_curve = [np.array([val[0] for val in self.log_dict[key]]),np.array([val[1] for val in self.log_dict[key]])]
                    min_val,max_val = self.plot_curves(cur_curve[0],cur_curve[1])
                    if 'LR_decrease' in self.log_dict.keys():
                        for decrease in self.log_dict['LR_decrease']:
                            plt.plot([decrease[0],decrease[0]],[min_val,max_val],'k')
                    if isinstance(self.log_dict[key][0][1],torch.Tensor):
                        series_avg = np.mean([val[1].data.cpu().numpy() for val in self.log_dict[key]])
                    else:
                        series_avg = np.mean([val[1] for val in self.log_dict[key]])
                else:
                    raise Exception('Should always have step numbers')
                    self.plot_curves(self.log_dict[key])
                    # plt.plot(self.log_dict[key])
                    if isinstance(self.log_dict[key][0][1],torch.Tensor):
                        series_avg = np.mean([val[1].data.cpu().numpy() for val in self.log_dict[key]])
                    else:
                        series_avg = np.mean(self.log_dict[key])
                cur_legend_string = key + ' (%.2e)' % (series_avg)
                if PER_KEY_FIGURE:
                    plt.xlabel('Steps')
                    plt.legend([cur_legend_string], loc='best')
                    # legend_strings = []
                    plt.savefig(os.path.join(self.log_path, 'logs_%s.pdf' % (key)))
                    plt.figure(2)
                    if key=='psnr_val':
                        cur_legend_string = 'MSE_val' + ' (%s:%.2e)' % (key,series_avg)
                        cur_curve[1] = 255*np.exp(-cur_curve[1]/20)
                    cur_curve[1] = (cur_curve[1]-np.mean(cur_curve[1]))/np.std(cur_curve[1])
                    min_val,max_val = self.plot_curves(cur_curve[0],cur_curve[1])
                    min_global_val,max_global_val = np.minimum(min_global_val,min_val),np.maximum(max_global_val,max_val)
                legend_strings.append(cur_legend_string)
        plt.legend(legend_strings,loc='best')
        plt.xlabel('Steps')
        if 'LR_decrease' in self.log_dict.keys():
            for decrease in self.log_dict['LR_decrease']:
                plt.plot([decrease[0], decrease[0]], [min_global_val,max_global_val], 'k')
        plt.savefig(os.path.join(self.log_path,'logs.pdf'))
        # plt.close(general_fig)
        # plt.close(per_key_fig)


    def get_current_visuals(self, need_HR=True,entire_batch=False):
        out_dict = OrderedDict()
        if entire_batch:
            out_dict['LR'] = self.var_L.detach().float().cpu()
            out_dict['SR'] = (self.fake_H[0]+self.fake_H[1] if isinstance(self.fake_H,list) else self.fake_H).detach().float().cpu()
            if need_HR:
                out_dict['HR'] = self.var_H.detach().float().cpu()
        else:
            out_dict['LR'] = self.var_L.detach()[0].float().cpu()
            out_dict['SR'] = (self.fake_H[0]+self.fake_H[1] if isinstance(self.fake_H,list) else self.fake_H).detach()[0].float().cpu()
            if need_HR:
                out_dict['HR'] = self.var_H.detach()[0].float().cpu()
        return out_dict
    def plot_curves(self,steps,loss):
        SMOOTH_CURVES = True
        if SMOOTH_CURVES:
            smoothing_win = np.minimum(np.maximum(len(loss)/20,np.sqrt(len(loss))),1000).astype(np.int32)
            loss = np.convolve(loss,np.ones([smoothing_win])/smoothing_win,'valid')
            if steps is not None:
                steps = np.convolve(steps, np.ones([smoothing_win]) / smoothing_win,'valid')
        if steps is not None:
            plt.plot(steps,loss)
        else:
            plt.plot(loss)
        return np.min(loss),np.max(loss)

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
            s, n,receptive_field = self.get_network_description(self.netD)
            print('Number of parameters in D: {:,d}. Receptive field size: {:,d}'.format(n,receptive_field))
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

    def load(self,max_step=None):
        resume_training = self.opt['is_train'] and self.opt['train']['resume']
        if max_step is not None or (resume_training is not None and resume_training):
            model_name = [name for name in os.listdir(self.opt['path']['models']) if '_G.pth' in name]
            model_name = sorted(model_name,key=lambda x: int(re.search('(\d)+(?=_G.pth)',x).group(0)))
            if max_step is not None:
                model_name = [model for model in model_name if int(re.search('(\d)+(?=_G.pth)',model).group(0))<=max_step]
            model_name = model_name[-1]
            loaded_model_step = int(re.search('(\d)+(?=_G.pth)',model_name).group(0))
            self.step = (loaded_model_step+1)*self.max_accumulation_steps
            print('Resuming training with model for G [{:s}] ...'.format(os.path.join(self.opt['path']['models'],model_name)))
            self.load_network(os.path.join(self.opt['path']['models'],model_name), self.netG)
            self.load_log(max_step=loaded_model_step)
            if self.opt['is_train']:
                # model_name = [name for name in os.listdir(self.opt['path']['models']) if '_D.pth' in name]
                # model_name = sorted(model_name, key=lambda x: int(re.search('(\d)+(?=_D.pth)', x).group(0)))[-1]
                model_name = str(loaded_model_step)+'_D.pth'
                print('Resuming training with model for D [{:s}] ...'.format(os.path.join(self.opt['path']['models'],model_name)))
                self.load_network(os.path.join(self.opt['path']['models'],model_name), self.netD)

        else:
            load_path_G = self.opt['path']['pretrain_model_G']
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
