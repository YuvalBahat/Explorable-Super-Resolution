import os
from collections import OrderedDict
# import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import re
import models.networks as networks
from .base_model import BaseModel
from models.modules.loss import GANLoss, GradientPenaltyLoss,CreateRangeLoss,FilterLoss,Latent_channels_desc_2_num_channels
import CEM.CEMnet as CEMnet
import numpy as np
from collections import deque
from utils.util import SVD_2_LatentZ
from Z_optimization import Z_optimizer
import tqdm
from utils import util



def Unit_Circle_rejection_Sampling(batch_size):
    cur_Z = torch.rand([batch_size,3,1,1])
    while torch.any((cur_Z[:,1:3,:,:]**2).sum(1)>1):
        rejected_samples = ((cur_Z[:,1:3,:,:]**2).sum(1)>1).squeeze()
        cur_Z[rejected_samples] = torch.rand([rejected_samples.sum().item(),3,1,1])
    return cur_Z

class SRRaGANModel(BaseModel):
    def __init__(self, opt,accumulation_steps_per_batch=1,init_Fnet=None,init_Dnet=None,**kwargs):
        super(SRRaGANModel, self).__init__(opt)
        train_opt = opt['train']
        self.log_path = opt['path']['log']
        self.latent_input_domain = opt['network_G']['latent_input_domain']
        self.latent_input = opt['network_G']['latent_input'] if opt['network_G']['latent_input']!='None' else None
        if self.latent_input is not None:
            self.Z_size_factor = opt['scale'] if 'HR' in opt['network_G']['latent_input_domain'] else 1
        self.num_latent_channels = 0
        self.debug = 'debug' in opt['path']['log']
        self.using_encoder = False  # train_opt['latent_weight'] > 0 Now using 'latent_weight' parameter for the new latent input configuration
        self.cri_latent = None
        self.optimalZ_loss_type = None
        self.generator_started_learning = False #I'm adding this flag to avoid wasting time optimizing over the Z space when D is still in its early learning phase. I don't change it when resuming training of a saved model - it would change by itself after 1 generator step.
        self.num_latent_channels = FilterLoss(latent_channels=opt['network_G']['latent_channels']).num_channels
        if self.latent_input is not None:
            if self.is_train:
                # Loss encouraging effect of Z:
                self.l_latent_w = train_opt['latent_weight']
                if train_opt['latent_weight']>0 or self.debug:
                    self.cri_latent = FilterLoss(latent_channels=opt['network_G']['latent_channels'])
            else:
                assert isinstance(opt['network_G']['latent_channels'],int)
        # define networks and load pretrained models
        self.CEM_net = None
        self.CEM_arch = opt['network_G']['CEM_arch']
        self.step = 0
        if self.CEM_arch or (opt['is_train'] and train_opt['CEM_exp']) or self.latent_input is not None: #The last option is for testing ESRGAN with latent input, so that I can use CEM_net.Project_2_ortho_2_NS()
            CEM_conf = CEMnet.Get_CEM_Conf(opt['scale'])
            CEM_conf.sigmoid_range_limit = bool(opt['network_G']['sigmoid_range_limit'])
            CEM_conf.input_range = np.array(opt['range'])
            if self.is_train:
                assert train_opt['pixel_domain']=='HR' or not self.CEM_arch,'Why should I use CEM_arch AND penalize MSE in the LR domain?'
                CEM_conf.decomposed_output = bool(opt['network_D']['decomposed_input'])
            if opt['test'] is not None and opt['test']['kernel']=='estimated':
                # Using a non-accurate estimated kernel increases the risk of insability when inverting hTh, so I take a higher lower bound:
                CEM_conf.lower_magnitude_bound = 0.1
            self.CEM_net = CEMnet.CEMnet(CEM_conf,upscale_kernel=kwargs['kernel'] if 'kernel' in kwargs.keys() else None if opt['test'] is None else opt['test']['kernel'])
            if not self.CEM_arch:
                self.CEM_net.WrapArchitecture_PyTorch(only_padders=True)
        self.netG = networks.define_G(opt,CEM=self.CEM_net,num_latent_channels=self.num_latent_channels)  # G
        # print('Receptive field of G:',util.compute_RF_numerical(self.netG.module.cpu(),np.ones([1,3,256,256])))
        self.netG.to(self.device)
        logs_2_keep = ['l_g_pix', 'l_g_fea', 'l_g_range', 'l_g_gan', 'l_d_real', 'l_d_fake','D_loss_STD','l_d_real_fake','l_g_highpass','l_g_shift_invariant',
                       'D_real', 'D_fake','D_logits_diff','psnr_val','D_update_ratio','LR_decrease','Correctly_distinguished','l_d_gp',
                       'l_e','l_g_optimalZ']+['l_g_latent_%d'%(i) for i in range(self.num_latent_channels)]
        self.log_dict = OrderedDict(zip(logs_2_keep, [[] for i in logs_2_keep]))
        if self.is_train:
            if self.latent_input:
                self.using_encoder = False # train_opt['latent_weight'] > 0 Now using 'latent_weight' parameter for the new latent input configuration
                self.latent_grads_multiplier = train_opt['lr_latent']/train_opt['lr_G'] if train_opt['lr_latent'] else 1
                self.channels_idx_4_grad_amplification = [[] for i in self.netG.parameters()]
                if train_opt['optimalZ_loss_type'] is not None and (train_opt['optimalZ_loss_weight']>0 or self.debug):
                    self.optimalZ_loss_type = train_opt['optimalZ_loss_type']
            self.D_verification = train_opt['D_verification']
            assert self.D_verification in ['current', 'convergence', 'past',None]
            if self.D_verification=='convergence':
                self.D_converged = False
            self.relativistic_D = opt['network_D']['relativistic'] is None or bool(opt['network_D']['relativistic'])
            self.add_quantization_noise = bool(opt['network_D']['add_quantization_noise'])
            self.min_accumulation_steps = min(
                [train_opt['grad_accumulation_steps_G'], train_opt['grad_accumulation_steps_D']])
            self.max_accumulation_steps = accumulation_steps_per_batch
            self.grad_accumulation_steps_G = train_opt['grad_accumulation_steps_G']
            self.grad_accumulation_steps_D = train_opt['grad_accumulation_steps_D']
            self.decomposed_output = self.CEM_arch and bool(opt['network_D']['decomposed_input'])
            self.netG.train()
            self.l_gan_w = train_opt['gan_weight']
            self.D_exists = self.l_gan_w>0 or self.debug
            if self.D_exists:
                self.netD = networks.define_D(opt,CEM=self.CEM_net).to(self.device)  # D
                self.netD.train()
            if self.using_encoder:
                self.netE = networks.define_E(input_nc=opt['network_G']['out_nc'],output_nc=1,ndf=opt['network_D']['nf'],
                                              net_type='resnet_256',gpu_ids=opt['gpu_ids']).to(self.device)
                self.netE.train()

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

            if train_opt['highpass_weight'] > 0 or self.debug:
                import sys
                sys.path.append(os.path.abspath('../../RandomPooling'))
                from highpassed_loss import HighPass_Loss
                filter = np.load('/media/ybahat/data/projects/SRGAN/experiments/MSE_srResNet/highpass_filter_Square.npz')['filter']
                self.cri_highpass = HighPass_Loss(high_pass_filter=filter).to(self.device)
                self.l_highpass_w = train_opt['highpass_weight']
            else:
                print('Remove highpass loss.')
                self.cri_highpass = None

            if train_opt['shift_invariant_weight'] > 0 or self.debug:
                import sys
                sys.path.append(os.path.abspath('../../RandomPooling'))
                from shift_invariant_loss import ShiftInvariant_Loss
                self.cri_shift_invariant = ShiftInvariant_Loss(shift_size=opt['scale']).to(self.device)
                self.l_shift_invariant_w = train_opt['shift_invariant_weight']
            else:
                print('Remove highpass loss.')
                self.cri_shift_invariant = None

            # Reference loss after optimizing latent input:
            if self.optimalZ_loss_type is not None and (train_opt['optimalZ_loss_weight'] > 0 or self.debug):
                self.l_g_optimalZ_w = train_opt['optimalZ_loss_weight']
                self.Z_optimizer = Z_optimizer(objective=self.optimalZ_loss_type,Z_size=2*[int(opt['datasets']['train']['patch_size']/(opt['scale']/self.Z_size_factor))],model=self,Z_range=1,
                    max_iters=10,initial_LR=1,batch_size=opt['datasets']['train']['batch_size'],HR_unpadder=self.CEM_net.HR_unpadder)
                if self.optimalZ_loss_type == 'l2':
                    self.cri_optimalZ = nn.MSELoss().to(self.device)
                elif self.optimalZ_loss_type == 'l1':
                    self.cri_optimalZ = nn.L1Loss().to(self.device)
                elif self.optimalZ_loss_type == 'hist':
                    self.cri_optimalZ = self.Z_optimizer.loss
                else:
                    raise NotImplementedError('Loss type [{:s}] not recognized.'.format(self.optimalZ_loss_type))
            else:
                print('Remove reference loss with optimal Z.')
                self.cri_optimalZ = None

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
                self.reshuffle_netF_weights = False
                if 'feature_pooling' in train_opt or 'feature_model_arch' in train_opt:
                    if 'feature_model_arch' not in train_opt:
                        train_opt['feature_model_arch'] = 'vgg19'
                    elif 'feature_pooling' not in train_opt:
                        train_opt['feature_pooling'] = ''
                    self.reshuffle_netF_weights = 'shuffled' in train_opt['feature_pooling']
                    train_opt['feature_pooling'] = train_opt['feature_pooling'].replace('untrained_shuffled_','untrained_').replace('untrained_shuffled','untrained')
                    saved_config_params = torch.load(os.path.join(opt['path']['models'],'F_config_params.pth')) if os.path.isfile(os.path.join(opt['path']['models'],'F_config_params.pth')) else None
                    loaded_state_dict = torch.load(train_opt['netF_checkpoint'])['state_dict'] if 'netF_checkpoint' in train_opt else None
                    if loaded_state_dict is not None:
                        print('Loaded state-dict for feature loss: ',train_opt['netF_checkpoint'])
                    self.netF = networks.define_F(opt, use_bn=False,state_dict=loaded_state_dict,arch=train_opt['feature_model_arch'],
                        arch_config=train_opt['feature_pooling'],saved_config_params=saved_config_params).to(self.device)
                else:
                    self.netF = networks.define_F(opt, use_bn=False).to(self.device)

            # Range limiting loss:
            if train_opt['range_weight'] > 0 or self.debug:
                self.cri_range = CreateRangeLoss(opt['range'])
                self.l_range_w = train_opt['range_weight']
            else:
                print('Remove range loss.')
                self.cri_range = None

            if self.D_exists:
                # GD gan loss
                self.cri_gan = GANLoss(train_opt['gan_type'], 1.0, 0.0).to(self.device)
                self.l_gan_w = train_opt['gan_weight']
                # D_update_ratio and D_init_iters are for WGAN
                self.global_D_update_ratio = train_opt['D_update_ratio'] if train_opt['D_update_ratio'] is not None else 1
                self.D_init_iters = train_opt['D_init_iters'] if train_opt['D_init_iters'] else 0
                self.E_init_iters = train_opt['E_init_iters'] if (train_opt['E_init_iters'] and self.using_encoder) else 0

                if train_opt['gan_type'] == 'wgan-gp':
                    self.random_pt = torch.Tensor(1, 1, 1, 1).to(self.device)
                    # gradient penalty loss
                    self.cri_gp = GradientPenaltyLoss(device=self.device).to(self.device)
                    self.l_gp_w = train_opt['gp_weigth']
            else:
                print('Remove GAN loss')
                self.cri_gan = None
                self.D_init_iters = 0
                self.global_D_update_ratio = 1
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
                if self.using_encoder:
                    lr_E = np.load(os.path.join(self.log_path, 'lr.npz'))['lr_E']
            else:
                lr_G = train_opt['lr_G']
                lr_D = train_opt['lr_D']
                if self.using_encoder:
                    lr_E = train_opt['lr_E']
            self.optimizer_G = torch.optim.Adam(optim_params, lr=lr_G, \
                weight_decay=wd_G, betas=(train_opt['beta1_G'], 0.999))
            self.optimizers.append(self.optimizer_G)
            # D
            if self.D_exists:
                wd_D = train_opt['weight_decay_D'] if train_opt['weight_decay_D'] else 0
                self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=lr_D, \
                    weight_decay=wd_D, betas=(train_opt['beta1_D'], 0.999))
                self.optimizers.append(self.optimizer_D)
            # E
            if self.using_encoder:
                self.optimizer_E = torch.optim.Adam(self.netE.parameters(),lr=lr_E,betas=(train_opt['beta1_D'], 0.999))
                self.optimizers.append(self.optimizer_E)
            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(lr_scheduler.MultiStepLR(optimizer, \
                        train_opt['lr_steps'], train_opt['lr_gamma']))
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')
            self.generator_step = False
            self.generator_changed = True#Initializing to true,to save the initial state```````
        elif init_Fnet or init_Dnet:
            if init_Fnet:
                self.netF = networks.define_F(opt, use_bn=False).to(self.device)
                self.netF.eval()
            if init_Dnet:
                self.netD = networks.define_D(opt,CEM=self.CEM_net).to(self.device)
                self.netD.eval()
        self.load()

        print('---------- Model initialized ------------------')
        self.print_network()
        print('-----------------------------------------------')
    def ConcatLatent(self,LR_image,latent_input):
        if latent_input is not None:
            if LR_image.size()[2:]!=latent_input.size()[2:]:
                latent_input = latent_input.contiguous().view([latent_input.size(0)]+[latent_input.size(1)*self.opt['scale']**2]+list(LR_image.size()[2:]))
            self.model_input = torch.cat([latent_input,LR_image],dim=1)
        else:
            self.model_input = 1*LR_image
    def Assing_LR_and_Latent(self,LR_image,latent_input):
        self.AssignLatent(latent_input)
        self.model_input = LR_image

    def AssignLatent(self,latent_input):
        self.netG.module.generated_image_model.Z = latent_input

    def GetLatent(self):
        latent = 1*self.model_input[:,:-3,...]
        if latent.size(1)!=self.num_latent_channels:
            latent = latent.view([latent.size(0)]+[self.num_latent_channels]+[self.opt['scale']*val for val in list(latent.size()[2:])])
        return latent

    def feed_data(self, data, need_HR=True):
        # LR
        self.var_L = data['LR'].to(self.device)
        if self.latent_input is not None:
            if 'Z' in data.keys():
                cur_Z = data['Z']
            else:
                if self.opt['network_G']['latent_channels']=='STD_directional':
                    cur_Z = Unit_Circle_rejection_Sampling(batch_size=self.var_L.size(dim=0))
                else:
                    cur_Z = torch.rand([self.var_L.size(dim=0), self.num_latent_channels, 1, 1])
                if self.opt['network_G']['latent_channels'] in ['SVD_structure_tensor','SVDinNormedOut_structure_tensor']:
                    cur_Z[:,-1,...] = 2*np.pi*cur_Z[:,-1,...]
                    self.SVD = {'theta':cur_Z[:,-1,...],'lambda0_ratio':1*cur_Z[:,0,...],'lambda1_ratio':1*cur_Z[:,1,...]}
                    cur_Z = SVD_2_LatentZ(cur_Z).detach()
                else:
                    cur_Z = 2*cur_Z-1

            if isinstance(cur_Z,int) or len(cur_Z.shape)<4 or (cur_Z.shape[2]==1 and not torch.is_tensor(cur_Z)):
                cur_Z = cur_Z*np.ones([1,self.num_latent_channels]+[self.Z_size_factor*val for val in list(self.var_L.size()[2:])])
            elif torch.is_tensor(cur_Z) and cur_Z.size(dim=2)==1:
                cur_Z = (cur_Z*torch.ones([1,1]+[self.Z_size_factor*val for val in list(self.var_L.size()[2:])])).type(self.var_L.type())
            if not torch.is_tensor(cur_Z):
                cur_Z = torch.from_numpy(cur_Z).type(self.var_L.type())
        else:
            cur_Z = None
        self.ConcatLatent(LR_image=self.var_L,latent_input=cur_Z)
        if need_HR:  # train or val
            if self.is_train and self.add_quantization_noise:
                data['HR'] += (torch.rand_like(data['HR'])-0.5)/255 # Adding quantization noise to real images to avoid discriminating based on quantization differences between real and fake
            self.var_H = data['HR'].to(self.device)

            input_ref = data['ref'] if 'ref' in data else data['HR']
            self.var_ref = input_ref.to(self.device)

    def optimize_parameters(self):
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
        # G
        if first_grad_accumulation_step_D or self.generator_step:
            G_grads_retained = True
            for p in self.netG.parameters():
                p.requires_grad = True
        else:
            G_grads_retained = False
            for p in self.netG.parameters():
                p.requires_grad = False
        actual_dual_step_steps = int(self.optimalZ_loss_type is not None and self.generator_started_learning)+1 # 2 if I actually have an optimized-Z step, 1 otherwise
        for possible_dual_step_num in range(actual_dual_step_steps):
            optimized_Z_step = possible_dual_step_num==(actual_dual_step_steps-2)#I first perform optimized Z step to avoid saving Gradients for the Z optimization, then I restore the assigned Z and perform the static Z step.
            first_dual_batch_step = possible_dual_step_num==0
            last_dual_batch_step = possible_dual_step_num==(actual_dual_step_steps-1)
            if self.CEM_net is not None and first_dual_batch_step:
                self.var_H, self.var_ref = self.CEM_net.HR_unpadder(self.var_H), self.CEM_net.HR_unpadder(self.var_ref)
            if first_dual_batch_step:
                static_Z = self.GetLatent()
            if optimized_Z_step:
                self.Z_optimizer.feed_data({'LR':self.var_L,'HR':self.var_H})
                self.Z_optimizer.optimize()
            else:
                self.ConcatLatent(LR_image=self.var_L, latent_input=static_Z)
                self.fake_H = self.netG(self.model_input)
            if self.CEM_net is not None:
                if self.decomposed_output:
                    self.fake_H = [self.CEM_net.HR_unpadder(self.fake_H[0]),self.CEM_net.HR_unpadder(self.fake_H[1])]
                else:
                    self.fake_H = self.CEM_net.HR_unpadder(self.fake_H)

            # D (and E, if exists)
            l_d_total = 0
            if not self.D_exists:
                self.generator_step = self.gradient_step_num>0 #Allow one first idle iteration to save initital validation results
            else:
                if (self.gradient_step_num) % max([1,np.ceil(1/self.cur_D_update_ratio)]) == 0 and self.gradient_step_num > -self.D_init_iters:
                    not_E_only_step = not (self.using_encoder and self.gradient_step_num<self.E_init_iters)
                    for p in self.netD.parameters():
                        p.requires_grad = not_E_only_step
                    for p in self.netG.parameters():
                        p.requires_grad = False
                    if self.using_encoder:
                        for p in self.netE.parameters():
                            p.requires_grad = True
                    if first_grad_accumulation_step_D and first_dual_batch_step:
                        if not_E_only_step:
                            self.optimizer_D.zero_grad()
                            self.l_d_real_grad_step,self.l_d_fake_grad_step,self.D_real_grad_step,self.D_fake_grad_step,self.D_logits_diff_grad_step = [],[],[],[],[]
                        if self.using_encoder:
                            self.optimizer_E.zero_grad()
                            self.l_e_grad_step = []
                    if not_E_only_step:
                        if first_dual_batch_step:
                            pred_d_real = self.netD([self.fake_H[0],self.var_ref-self.fake_H[0]] if self.decomposed_output else self.var_ref)
                        pred_d_fake = self.netD([t.detach() for t in self.fake_H] if self.decomposed_output else self.fake_H.detach())  # detach to avoid BP to G
                        if self.relativistic_D:
                            l_d_real = self.cri_gan(pred_d_real - torch.mean(pred_d_fake), True)
                            l_d_fake = self.cri_gan(pred_d_fake - torch.mean(pred_d_real), False)
                        else:
                            if first_dual_batch_step:
                                l_d_real = 2*self.cri_gan(pred_d_real, True)#Multiplying by 2 to be consistent with the SRGAN code, where losses are summed and not averaged.
                            l_d_fake = 2*self.cri_gan(pred_d_fake, False)

                        l_d_total += (l_d_real + l_d_fake) / 2

                        if self.opt['train']['gan_type'] == 'wgan-gp':
                            batch_size = self.var_ref.size(0)
                            if self.random_pt.size(0) != batch_size:
                                self.random_pt.resize_(batch_size, 1, 1, 1)
                            self.random_pt.uniform_()  # Draw random interpolation points
                            interp = self.random_pt * ((self.fake_H[0].detach()+self.fake_H[1].detach()) if self.decomposed_output else self.fake_H.detach()) + (1 - self.random_pt) * self.var_ref
                            interp.requires_grad = True
                            interp_crit = self.netD([self.fake_H[0].detach(),interp-self.fake_H[0].detach()] if self.decomposed_output else interp)
                            l_d_gp = self.l_gp_w * self.cri_gp(interp, interp_crit)  # maybe wrong in cls?
                            l_d_total += l_d_gp
                        self.l_d_real_grad_step.append(l_d_real.item())
                        self.l_d_fake_grad_step.append(l_d_fake.item())
                        self.D_real_grad_step.append(torch.mean(pred_d_real.detach()).item())
                        self.D_fake_grad_step.append(torch.mean(pred_d_fake.detach()).item())
                        self.D_logits_diff_grad_step.append(list(torch.mean(pred_d_real.detach()-pred_d_fake.detach(),dim=[d for d in range(1,pred_d_real.dim())]).data.cpu().numpy()))
                        if first_grad_accumulation_step_D and first_dual_batch_step:
                            self.generator_step = (self.gradient_step_num) % max(
                                [1, self.cur_D_update_ratio]) == 0 and self.gradient_step_num > max([self.D_init_iters,self.E_init_iters])
                            # When D batch is larger than G batch, run G iter on final D iter steps, to avoid updating G in the middle of calculating D gradients.
                            self.generator_step = self.generator_step and self.step % \
                                                  self.grad_accumulation_steps_D >= self.grad_accumulation_steps_D - self.grad_accumulation_steps_G
                            if self.generator_step:
                                if self.D_verification=='past' and self.opt['train']['D_valid_Steps_4_G_update'] > 0:
                                    self.generator_step = len(self.log_dict['D_logits_diff']) >= self.opt['train']['D_valid_Steps_4_G_update'] and \
                                        all([val[1] > np.log(self.opt['train']['min_D_prob_ratio_4_G']) for val in self.log_dict['D_logits_diff'][-self.opt['train']['D_valid_Steps_4_G_update']:]]) and \
                                        all([val[1] > self.opt['train']['min_mean_D_correct'] for val in self.log_dict['Correctly_distinguished'][-self.opt['train']['D_valid_Steps_4_G_update']:]])
                                elif self.D_verification=='convergence':
                                    if not self.D_converged and self.gradient_step_num>=self.opt['train']['steps_4_D_convergence']:
                                        std, slope = 0, 0
                                        for key in ['l_d_real', 'l_d_fake']:
                                            relevant_loss_vals = [val[1] for val in self.log_dict[key] if val[0] >= self.gradient_step_num - self.opt['train']['steps_4_loss_std']]
                                            [cur_slope, _], [[cur_var, _], _] = np.polyfit([i for i in range(len(relevant_loss_vals))],relevant_loss_vals, 1, cov=True)
                                            # We take the the standard deviation as a measure
                                            std += 0.5 * np.sqrt(cur_var)
                                            slope += 0.5 * cur_slope
                                        self.D_converged = -self.opt['train']['lr_change_ratio'] * np.minimum(-1e-5,slope) < std
                                    self.generator_step = 1*self.D_converged
                        if self.D_verification=='current' and self.generator_step:
                            self.generator_step = all([val > 0 for val in self.D_logits_diff_grad_step[-1]]) \
                                and np.mean(self.D_logits_diff_grad_step[-1])>np.log(self.opt['train']['min_D_prob_ratio_4_G'])
                        if G_grads_retained and not self.generator_step:# Freeing up the unnecessary gradients memory:
                                self.fake_H = [var.detach() for var in self.fake_H] if self.decomposed_output else self.fake_H.detach()
                        l_d_total /= (self.grad_accumulation_steps_D*actual_dual_step_steps)
                        l_d_total.backward(retain_graph=self.generator_step or (self.opt['train']['gan_type']=='wgan-gp'))

                    if self.using_encoder:
                        estimated_z = self.netE(self.fake_H.detach())
                        l_e = self.cri_latent(estimated_z,self.cur_Z.to(estimated_z.device))
                        self.l_e_grad_step.append(l_e.item())
                        l_e /= (self.grad_accumulation_steps_D*actual_dual_step_steps)
                        l_e.backward(retain_graph=self.generator_step or (self.opt['train']['gan_type']=='wgan-gp'))

                    if last_grad_accumulation_step_D and last_dual_batch_step:
                        if not_E_only_step:
                            self.optimizer_D.step()
                            # set log
                            self.log_dict['l_d_real'].append((self.gradient_step_num,np.mean(self.l_d_real_grad_step)))
                            self.log_dict['l_d_fake'].append((self.gradient_step_num,np.mean(self.l_d_fake_grad_step)))
                            self.log_dict['l_d_real_fake'].append((self.gradient_step_num,np.mean(self.l_d_fake_grad_step)+np.mean(self.l_d_real_grad_step)))
                            if self.opt['train']['gan_type'] == 'wgan-gp':
                                self.log_dict['l_d_gp'].append((self.gradient_step_num,l_d_gp.item()))
                            # D outputs
                            self.log_dict['D_real'].append((self.gradient_step_num,np.mean(self.D_real_grad_step)))
                            self.log_dict['D_fake'].append((self.gradient_step_num,np.mean(self.D_fake_grad_step)))
                            self.log_dict['D_logits_diff'].append((self.gradient_step_num,np.mean(self.D_logits_diff_grad_step)))
                            self.log_dict['Correctly_distinguished'].append((self.gradient_step_num,np.mean([val0>0 for val1 in self.D_logits_diff_grad_step for val0 in val1])))
                            self.log_dict['D_update_ratio'].append((self.gradient_step_num,self.cur_D_update_ratio))
                        if self.using_encoder:
                            self.optimizer_E.step()
                            self.log_dict['l_e'].append((self.gradient_step_num,np.mean(self.l_e_grad_step)))

            # G step:
            l_g_total = 0#torch.zeros(size=[],requires_grad=True).type(torch.cuda.FloatTensor)
            if self.generator_step:
                self.generator_started_learning = True
                if self.D_exists:
                    for p in self.netD.parameters():
                        p.requires_grad = False
                for p in self.netG.parameters():
                    p.requires_grad = True
                if self.using_encoder:
                    for p in self.netE.parameters():
                        p.requires_grad = False
                if first_grad_accumulation_step_G and first_dual_batch_step:
                    self.optimizer_G.zero_grad()
                    self.l_g_pix_grad_step,self.l_g_fea_grad_step,self.l_g_gan_grad_step,self.l_g_range_grad_step,self.l_g_latent_grad_step,self.l_g_optimalZ_grad_step = [],[],[],[],[],[]
                    self.l_g_highpass_grad_step,self.l_g_shift_invariant_grad_step = [],[]
                if self.cri_pix:  # pixel loss
                    if 'pixel_domain' in self.opt['train'] and self.opt['train']['pixel_domain']=='LR':
                        LR_size = list(self.var_L.size()[-2:])
                        l_g_pix = self.cri_pix(self.Convert_2_LR(LR_size)(self.fake_H), self.Convert_2_LR(LR_size)(self.var_H))
                    else:
                        l_g_pix = self.cri_pix((self.fake_H[0]+self.fake_H[1]) if self.decomposed_output else self.fake_H, self.var_H)
                    l_g_total += self.l_pix_w * l_g_pix/(self.grad_accumulation_steps_G*actual_dual_step_steps)
                if self.cri_highpass:  # pixel loss
                    l_g_highpass = self.cri_highpass(self.fake_H, self.var_H)
                    l_g_total += self.l_highpass_w * l_g_highpass/(self.grad_accumulation_steps_G*actual_dual_step_steps)
                if self.cri_shift_invariant:  # Shift invariant loss
                    l_g_shift_invariant = self.cri_shift_invariant(self.fake_H, self.var_H)
                    l_g_total += self.l_shift_invariant_w * l_g_shift_invariant/(self.grad_accumulation_steps_G*actual_dual_step_steps)
                if self.cri_fea:  # feature loss
                    if 'feature_domain' in self.opt['train'] and self.opt['train']['feature_domain']=='LR':
                        LR_size = list(self.var_L.size()[-2:])
                        real_fea = self.netF(self.Convert_2_LR(LR_size)(self.var_H)).detach()
                        fake_fea = self.netF(self.Convert_2_LR(LR_size)(self.fake_H))
                    else:
                        real_fea = self.netF(self.var_H).detach()
                        fake_fea = self.netF((self.fake_H[0]+self.fake_H[1]) if self.decomposed_output else self.fake_H)
                    l_g_fea = self.cri_fea(fake_fea, real_fea)
                    l_g_total += self.l_fea_w * l_g_fea/(self.grad_accumulation_steps_G*actual_dual_step_steps)
                if self.cri_range: #range loss
                    l_g_range = self.cri_range((self.fake_H[0]+self.fake_H[1]) if self.decomposed_output else self.fake_H)
                    l_g_total += self.l_range_w * l_g_range/(self.grad_accumulation_steps_G*actual_dual_step_steps)
                if self.cri_latent and last_dual_batch_step:
                    latent_loss_dict = {'SR':self.fake_H,'HR':self.var_H,'Z':static_Z}
                    if self.opt['network_G']['latent_channels'] == 'SVD_structure_tensor':
                        latent_loss_dict['SVD'] = self.SVD
                    l_g_latent = self.cri_latent(latent_loss_dict)
                    l_g_total += self.l_latent_w * l_g_latent.mean()/self.grad_accumulation_steps_G
                    self.l_g_latent_grad_step.append([l.item() for l in l_g_latent])
                if self.cri_optimalZ and first_dual_batch_step:  # optimized-Z reference image loss
                    l_g_optimalZ = self.cri_optimalZ(self.fake_H, self.var_H)
                    l_g_total += self.l_g_optimalZ_w * l_g_optimalZ/self.grad_accumulation_steps_G
                    self.l_g_optimalZ_grad_step.append(l_g_optimalZ.item())

                # G gan + cls loss
                if not self.D_exists:
                    l_g_gan = 0
                else:
                    pred_g_fake = self.netD(self.fake_H)

                    if self.relativistic_D:
                        pred_d_real = self.netD([self.fake_H[0], self.var_ref - self.fake_H[0]] if self.decomposed_output else self.var_ref).detach()
                        l_g_gan = self.l_gan_w * (self.cri_gan(pred_d_real - torch.mean(pred_g_fake), False) +
                                                  self.cri_gan(pred_g_fake - torch.mean(pred_d_real), True)) / 2/(self.grad_accumulation_steps_G*actual_dual_step_steps)
                    else:
                        l_g_gan = self.l_gan_w * self.cri_gan(pred_g_fake, True)/(self.grad_accumulation_steps_G*actual_dual_step_steps)

                l_g_total += l_g_gan
                l_g_total.backward()
                if self.cri_pix:
                    self.l_g_pix_grad_step.append(l_g_pix.item())
                if self.cri_highpass:
                    self.l_g_highpass_grad_step.append(l_g_highpass.item())
                if self.cri_shift_invariant:
                    self.l_g_shift_invariant_grad_step.append(l_g_shift_invariant.item())
                if self.cri_fea:
                    self.l_g_fea_grad_step.append(l_g_fea.item())
                if self.cri_gan:
                    self.l_g_gan_grad_step.append(l_g_gan.item())
                if self.cri_range: #range loss
                    self.l_g_range_grad_step.append(l_g_range.item())
                if last_grad_accumulation_step_G and last_dual_batch_step:
                    if self.latent_input and self.latent_grads_multiplier!=1:
                        for p_num,p in enumerate(self.netG.parameters()):
                            for channel_num in self.channels_idx_4_grad_amplification[p_num]:
                                p.grad[:,channel_num,...] *= self.latent_grads_multiplier
                    self.optimizer_G.step()
                    self.generator_changed = True
                    # set log
                    if self.cri_pix:
                        self.log_dict['l_g_pix'].append((self.gradient_step_num,np.mean(self.l_g_pix_grad_step)))
                    if self.cri_highpass:
                        self.log_dict['l_g_highpass'].append((self.gradient_step_num,np.mean(self.l_g_highpass_grad_step)))
                    if self.cri_shift_invariant:
                        self.log_dict['l_g_shift_invariant'].append((self.gradient_step_num,np.mean(self.l_g_shift_invariant_grad_step)))
                    if self.cri_fea:
                        self.log_dict['l_g_fea'].append((self.gradient_step_num,np.mean(self.l_g_fea_grad_step)))
                        if self.reshuffle_netF_weights:
                            self.netF.module._initialize_weights()
                        if self.step<=1:
                            if 'max_2_random_max_size_once' in self.opt['train']['feature_pooling']:
                                sys.path.append(os.path.abspath('../../RandomPooling'))
                                from RandomMaxArea import RandomMaxArea
                                torch.save([([ri._indices() for ri in l.random_indexes['pooling_mats']],l.random_indexes['max_areas']) for l in
                                    self.netF.module.features.children() if isinstance(l, RandomMaxArea)],
                                                       os.path.join(self.opt['path']['models'],'F_config_params.pth'))
                            elif 'patches_init_first' in self.opt['train']['feature_pooling']:
                                torch.save(next(self.netF.module.features.parameters()),os.path.join(self.opt['path']['models'],'F_config_params.pth'))
                            # elif 'patches_init_all' in self.opt['train']['feature_pooling']:
                            #     torch.save(self.netF.state_dict(),os.path.join(self.opt['path']['models'],'F_config_params.pth'))

                    if self.cri_range:
                        self.log_dict['l_g_range'].append((self.gradient_step_num,np.mean(self.l_g_range_grad_step)))
                    if self.cri_latent:
                        for channel_num in range(self.num_latent_channels):
                            self.log_dict['l_g_latent_%d'%(channel_num)].append((self.gradient_step_num, np.mean([val[channel_num] for val in self.l_g_latent_grad_step])))
                    if self.cri_optimalZ:
                        self.log_dict['l_g_optimalZ'].append((self.gradient_step_num,np.mean(self.l_g_optimalZ_grad_step)))
                    if self.cri_gan:
                        self.log_dict['l_g_gan'].append((self.gradient_step_num,np.mean(self.l_g_gan_grad_step)))
        self.step += 1

    def test(self,prevent_grads_calc=True):
        self.netG.eval()
        if prevent_grads_calc:
            with torch.no_grad():
                self.fake_H = self.netG(self.model_input)
        else:
            self.fake_H = self.netG(self.model_input)
        self.netG.train()

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
                    save_img_path = os.path.join(img_dir, '{:s}_{:d}.png'.format(img_name, self.gradient_step_num))
                    util.save_img(np.clip(sr_img, 0, 255).astype(np.uint8), save_img_path)
        if save_images:
            self.generator_changed = False
        avg_psnr = 1 * np.mean(avg_psnr)
        if SAVE_IMAGE_COLLAGE and save_images:
            save_img_path = os.path.join(os.path.join(self.opt['path']['val_images']),'{:d}_{}PSNR{:.3f}.png'.format(self.gradient_step_num,
                ('Z' + str(cur_Z)) if self.opt['network_G']['latent_input'] else '', avg_psnr))
            util.save_img(np.concatenate([np.concatenate(col, 0) for col in image_collage], 1), save_img_path)
            if save_GT_HR:  # Save GT HR images
                util.save_img(np.concatenate([np.concatenate(col, 0) for col in GT_image_collage], 1),os.path.join(os.path.join(self.opt['path']['val_images']), 'GT_HR.png'))
        print_rlt['psnr'] += avg_psnr
        return sr_images

    def update_learning_rate(self,cur_step=None):
        #The returned value is LR_too_low
        SLOPE_BASED = False
        LOSS_BASED = True
        if SLOPE_BASED:
            std,slope = 0,0
            for key in ['l_d_real','l_d_fake','l_g_gan']:
                relevant_loss_vals = [val[1] for val in self.log_dict[key] if val[0] >= cur_step - self.opt['train']['steps_4_loss_std']]
                [cur_slope, _], [[cur_var, _], _] = np.polyfit([i for i in range(len(relevant_loss_vals))],relevant_loss_vals,1, cov=True)
                # We take the the standard deviation as a measure
                std += 0.5*(0.5 if 'l_d' in key else 1.)*np.sqrt(cur_var)
                slope += 0.5*(0.5 if 'l_d' in key else 1.)*cur_slope
            reduce_lr = -self.opt['train']['lr_change_ratio']*slope<std
        elif LOSS_BASED:
            if len(self.log_dict['D_logits_diff'])>=self.opt['train']['steps_4_loss_std']:
                relevant_loss_vals = [(val[1]+self.log_dict['l_d_fake'][i][1])/2 for i,val in enumerate(self.log_dict['l_d_real']) if val[0] >= cur_step - self.opt['train']['steps_4_loss_std']]
                self.log_dict['D_loss_STD'].append([self.gradient_step_num,np.std(relevant_loss_vals)])
                reduce_lr = (self.opt['train']['std_4_lr_drop'] is not None) and self.log_dict['D_loss_STD'][-1][1]>self.opt['train']['std_4_lr_drop']
            else:
                reduce_lr = False
        else:
            relevant_D_logits_difs = [val[1] for val in self.log_dict['D_logits_diff'] if val[0] >= cur_step - self.opt['train']['steps_4_loss_std']]
            reduce_lr = np.std(relevant_D_logits_difs)>self.opt['train']['std_4_lr_drop']
        if len(self.log_dict['D_logits_diff'])<2 * self.opt['train']['steps_4_loss_std'] or \
                self.log_dict['D_logits_diff'][0][0]>cur_step-self.opt['train']['steps_4_loss_std']:#Check after a minimal number of steps
            return False
        if reduce_lr:
            if SLOPE_BASED:
                print('slope: ', slope, 'STD: ', std)
            cur_LR = [opt.param_groups[0]['lr'] for opt in self.optimizers]
            self.load(max_step=cur_step - self.opt['train']['steps_4_loss_std'],resume_train=True)
            for opt_num,optimizer in enumerate(self.optimizers):
                for param_group in optimizer.param_groups:
                    param_group['lr'] = cur_LR[opt_num]*self.opt['train']['lr_gamma']
                    if param_group['lr']<1e-8:
                        return True
            lr_decrease_dict = {'lr_G':self.optimizer_G.param_groups[0]['lr'],'lr_D':self.optimizer_D.param_groups[0]['lr']}
            print('LR(D) reduced to %.2e, LR(G) reduced to %.2e.'%(self.optimizer_D.param_groups[0]['lr'],self.optimizer_G.param_groups[0]['lr']))
            if self.using_encoder:
                np.savez(os.path.join(self.log_path,'lr.npz'),step_num=cur_step,lr_G =self.optimizer_G.param_groups[0]['lr'],lr_D =self.optimizer_D.param_groups[0]['lr'],
                         lr_E=self.optimizer_E.param_groups[0]['lr'])
                lr_decrease_dict['lr_E'] = self.optimizer_E.param_groups[0]['lr']
                print('LR(E) reduced to %.2e.' % (self.optimizer_E.param_groups[0]['lr']))
            else:
                np.savez(os.path.join(self.log_path,'lr.npz'),step_num=cur_step,lr_G =self.optimizer_G.param_groups[0]['lr'],lr_D =self.optimizer_D.param_groups[0]['lr'])
            self.log_dict['LR_decrease'].append([self.step//self.max_accumulation_steps,lr_decrease_dict])
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
        if self.cri_latent is not None and 'collected_ratios' in self.cri_latent.__dir__():
            np.savez(os.path.join(self.log_path,'collected_stats.npz'),*self.cri_latent.collected_ratios)

    def load_log(self,max_step=None):
        PREPEND_OLD_LOG = False
        loaded_log = np.load(os.path.join(self.log_path,'logs.npz'))
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
        if self.cri_latent is not None and 'collected_ratios' in self.cri_latent.__dir__():
            collected_stats = np.load(os.path.join(self.log_path,'collected_stats.npz'))
            for i,file in enumerate(collected_stats.files):
                self.cri_latent.collected_ratios[i] = deque(collected_stats[file],maxlen=self.cri_latent.collected_ratios[i].maxlen)

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

    def print_network(self):
        # Generator
        # s, n, receptive_field = self.get_network_description(self.netG)
        # print('Number of parameters in G: {:,d}. Receptive field size: ({:,d},{:,d})'.format(n, *receptive_field))
        net_desc = self.get_network_description(self.netG)
        s,n = net_desc['s'],net_desc['n']
        # # a = receptive_field(self.netG.module, input_size=(3, 256, 256))
        print('Number of parameters in G: {:,d}'.format(n))
        if self.is_train:
            message = '-------------- Generator --------------\n' + s + '\n'
            network_path = os.path.join(self.save_dir, '../', 'network.txt')
            if not self.opt['train']['resume']:
                with open(network_path, 'w') as f:
                    f.write(message)
            if self.D_exists:
                # Discriminator
                s, n,receptive_field_D = self.get_network_description(self.netD)
                print('Number of parameters in D: {:,d}. Receptive field size: {:,d}'.format(n,receptive_field_D))
                message = '\n\n\n-------------- Discriminator --------------\n' + s + '\n'
                if not self.opt['train']['resume']:
                    with open(network_path, 'a') as f:
                        f.write(message)

            if self.cri_fea:  # F, Perceptual Network
                # s, n,receptive_field = self.get_network_description(self.netF)
                # print('Number of parameters in F: {:,d}. Receptive field size: ({:,d},{:,d})'.format(n, *receptive_field))
                net_desc = self.get_network_description(self.netF)
                s, n = net_desc['s'], net_desc['n']
                # receptive_field = net_desc['receptive_field']
                print('Number of parameters in F: {:,d}'.format(n))
                message = '\n\n\n-------------- Perceptual Network --------------\n' + s + '\n'
                if not self.opt['train']['resume']:
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
                self.step = (loaded_model_step+1)*self.max_accumulation_steps
                print('Resuming training with model for G [{:s}] ...'.format(os.path.join(self.opt['path']['models'],model_name)))
                self.load_network(os.path.join(self.opt['path']['models'],model_name), self.netG,optimizer=self.optimizer_G)
                self.load_log(max_step=loaded_model_step)
                if self.D_exists:
                    model_name = str(loaded_model_step)+'_D.pth'
                    print('Resuming training with model for D [{:s}] ...'.format(os.path.join(self.opt['path']['models'],model_name)))
                    self.load_network(os.path.join(self.opt['path']['models'],model_name), self.netD,optimizer=self.optimizer_D)
                    if self.using_encoder:
                        model_name = str(loaded_model_step)+'_E.pth'
                        print('Resuming training with model for E [{:s}] ...'.format(os.path.join(self.opt['path']['models'],model_name)))
                        self.load_network(os.path.join(self.opt['path']['models'],model_name), self.netE,optimizer=self.optimizer_E)
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

    def save(self, iter_label):
        saving_path = self.save_network(self.save_dir, self.netG, 'G', iter_label,self.optimizer_G)
        if self.D_exists:
            self.save_network(self.save_dir, self.netD, 'D', iter_label,self.optimizer_D)
            if self.using_encoder:
                self.save_network(self.save_dir, self.netE, 'E', iter_label, self.optimizer_E)
        return saving_path

