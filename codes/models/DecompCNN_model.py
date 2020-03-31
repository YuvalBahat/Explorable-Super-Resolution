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
import numpy as np
from collections import deque
from Z_optimization import Z_optimizer
from utils.util import SVD_2_LatentZ
from JPEG_module.JPEG import JPEG
import tqdm
from utils import util
import cv2

USE_Y_GENERATOR_4_CHROMA = True

class DecompCNNModel(BaseModel):
    def __init__(self, opt,accumulation_steps_per_batch=None,init_Fnet=None,init_Dnet=None,**kwargs):
        super(DecompCNNModel, self).__init__(opt)
        train_opt = opt['train']
        self.log_path = opt['path']['log']
        self.latent_input = opt['network_G']['latent_input'] if opt['network_G']['latent_input']!='None' else None
        # if self.latent_input is not None:
        #     self.Z_size_factor = opt['scale'] if 'HR' in opt['network_G']['latent_input_domain'] else 1
        self.num_latent_channels = 0
        self.debug = 'debug' in opt['path']['log']
        self.chroma_mode = self.opt['name'][:len('JPEG/chroma')]=='JPEG/chroma'
        self.cri_latent = None
        self.optimalZ_loss_type = None
        self.generator_started_learning = False #I'm adding this flag to avoid wasting time optimizing over the Z space when D is still in its early learning phase. I don't change it when resuming training of a saved model - it would change by itself after 1 generator step.
        self.num_latent_channels = FilterLoss(latent_channels=opt['network_G']['latent_channels']).num_channels
        if self.latent_input is not None:
            if self.is_train:
                # Loss encouraging effect of Z:
                self.l_latent_w = train_opt['latent_weight']
                if train_opt['latent_weight']>0 or self.debug:
                    self.cri_latent = FilterLoss(latent_channels=opt['network_G']['latent_channels'],task='JPEG',gray_scale=True)
            else:
                assert isinstance(opt['network_G']['latent_channels'],int)
        # define networks and load pretrained models
        self.step = 0
        self.D_steps_since_G = 0
        self.G_steps_since_D = 0
        self.jpeg_compressor = JPEG(compress=True,chroma_mode=self.chroma_mode, downsample_and_quantize=True,block_size=self.opt['scale']).to(self.device)
        self.jpeg_extractor = JPEG(compress=False,chroma_mode=self.chroma_mode,block_size=self.opt['scale']).to(self.device)

        self.netG = networks.define_G(opt,num_latent_channels=self.num_latent_channels,chroma_mode=self.chroma_mode).to(self.device)  # G
        # print('Receptive field of G:',util.compute_RF_numerical(self.netG.module.cpu(),np.ones([1,64,64,64])))
        G_kernel_sizes = [l.kernel_size[0] for l in next(self.netG.module.children()) if isinstance(l,nn.Conv2d)]
        G_receptive_filed = G_kernel_sizes[0]+sum([k-1 for k in G_kernel_sizes[1:]])
        print('Receptive field of G: %d = 8*%d'%(8*G_receptive_filed,G_receptive_filed))
        # if train_opt['gan_type'] == 'wgan-gp':
        #     input = torch.zeros([1, self.opt['network_G']['in_nc']] + 2 * [opt['datasets']['train']['patch_size']]).to(next(self.netG.parameters()).device)
        #     self.netG.module.dncnn,_ = util.convert_batchNorm_2_layerNorm(self.netG.module.dncnn,input=input)
        self.netG.to(self.device)
        if self.chroma_mode and USE_Y_GENERATOR_4_CHROMA:
            self.netG_Y = networks.define_G(opt,num_latent_channels=self.num_latent_channels,chroma_mode=False).to(self.device).cuda()
            self.netG_Y.eval()
            self.jpeg_compressor_Y = JPEG(compress=True,chroma_mode=False, downsample_and_quantize=True,block_size=8).to(self.device)
            self.jpeg_extractor_Y = JPEG(compress=False,chroma_mode=False,block_size=8).to(self.device)
        logs_2_keep = ['l_g_pix_log_rel', 'l_g_fea', 'l_g_range', 'l_g_gan', 'l_d_real', 'l_d_fake','D_loss_STD','l_d_real_fake',
                       'D_real', 'D_fake','D_logits_diff','psnr_val','D_update_ratio','LR_decrease','Correctly_distinguished','l_d_gp',
                       'l_e','l_g_optimalZ','D_G_prob_ratio','mean_D_correct']+['l_g_latent_%d'%(i) for i in range(self.num_latent_channels)]
        self.log_dict = OrderedDict(zip(logs_2_keep, [[] for i in logs_2_keep]))
        self.avg_estimated_err = np.empty(shape=[8,8,0])
        self.avg_estimated_err_step = []
        if self.is_train:
            if self.latent_input:
                # self.latent_grads_multiplier = train_opt['lr_latent']/train_opt['lr_G'] if train_opt['lr_latent'] else 1
                # self.channels_idx_4_grad_amplification = [[] for i in self.netG.parameters()]
                if opt['train']['optimalZ_loss_type'] is not None and (opt['train']['optimalZ_loss_weight']>0 or self.debug):
                    self.optimalZ_loss_type = opt['train']['optimalZ_loss_type']
            self.D_verification = opt['train']['D_verification']
            assert self.D_verification in ['current', 'convergence', 'past','initial',None]
            self.D_verified = False
            if self.D_verification=='convergence':
                self.D_converged = False
            self.relativistic_D = opt['network_D']['relativistic'] is None or bool(opt['network_D']['relativistic'])
            self.add_quantization_noise = bool(opt['network_D']['add_quantization_noise'])
            self.min_accumulation_steps = min(
                [opt['train']['grad_accumulation_steps_G'], opt['train']['grad_accumulation_steps_D']])
            self.max_accumulation_steps = accumulation_steps_per_batch
            self.grad_accumulation_steps_G = opt['train']['grad_accumulation_steps_G']
            self.grad_accumulation_steps_D = opt['train']['grad_accumulation_steps_D']
            self.l_gan_w = train_opt['gan_weight']
            self.D_exists = self.l_gan_w>0 or self.debug
            self.DCT_discriminator = self.D_exists and self.opt['network_D']['DCT_D']
            self.concatenated_D_input = self.D_exists and self.opt['network_D']['concat_input']
            if self.D_exists:
                self.netD = networks.define_D(opt,chroma_mode=self.chroma_mode).to(self.device)  # D
                if self.DCT_discriminator:
                    self.jpeg_non_quantized_compressor = JPEG(compress=True, downsample_and_quantize=False,chroma_mode=self.chroma_mode,block_size=self.opt['scale']).to(self.device)
                if train_opt['gan_type'] == 'wgan-gp' and not self.DCT_discriminator:
                    # if self.DCT_discriminator:
                    #     input = torch.zeros([1, 64] + 2 * [opt['datasets']['train']['patch_size'] // 8]).to(next(self.netD.parameters()).device)
                    # else:
                    input = torch.zeros([1,1]+2*[opt['datasets']['train']['patch_size']]).to(next(self.netD.parameters()).device)
                    # if 'DnCNN' in str(self.netD.module.__class__):
                    #     self.netD.module,_ = util.convert_batchNorm_2_layerNorm(self.netD.module,input=input)
                    # else:
                    self.netD.module.features,input = util.convert_batchNorm_2_layerNorm(self.netD.module.features,input=input)
                    self.netD.module.classifier,_ = util.convert_batchNorm_2_layerNorm(self.netD.module.classifier,input=input)
                    self.netD.cuda()
                self.netD.train()
            self.netG.train()

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

            # Reference loss after optimizing latent input:
            if self.optimalZ_loss_type is not None and (train_opt['optimalZ_loss_weight'] > 0 or self.debug):
                self.l_g_optimalZ_w = train_opt['optimalZ_loss_weight']
                self.Z_optimizer = Z_optimizer(objective=self.optimalZ_loss_type,
                    Z_size=2*[int(opt['datasets']['train']['patch_size']/(opt['scale']))],model=self,Z_range=1,
                    max_iters=10,initial_LR=1,batch_size=opt['datasets']['train']['batch_size'],HR_unpadder=lambda x:x,jpeg_extractor=self.jpeg_extractor)
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
                self.netF = networks.define_F(opt, use_bn=False).to(self.device)

            # Range limiting loss:
            if train_opt['range_weight'] > 0 or self.debug:
                self.cri_range = CreateRangeLoss(opt['range'])
                self.l_range_w = train_opt['range_weight']
            else:
                print('Remove range loss.')
                self.cri_range = None

            # GD gan loss
            if self.D_exists:
                self.cri_gan = GANLoss(train_opt['gan_type'], 1.0, 0.0).to(self.device)
                # D_update_ratio and D_init_iters are for WGAN
                self.global_D_update_ratio = train_opt['D_update_ratio'] if train_opt['D_update_ratio'] is not None else 1
                self.D_init_iters = train_opt['D_init_iters'] if train_opt['D_init_iters'] else 0

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
            else:
                lr_G = train_opt['lr_G']
                lr_D = train_opt['lr_D']
            self.optimizer_G = torch.optim.Adam(optim_params, lr=lr_G,weight_decay=wd_G, betas=(train_opt['beta1_G'], 0.999))
            self.optimizers.append(self.optimizer_G)
            # D
            if self.D_exists:
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
        elif init_Fnet or init_Dnet:
            if init_Fnet:
                self.netF = networks.define_F(opt, use_bn=False).to(self.device)
                self.netF.eval()
            if init_Dnet:
                self.netD = networks.define_D(opt).to(self.device)
                self.netD.eval()
        self.load()

        print('---------- Model initialized ------------------')
        self.print_network()
        print('-----------------------------------------------')

    def ConcatLatent(self,Comp_image,latent_input):
        # if self.chroma_mode:
        #     Comp_image = torch.cat([Comp_image[:,0,...],Comp_image[:,1,:64,...],Comp_image[:,2,:64,...]],1)
        if latent_input is not None:
            if Comp_image.size()[2:]!=latent_input.size()[2:]:
                latent_input = latent_input.contiguous().view([latent_input.size(0)]+[latent_input.size(1)*self.opt['scale']**2]+list(Comp_image.size()[2:]))
            self.model_input = torch.cat([latent_input.type(Comp_image.type()),Comp_image],dim=1)
        else:
            self.model_input = 1*Comp_image

    def GetLatent(self):
        latent = 1*self.model_input[:,:self.num_latent_channels,...]
        # if latent.size(1)!=self.num_latent_channels:
        #     latent = latent.view([latent.size(0)]+[self.num_latent_channels]+[self.opt['scale']*val for val in list(latent.size()[2:])])
        return latent

    def feed_data(self, data, need_GT=True):
        self.QF = data['QF']
        self.jpeg_compressor.Set_QF(self.QF)
        self.jpeg_extractor.Set_QF(self.QF)
        if self.DCT_discriminator:
            self.jpeg_non_quantized_compressor.Set_QF(self.QF)
        if self.latent_input is not None:
            DCT_dims = list(np.array(data['Uncomp'].size())[2:]//8) # Spatial dimensions of the latent channel correspond to those of the Y channel DCT coefficients.
            # Z is downsampled for the chroma channels generator
            if 'Z' in data.keys():
                cur_Z = data['Z']
            else:
                cur_Z = torch.rand([data['Uncomp'].size(dim=0), self.num_latent_channels, 1, 1])
                if self.opt['network_G']['latent_channels'] in ['SVD_structure_tensor','SVDinNormedOut_structure_tensor']:
                    cur_Z[:,-1,...] = 2*np.pi*cur_Z[:,-1,...]
                    self.SVD = {'theta':cur_Z[:,-1,...],'lambda0_ratio':1*cur_Z[:,0,...],'lambda1_ratio':1*cur_Z[:,1,...]}
                    cur_Z = SVD_2_LatentZ(cur_Z).detach()
                else:
                    cur_Z = 2*cur_Z-1

            if isinstance(cur_Z,int) or len(cur_Z.shape)<4 or (cur_Z.shape[2]==1 and not torch.is_tensor(cur_Z)):
                cur_Z = cur_Z*np.ones([1,self.num_latent_channels]+DCT_dims)
            elif torch.is_tensor(cur_Z) and cur_Z.size(dim=2)==1:
                cur_Z = (cur_Z*torch.ones([1,1]+DCT_dims))#.type(self.var_Comp.type())
            if not torch.is_tensor(cur_Z):
                cur_Z = torch.from_numpy(cur_Z)#.type(self.var_Comp.type())
        else:
            cur_Z = None
        if self.chroma_mode:
            if USE_Y_GENERATOR_4_CHROMA:
                self.jpeg_compressor_Y.Set_QF(self.QF)
                self.jpeg_extractor_Y.Set_QF(self.QF)
                self.var_Comp_Y = self.jpeg_compressor_Y(data['Uncomp'][:,0,...].unsqueeze(1).to(self.device))
                self.ConcatLatent(Comp_image=self.var_Comp_Y, latent_input=cur_Z)
                self.y_channel_input = self.jpeg_extractor_Y(self.netG_Y(self.model_input)).detach()
                self.var_Comp = self.jpeg_compressor(torch.cat([self.y_channel_input,data['Uncomp'][:,1:,...].type(self.y_channel_input.type())],1))
            else:
                self.y_channel_input = data['Uncomp'][:,0,...].unsqueeze(1).to(self.device)
        if not self.chroma_mode or not USE_Y_GENERATOR_4_CHROMA:
            self.var_Comp = self.jpeg_compressor(data['Uncomp'].to(self.device))
        self.ConcatLatent(Comp_image=self.var_Comp,latent_input=cur_Z)
        if need_GT:  # train or val
            self.var_Uncomp = data['Uncomp'].to(self.device)
            input_ref = data['ref'] if 'ref' in data else data['Uncomp']
            if self.DCT_discriminator:
                input_ref = self.jpeg_non_quantized_compressor(input_ref)
                if self.chroma_mode:
                    input_ref = input_ref[:,self.opt['scale']**2:,...]
                    # Even when not in concat_inpiut mode, I'm supplying D with channel Y, so it does not need to determine realness based only on the chroma channels
                    if self.concatenated_D_input:
                        input_ref = torch.cat([self.var_Comp, input_ref], 1)
                    else:
                        input_ref = torch.cat([self.var_Comp[:,:self.opt['scale']**2,...], input_ref], 1)
                elif self.concatenated_D_input:
                    input_ref = torch.cat([self.var_Comp, input_ref], 1)
            self.var_ref = input_ref.to(self.device)

    def optimize_parameters(self):
        # self.gradient_step_num = self.step//self.max_accumulation_steps
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

            if first_dual_batch_step:
                if self.num_latent_channels>0:
                    static_Z = self.GetLatent()
                else:
                    static_Z = None
            if optimized_Z_step:
                self.Z_optimizer.feed_data({'Comp':self.var_Comp,'Uncomp':self.var_Uncomp,'QF':self.QF})
                self.Z_optimizer.optimize()
            else:
                self.ConcatLatent(Comp_image=self.var_Comp, latent_input=static_Z)
                self.fake_H = self.netG(self.model_input)
            self.fake_H_4_D = self.fake_H
            self.fake_H = self.jpeg_extractor(self.fake_H_4_D)
            if self.chroma_mode:
                self.fake_H = torch.cat([self.y_channel_input, self.fake_H], 1)
            if not self.DCT_discriminator:
                self.fake_H_4_D = self.fake_H
            if self.concatenated_D_input:
                self.fake_H_4_D = torch.cat([self.var_Comp, self.fake_H_4_D], 1)

            # D
            l_d_total = 0
            if not self.D_exists:
                self.generator_step = self.gradient_step_num>0 #Allow one first idle iteration to save initital validation results
            else:
                if ((self.gradient_step_num) % max([1,np.ceil(1/self.cur_D_update_ratio)]) == 0) and self.gradient_step_num > -self.D_init_iters:
                    if self.G_steps_since_D>0:
                        self.log_dict['D_update_ratio'].append((self.gradient_step_num, 1/self.G_steps_since_D))
                    else:
                        self.D_steps_since_G += 1
                    self.G_steps_since_D = 0
                    for p in self.netD.parameters():
                        p.requires_grad = True
                    for p in self.netG.parameters():
                        p.requires_grad = False
                    if first_grad_accumulation_step_D and first_dual_batch_step:
                        self.optimizer_D.zero_grad()
                        self.l_d_real_grad_step,self.l_d_fake_grad_step,self.D_real_grad_step,self.D_fake_grad_step,self.D_logits_diff_grad_step,self.D_G_prob_ratio_grad_step\
                            = [],[],[],[],[],[]
                    if first_dual_batch_step:
                        pred_d_real = self.netD(self.var_ref)
                    pred_d_fake = self.netD(self.fake_H_4_D.detach())  # detach to avoid BP to G
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
                        interp = self.random_pt * self.fake_H_4_D.detach() + (1 - self.random_pt) * self.var_ref
                        interp.requires_grad = True
                        interp_crit = self.netD(interp).mean(-1).mean(-1)
                        l_d_gp = self.l_gp_w * self.cri_gp(interp, interp_crit)  # maybe wrong in cls?
                        l_d_total += l_d_gp
                    self.l_d_real_grad_step.append(l_d_real.item())
                    self.l_d_fake_grad_step.append(l_d_fake.item())
                    self.D_real_grad_step.append(torch.mean(pred_d_real.detach()).item())
                    self.D_fake_grad_step.append(torch.mean(pred_d_fake.detach()).item())
                    self.D_logits_diff_grad_step.append(list(torch.mean(pred_d_real.detach()-pred_d_fake.detach(),dim=[d for d in range(1,pred_d_real.dim())]).data.cpu().numpy()))
                    # self.D_logits_diff_grad_step.append(torch.mean(pred_d_real.detach()-pred_d_fake.detach()).item())
                    if first_grad_accumulation_step_D and first_dual_batch_step:
                        self.generator_step = (self.gradient_step_num) % max(
                            [1, self.cur_D_update_ratio]) == 0 and self.gradient_step_num > self.D_init_iters
                        # When D batch is larger than G batch, run G iter on final D iter steps, to avoid updating G in the middle of calculating D gradients.
                        self.generator_step = self.generator_step and self.step % \
                                              self.grad_accumulation_steps_D >= self.grad_accumulation_steps_D - self.grad_accumulation_steps_G
                        if self.generator_step:
                            if self.D_verification in ['past','initial'] and self.opt['train']['D_valid_Steps_4_G_update'] > 0:
                                if not self.D_verified:
                                    if len(self.log_dict['D_logits_diff']) >= self.opt['train']['D_valid_Steps_4_G_update']:
                                        self.D_G_prob_ratio_grad_step.append(np.mean([np.exp(val[1]) for val in self.log_dict['D_logits_diff'][-self.opt['train']['D_valid_Steps_4_G_update']:]]))
                                        if last_grad_accumulation_step_D and last_dual_batch_step:
                                            self.log_dict['D_G_prob_ratio'].append((self.gradient_step_num, np.mean(self.D_G_prob_ratio_grad_step)))
                                        self.generator_step = all([val[1] > np.log(self.opt['train']['min_D_prob_ratio_4_G']) for val in self.log_dict['D_logits_diff'][-self.opt['train']['D_valid_Steps_4_G_update']:]])\
                                        and all([val[1] > self.opt['train']['min_mean_D_correct'] for val in self.log_dict['Correctly_distinguished'][-self.opt['train']['D_valid_Steps_4_G_update']:]])
                                    else:
                                        self.generator_step = False
                                    if self.D_verification=='initial' and self.generator_step:
                                        self.D_verified = True
                                        self.save(self.gradient_step_num,first_verified_D=True)
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
                            self.fake_H_4_D = self.fake_H_4_D.detach()
                    l_d_total /= (self.grad_accumulation_steps_D*actual_dual_step_steps)
                    l_d_total.backward(retain_graph=self.generator_step or (self.opt['train']['gan_type']=='wgan-gp'))

                    if last_grad_accumulation_step_D and last_dual_batch_step:
                        if True:
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
                            self.log_dict['D_logits_diff'].append((self.gradient_step_num,np.mean(np.concatenate(self.D_logits_diff_grad_step))))
                            self.log_dict['Correctly_distinguished'].append((self.gradient_step_num,np.mean([val0>0 for val1 in self.D_logits_diff_grad_step for val0 in val1])))
                            # self.log_dict['D_update_ratio'].append((self.gradient_step_num,self.cur_D_update_ratio))

            # G step:
            l_g_total = 0
            if self.generator_step:
                self.generator_started_learning = True
                if self.D_steps_since_G > 0:
                    self.log_dict['D_update_ratio'].append((self.gradient_step_num, self.D_steps_since_G))
                else:
                    self.G_steps_since_D += 1
                self.D_steps_since_G = 0
                if self.D_exists:
                    for p in self.netD.parameters():
                        p.requires_grad = False
                for p in self.netG.parameters():
                    p.requires_grad = True
                if first_grad_accumulation_step_G and first_dual_batch_step:
                    self.optimizer_G.zero_grad()
                    self.l_g_pix_grad_step,self.l_g_fea_grad_step,self.l_g_gan_grad_step,self.l_g_range_grad_step,self.l_g_latent_grad_step,self.l_g_optimalZ_grad_step = [],[],[],[],[],[]
                if self.cri_pix:  # pixel loss
                    l_g_pix = self.cri_pix(self.fake_H, self.var_Uncomp)
                    l_g_total += self.l_pix_w * l_g_pix/(self.grad_accumulation_steps_G*actual_dual_step_steps)
                if self.cri_fea:  # feature loss
                    real_fea = self.netF(self.var_Uncomp).detach()
                    fake_fea = self.netF(self.fake_H)
                    l_g_fea = self.cri_fea(fake_fea, real_fea)
                    l_g_total += self.l_fea_w * l_g_fea/(self.grad_accumulation_steps_G*actual_dual_step_steps)
                if self.cri_range: #range loss
                    l_g_range = self.cri_range(self.fake_H)
                    l_g_total += self.l_range_w * l_g_range/(self.grad_accumulation_steps_G*actual_dual_step_steps)
                if self.cri_latent and last_dual_batch_step:
                    latent_loss_dict = {'Decomp':self.fake_H,'Uncomp':self.var_Uncomp,'Z':static_Z}
                    if self.opt['network_G']['latent_channels'] == 'SVD_structure_tensor':
                        latent_loss_dict['SVD'] = self.SVD
                    l_g_latent = self.cri_latent(latent_loss_dict)
                    l_g_total += self.l_latent_w * l_g_latent.mean()/self.grad_accumulation_steps_G
                    self.l_g_latent_grad_step.append([l.item() for l in l_g_latent])
                if self.cri_optimalZ and first_dual_batch_step:  # optimized-Z reference image loss
                    l_g_optimalZ = self.cri_optimalZ(self.fake_H, self.var_Uncomp)
                    l_g_total += self.l_g_optimalZ_w * l_g_optimalZ/self.grad_accumulation_steps_G
                    self.l_g_optimalZ_grad_step.append(l_g_optimalZ.item())

                # G gan + cls loss
                if not self.D_exists:
                    l_g_gan = 0
                else:
                    pred_g_fake = self.netD(self.fake_H_4_D)

                    if self.relativistic_D:
                        pred_d_real = self.netD(self.var_ref).detach()
                        l_g_gan = self.l_gan_w * (self.cri_gan(pred_d_real - torch.mean(pred_g_fake), False) +
                                                  self.cri_gan(pred_g_fake - torch.mean(pred_d_real), True)) / 2/(self.grad_accumulation_steps_G*actual_dual_step_steps)
                    else:
                        l_g_gan = self.l_gan_w * self.cri_gan(pred_g_fake, True)/(self.grad_accumulation_steps_G*actual_dual_step_steps)

                l_g_total += l_g_gan
                l_g_total.backward()
                if self.cri_pix:
                    quantized_l_pix = self.cri_pix(self.jpeg_extractor(self.var_Comp), self.var_Uncomp)
                    self.l_g_pix_grad_step.append((l_g_pix/quantized_l_pix).log().item())
                if self.cri_fea:
                    self.l_g_fea_grad_step.append(l_g_fea.item())
                if self.cri_gan:
                    self.l_g_gan_grad_step.append(l_g_gan.item())
                if self.cri_range: #range loss
                    self.l_g_range_grad_step.append(l_g_range.item())
                if last_grad_accumulation_step_G and last_dual_batch_step:
                    # if self.latent_input and self.latent_grads_multiplier!=1:
                    #     for p_num,p in enumerate(self.netG.parameters()):
                    #         for channel_num in self.channels_idx_4_grad_amplification[p_num]:
                    #             p.grad[:,channel_num,...] *= self.latent_grads_multiplier
                    self.optimizer_G.step()
                    self.generator_changed = True
                    # set log
                    if self.cri_pix:
                        self.log_dict['l_g_pix_log_rel'].append((self.gradient_step_num,np.mean(self.l_g_pix_grad_step)))
                    if self.cri_fea:
                        self.log_dict['l_g_fea'].append((self.gradient_step_num,np.mean(self.l_g_fea_grad_step)))
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
                self.fake_H = self.jpeg_extractor(self.netG(self.model_input))
        else:
            self.fake_H = self.jpeg_extractor(self.netG(self.model_input))
        if self.chroma_mode:
            self.fake_H = torch.cat([self.y_channel_input,self.fake_H],1)
        self.netG.train()

    def perform_validation(self,data_loader,cur_Z,print_rlt,GT_and_quantized,save_images,collect_avg_err_est=True):
        SAVE_IMAGE_COLLAGE = True
        avg_psnr, avg_quantized_psnr = [], []
        idx = 0
        image_collage = []
        if save_images:
            num_val_images = len(data_loader.dataset)
            val_images_collage_rows = int(np.floor(np.sqrt(num_val_images)))
            while val_images_collage_rows > 1:
                if np.round(num_val_images / val_images_collage_rows) == num_val_images / val_images_collage_rows:
                    break
                val_images_collage_rows -= 1
            per_image_saved_patch = min([min(im['Uncomp'].shape[1:]) for im in data_loader.dataset]) - 2
            GT_image_collage, quantized_image_collage = [], []
        QF_images_counter = {}
        for val_data in tqdm.tqdm(data_loader):
            if save_images:
                if idx % val_images_collage_rows == 0:  image_collage.append([]);   GT_image_collage.append([]);    quantized_image_collage.append([])
            idx += 1
            QF = val_data['QF'].item()
            img_name = os.path.splitext(os.path.basename(val_data['Uncomp_path'][0]))[0]
            val_data['Z'] = cur_Z
            self.feed_data(val_data)
            self.test()
            visuals = self.get_current_visuals()
            sr_img = util.tensor2img(visuals['Decomp'], out_type=np.uint8, min_max=[0, 255],chroma_mode=self.chroma_mode)  # float32
            gt_img = util.tensor2img(visuals['Uncomp'], out_type=np.uint8, min_max=[0, 255],chroma_mode=self.chroma_mode)  # float32
            avg_psnr.append(util.calculate_psnr(sr_img, gt_img))
            if save_images:
                if SAVE_IMAGE_COLLAGE:
                    margins2crop = ((np.array(sr_img.shape[:2]) - per_image_saved_patch) / 2).astype(np.int32)
                    image_collage[-1].append(np.clip(sr_img[margins2crop[0]:-margins2crop[0], margins2crop[1]:-margins2crop[1], ...], 0,255).astype(np.uint8))
                    if GT_and_quantized:  # Save GT Uncomp images
                        GT_image_collage[-1].append(np.clip(gt_img[margins2crop[0]:-margins2crop[0], margins2crop[1]:-margins2crop[1], ...], 0,255).astype(np.uint8))
                        quantized_image = util.tensor2img(self.jpeg_extractor(self.var_Comp),out_type=np.uint8, min_max=[0, 255],chroma_mode=self.chroma_mode)
                        # quantized_image = util.tensor2img(self.jpeg_extractor(self.jpeg_compressor(val_data['Uncomp'].to(self.device))),out_type=np.uint8, min_max=[0, 255],chroma_mode=self.chroma_mode)
                        quantized_image_collage[-1].append(quantized_image[margins2crop[0]:-margins2crop[0], margins2crop[1]:-margins2crop[1], ...])
                        avg_quantized_psnr.append(util.calculate_psnr(quantized_image, gt_img))
                        quantized_image_collage[-1][-1] = cv2.putText(quantized_image_collage[-1][-1], str(QF), (0, 50),cv2.FONT_HERSHEY_PLAIN, fontScale=4.0,
                                    color=np.mod(255 / 2 + quantized_image_collage[-1][-1][:25, :25].mean(), 255),thickness=2)
                        if self.chroma_mode: # In this case cv2.putText returns cv2.Umat instead of an ndarray, so it should be converted:
                            quantized_image_collage[-1][-1] = quantized_image_collage[-1][-1].get()
                else:
                    # Save Decomp images for reference
                    img_dir = os.path.join(self.opt['path']['val_images'], img_name)
                    util.mkdir(img_dir)
                    save_img_path = os.path.join(img_dir, '{:s}_{:d}.png'.format(img_name, self.gradient_step_num))
                    util.save_img(np.clip(sr_img, 0, 255).astype(np.uint8), save_img_path)
            if QF in QF_images_counter.keys(): QF_images_counter[QF] += 1
            else:
                QF_images_counter[QF] = 1
                print_rlt['psnr_gain_QF%d' % (QF)] = 0
                if GT_and_quantized:
                    self.log_dict['per_im_psnr_baseline_QF%d' % (QF)] = [(0, 0)]
            if GT_and_quantized:
                self.log_dict['per_im_psnr_baseline_QF%d' % (QF)][0] = \
                    (0,((QF_images_counter[QF]-1)*self.log_dict['per_im_psnr_baseline_QF%d' % (QF)][0][1] + avg_quantized_psnr[-1])/QF_images_counter[QF])
        if save_images:
            self.generator_changed = False

        if False and collect_avg_err_est:#Disabled until I adapt it to the chroma case
            self.avg_estimated_err = np.concatenate([self.avg_estimated_err,np.expand_dims(self.netG.module.return_collected_err_avg(),-1)],-1)
            self.avg_estimated_err_step.append(self.gradient_step_num)
            # self.log_dict['avg_est_err'].append((self.gradient_step_num,self.netG.module.return_collected_err_avg()))
        avg_psnr = [51.14 if np.isinf(v) else v for v in avg_psnr] # Replacing inf values with PSNR corresponding to the error being the quantization error (0.5), to prevent contaminating the average
        for i, QF in enumerate(data_loader.dataset.per_index_QF):
            print_rlt['psnr_gain_QF%d' % (QF)] += (avg_psnr[i] - self.log_dict['per_im_psnr_baseline_QF%d' % (QF)][0][1])/QF_images_counter[QF]
        avg_psnr = 1 * np.mean(avg_psnr)
        if SAVE_IMAGE_COLLAGE and save_images:
            save_img_path = os.path.join(os.path.join(self.opt['path']['val_images']),'{:d}_{}PSNR{:.3f}.png'.format(self.gradient_step_num,
                ('Z' + str(cur_Z)) if self.opt['network_G']['latent_input'] else '', avg_psnr))
            util.save_img(np.concatenate([np.concatenate(col, 0) for col in image_collage], 1), save_img_path)
            if GT_and_quantized:  # Save GT Uncomp images
                util.save_img(np.concatenate([np.concatenate(col, 0) for col in GT_image_collage], 1),os.path.join(os.path.join(self.opt['path']['val_images']), 'GT_Uncomp.png'))
                avg_quantized_psnr = 1 * np.mean(avg_quantized_psnr)
                print_rlt['psnr_baseline'] = avg_quantized_psnr
                util.save_img(np.concatenate([np.concatenate(col, 0) for col in quantized_image_collage], 1),os.path.join(os.path.join(self.opt['path']['val_images']),
                                           'Quantized_PSNR{:.3f}.png'.format(avg_quantized_psnr)))
                self.log_dict['psnr_val_baseline'] = [(self.gradient_step_num, print_rlt['psnr_baseline'])]

        print_rlt['psnr'] += avg_psnr

    def update_learning_rate(self,cur_step=None):
        #The returned value is LR_too_low
        SLOPE_BASED = False
        LOSS_BASED = True
        return False#DISABLING this learning rate decreasing mechanism for now.
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
        dict_2_save = self.log_dict.copy()
        dict_2_save['D_verified'] = self.D_verified
        np.savez(os.path.join(self.log_path,'logs.npz'), ** dict_2_save)
        if self.cri_latent is not None and 'collected_ratios' in self.cri_latent.__dir__():
            np.savez(os.path.join(self.log_path,'collected_stats.npz'),*self.cri_latent.collected_ratios)
        if 'avg_estimated_err' in self.__dir__():
            np.savez(os.path.join(self.log_path,'avg_estimated_err.npz'),avg_estimated_err=self.avg_estimated_err,avg_estimated_err_step=self.avg_estimated_err_step)

    def load_log(self,max_step=None):
        PREPEND_OLD_LOG = False
        loaded_log = np.load(os.path.join(self.log_path,'logs.npz'))
        if PREPEND_OLD_LOG:
            old_log = np.load(os.path.join(self.log_path, 'old_logs.npz'))
        self.log_dict = OrderedDict([val for val in zip(self.log_dict.keys(),[[] for i in self.log_dict.keys()])])
        for key in loaded_log.files:
            if key=='psnr_val':
                self.log_dict[key] = ([tuple(val) for val in old_log[key]] if PREPEND_OLD_LOG else [])+[tuple(val) for val in loaded_log[key]]
            elif key=='D_verified':
                self.D_verified = bool(loaded_log[key])
                continue
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
        if os.path.isfile(os.path.join(self.log_path,'avg_estimated_err.npz')):
            self.avg_estimated_err = np.load(os.path.join(self.log_path,'avg_estimated_err.npz'))['avg_estimated_err']
            self.avg_estimated_err_step = list(np.load(os.path.join(self.log_path,'avg_estimated_err.npz'))['avg_estimated_err_step'])

    def get_current_visuals(self, need_Uncomp=True,entire_batch=False):
        out_dict = OrderedDict()
        if entire_batch:
            out_dict['Comp'] = self.var_Comp.detach().float().cpu()
            out_dict['Decomp'] = (self.fake_H[0]+self.fake_H[1] if isinstance(self.fake_H,list) else self.fake_H).detach().float().cpu()
            if need_Uncomp:
                out_dict['Uncomp'] = self.var_Uncomp.detach().float().cpu()
        else:
            out_dict['Comp'] = self.var_Comp.detach()[0].float().cpu()
            out_dict['Decomp'] = self.fake_H.detach()[0].float().cpu()
            if need_Uncomp:
                out_dict['Uncomp'] = self.var_Uncomp.detach()[0].float().cpu()
        return out_dict

    def print_network(self):
        # Generator
        # s, n, receptive_field = self.get_network_description(self.netG)
        # print('Number of parameters in G: {:,d}. Receptive field size: ({:,d},{:,d})'.format(n, *receptive_field))
        net_desc = self.get_network_description(self.netG)
        s,n = net_desc['s'],net_desc['n']
        print('Number of parameters in G: {:,d}'.format(n))
        if self.is_train:
            message = '-------------- Generator --------------\n' + s + '\n'
            network_path = os.path.join(self.save_dir, '../', 'network.txt')
            if not self.opt['train']['resume']:
                with open(network_path, 'w') as f:
                    f.write(message)

            # Discriminator
            if self.cri_gan:
                net_desc = self.get_network_description(self.netD)
                s, n = net_desc['s'], net_desc['n']
                receptive_field = net_desc['receptive_field']
                print('Number of parameters in D: {:,d}. Receptive field size: {:,d}'.format(n, receptive_field))
                message = '\n\n\n-------------- Discriminator --------------\n' + s + '\n'
                if not self.opt['train']['resume']:
                    with open(network_path, 'a') as f:
                        f.write(message)

            if self.cri_fea:  # F, Perceptual Network
                s, n,receptive_field = self.get_network_description(self.netF)
                print('Number of parameters in F: {:,d}. Receptive field size: ({:,d},{:,d})'.format(n, *receptive_field))
                # s, n = self.get_network_description(self.netF)
                # print('Number of parameters in F: {:,d}'.format(n))
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
            if self.opt['is_train'] and load_path_D is not None and self.D_exists:
                print('loading model for D [{:s}] ...'.format(load_path_D))
                self.load_network(load_path_D, self.netD,optimizer=self.optimizer_D)
        if self.chroma_mode and USE_Y_GENERATOR_4_CHROMA:
            print('loading model for G of channel Y [{:s}] ...'.format(self.opt['path']['Y_channel_model_G']))
            self.load_network(self.opt['path']['Y_channel_model_G'], self.netG_Y)

    def save(self, iter_label,first_verified_D=False):
        if first_verified_D:
            self.save_network(self.save_dir, self.netD, 'D_verified', iter_label, self.optimizer_D)
            return
        if self.D_exists:
            self.save_network(self.save_dir, self.netD, 'D', iter_label,self.optimizer_D)
        saving_path = self.save_network(self.save_dir, self.netG, 'G', iter_label,self.optimizer_G)
        return saving_path

