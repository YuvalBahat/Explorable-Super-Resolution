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
from data.util import rgb2ycbcr
from copy import deepcopy
from skvideo.measure import niqe

USE_Y_GENERATOR_4_CHROMA = True
NO_HIGH_FREQ_CHROMA_RECONSTRUCTION = True # If True, reconstructing only 8x8 DCT coefficients for each 16x16 block in the chroma mode, to comply with our re-modeling of the JPEG pipeline, which implicitly assumes there are no high-frequency chroma content, meaning there is no aliasing when compressing chroma channels, meaning all DCT coefficients but the upper left 8x8 block are zero.
ADDITIONALLY_SAVED_ATTRIBUTES = ['D_verified','verified_D_saved','lr_G','lr_D']

class DecompCNNModel(BaseModel):
    def __init__(self, opt,accumulation_steps_per_batch=None,init_Fnet=None,init_Dnet=None,chroma_mode=False,**kwargs):
        super(DecompCNNModel, self).__init__(opt)
        if self.opt['network_G']['DCT_G'] is None: #Legacy support
            self.opt['network_G']['DCT_G'] = 1
        train_opt = opt['train']
        self.log_path = opt['path']['log']
        self.latent_input = opt['network_G']['latent_input'] if opt['network_G']['latent_input']!='None' else None
        if self.latent_input is not None:
            self.Z_size_factor = 1
        self.num_latent_channels = 0
        self.debug = 'debug' in opt['path']['log']
        self.chroma_mode = chroma_mode
        self.cri_latent = None
        self.optimalZ_loss_type = None
        self.generator_started_learning = False #I'm adding this flag to avoid wasting time optimizing over the Z space when D is still in its early learning phase. I don't change it when resuming training of a saved model - it would change by itself after 1 generator step.
        self.num_latent_channels = FilterLoss(latent_channels=opt['network_G']['latent_channels']).num_channels
        if isinstance(opt['network_G']['latent_channels'],str):
            opt['network_G']['latent_channels'] = opt['network_G']['latent_channels'].replace('_%d'%(self.num_latent_channels),'')
        if self.latent_input is not None:
            if self.is_train:
                # Loss encouraging effect of Z:
                self.l_latent_w = train_opt['latent_weight']
                if train_opt['latent_weight']>0 or self.debug:
                    self.cri_latent = FilterLoss(latent_channels=opt['network_G']['latent_channels'],task='JPEG',gray_scale=not self.chroma_mode)
            else:
                assert isinstance(opt['network_G']['latent_channels'],int)
        # define networks and load pretrained models
        self.step = 0
        # self.D_steps_since_G = 0
        # self.G_steps_since_D = 0
        self.JPEG = {'compressor':JPEG(compress=True,chroma_mode=self.chroma_mode, downsample_or_quantize=True,block_size=self.opt['scale']).to(self.device),
                     'extractor':JPEG(compress=False,chroma_mode=self.chroma_mode,block_size=self.opt['scale']).to(self.device)}

        self.netG = networks.define_G(opt,num_latent_channels=self.num_latent_channels,chroma_mode=self.chroma_mode,
                                      no_high_freq_chroma_reconstruction=NO_HIGH_FREQ_CHROMA_RECONSTRUCTION).to(self.device)  # G
        # print('Receptive field of G:',util.compute_RF_numerical(self.netG.module.cpu(),np.ones([1,64,64,64])))
        G_kernel_sizes = [l.kernel_size[0] for l in next(self.netG.module.children()) if isinstance(l,nn.Conv2d)]
        G_receptive_filed = G_kernel_sizes[0]+sum([k-1 for k in G_kernel_sizes[1:]])
        print('Receptive field of G: %d = 8*%d'%(8*G_receptive_filed,G_receptive_filed))
        self.netG.to(self.device)
        if self.chroma_mode and USE_Y_GENERATOR_4_CHROMA:
            netG_Y_opt = deepcopy(opt)
            if 'network_G_Y' in netG_Y_opt.keys():
                for key in netG_Y_opt['network_G_Y'].keys():
                    netG_Y_opt['network_G'][key] = netG_Y_opt['network_G_Y'][key]
            self.netG_Y = networks.define_G(netG_Y_opt,num_latent_channels=self.num_latent_channels,chroma_mode=False).to(self.device).cuda()
            self.netG_Y.eval()
            self.JPEG['compressor_Y'] = JPEG(compress=True,chroma_mode=False, downsample_or_quantize=True,block_size=8).to(self.device)
            self.JPEG['extractor_Y'] = JPEG(compress=False,chroma_mode=False,block_size=8).to(self.device)
        logs_2_keep = ['l_g_pix_log_rel', 'l_g_fea', 'l_g_range', 'l_g_gan', 'l_d_real', 'l_d_fake','D_loss_STD','l_d_real_fake',
                       'D_real', 'D_fake','D_logits_diff','psnr_val','D_update_ratio','LR_decrease','Correctly_distinguished','l_d_gp',
                       'l_e','l_g_optimalZ','D_G_prob_ratio','mean_D_correct','Z_effect','post_train_D_diff','G_step_D_gain','niqe_val',
                       'per_pix_STD_val','clamped_portion']+['l_g_latent_%d'%(i) for i in range(self.num_latent_channels)]
        self.log_dict = OrderedDict(zip(logs_2_keep, [[] for i in logs_2_keep]))
        self.avg_estimated_err = np.empty(shape=[8,8,0])
        self.avg_estimated_err_step = []
        self.DCT_generator = self.opt['network_G']['DCT_G']
        self.G_margins = self.netG.module.margins
        if self.is_train:
            self.batch_size = self.opt['datasets']['train']['batch_size']
            if self.latent_input:
                # self.latent_grads_multiplier = train_opt['lr_latent']/train_opt['lr_G'] if train_opt['lr_latent'] else 1
                # self.channels_idx_4_grad_amplification = [[] for i in self.netG.parameters()]
                if opt['train']['optimalZ_loss_type'] is not None and (opt['train']['optimalZ_loss_weight'] or self.debug):
                    self.optimalZ_loss_type = opt['train']['optimalZ_loss_type']
            self.D_verification = opt['train']['D_verification']
            assert self.D_verification in ['current', 'convergence', 'past','initial','initial_gradual',None]
            self.D_verified, self.verified_D_saved = self.D_verification is None,self.D_verification is None
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
            self.Z_injected_2_D = self.D_exists and self.latent_input and self.opt['network_D']['inject_Z']
            if self.D_exists:
                assert self.opt['train']['G_Dbatch_separation'] in [None,'No','SameD','SeparateBatch']
                self.netD = networks.define_D(opt,chroma_mode=self.chroma_mode,no_high_freq_chroma_reconstruction=NO_HIGH_FREQ_CHROMA_RECONSTRUCTION).to(self.device)  # D
                if False and 'gp' in train_opt['gan_type'] and not self.DCT_discriminator:#Stopped replacing batch-norm layers with layer-norm layers, as I'm checking one last time WGAN-GP, and supporting this replacement was too much of a head-ache...
                    input = torch.zeros([1,1]+2*[opt['datasets']['train']['patch_size']]).to(next(self.netD.parameters()).device)
                    if self.opt['network_D']['pooling_no_fc']:#Patch-GAN:
                        self.netD.module.dncnn,input = util.convert_batchNorm_2_layerNorm(self.netD.module.dncnn,input=input)
                    else:
                        self.netD.module.features,input = util.convert_batchNorm_2_layerNorm(self.netD.module.features,input=input)
                        self.netD.module.classifier,_ = util.convert_batchNorm_2_layerNorm(self.netD.module.classifier,input=input)
                    self.netD.cuda()
                self.netD.train()
            self.netG.train()
            self.mixed_Y_4_training = self.D_exists and self.chroma_mode
            self.separate_G_D_batches_counter = util.Counter(2 if (self.D_exists and self.opt['train']['G_Dbatch_separation'] =='SeparateBatch') else 1)
        else:
            self.DCT_discriminator,self.mixed_Y_4_training,self.D_exists = False,False,False
        if True or self.DCT_discriminator or not self.DCT_generator: # Now always constructing a non-quantized compressor for to assess GT DCT errors as part of creating the quantization error visualization
            self.JPEG['non_quantized_compressor' + ('_Y' if self.chroma_mode else '')] = JPEG(compress=True, downsample_or_quantize=False,chroma_mode=False, block_size=8).to(self.device)
            if self.chroma_mode:
                self.JPEG['non_quantized_compressor'] = JPEG(compress=True,
                    downsample_or_quantize='downsample_only' if NO_HIGH_FREQ_CHROMA_RECONSTRUCTION else False,chroma_mode=True, block_size=self.opt['scale']).to(
                    self.device)
        # self.mixed_Y_4_training = self.D_exists and self.is_train and self.chroma_mode
        # define losses, optimizer and scheduler
        if self.is_train:
            # G pixel loss
            if isinstance(train_opt['pixel_weight'],list) or train_opt['pixel_weight'] > 0 or self.debug:
                l_pix_type = train_opt['pixel_criterion']
                if l_pix_type == 'l1':
                    self.cri_pix = nn.L1Loss().to(self.device)
                elif l_pix_type == 'l2':
                    self.cri_pix = nn.MSELoss().to(self.device)
                else:
                    raise NotImplementedError('Loss type [{:s}] not recognized.'.format(l_pix_type))
                if isinstance(train_opt['pixel_weight'], list):
                    self.l_pix_w = util.Varying_weight(train_opt['pixel_weight'][0],train_opt['pixel_weight'][1])
                else:
                    self.l_pix_w = lambda x : train_opt['pixel_weight']
            else:
                print('Remove pixel loss.')
                self.cri_pix = None

            # Reference loss after optimizing latent input:
            if self.optimalZ_loss_type is not None and opt['network_G']['latent_channels']>0 and (train_opt['optimalZ_loss_weight'] > 0 or self.debug):
                self.l_g_optimalZ_w = train_opt['optimalZ_loss_weight']
                if self.opt['train']['Num_Z_iterations'] is None: #legacy support
                    self.opt['train']['Num_Z_iterations'] = [10]
                if not isinstance(self.opt['train']['Num_Z_iterations'],list):
                    self.opt['train']['Num_Z_iterations'] = list(self.opt['train']['Num_Z_iterations'])
                # Dividing the batch size in 2 for the case of training a chroma Discriminator. See explanation for self.Y_channel_is_fake
                self.Z_optimizer = Z_optimizer(objective=self.optimalZ_loss_type,Z_size=2*[int(opt['datasets']['train']['patch_size']/(8 if self.DCT_generator else 1))],
                    model=self,Z_range=1,max_iters=self.opt['train']['Num_Z_iterations'][0],initial_LR=1,batch_size=opt['datasets']['train']['batch_size']//(2 if self.chroma_mode and self.D_exists else 1),
                    HR_unpadder=lambda x:x,jpeg_extractor=self.JPEG['extractor'])
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
                self.cri_range = CreateRangeLoss([16,240] if self.chroma_mode else [16,235],chroma_mode=self.chroma_mode)
                self.l_range_w = train_opt['range_weight']
            else:
                print('Remove range loss.')
                self.cri_range = None

            # GD gan loss
            # self.GD_update_controller = None
            if self.D_exists:
                self.cri_gan = GANLoss(train_opt['gan_type'], 1.0, 0.0).to(self.device)
                # D_update_ratio and D_init_iters are for WGAN
                self.global_D_update_ratio = train_opt['D_update_ratio'] if train_opt['D_update_ratio'] is not None else 1

                # if self.auto_GD_update_ratio:
                self.GD_update_controller = util.G_D_updates_controller(intervals_values=self.global_D_update_ratio)
                self.auto_GD_update_ratio = isinstance(self.global_D_update_ratio, list)
                if self.auto_GD_update_ratio:
                    assert self.grad_accumulation_steps_D==self.grad_accumulation_steps_G,'Different batch sizes for G and D not supported with automatic controller.'
                    assert self.separate_G_D_batches_counter.max_val==1,'Separate G-D batches policy not supported with automatic controller.'
                # else:
                #     self.GD_update_controller = util.G_D_updates_controller()
                self.D_init_iters = train_opt['D_init_iters'] if train_opt['D_init_iters'] else 0

                if 'gp' in train_opt['gan_type']:
                    self.random_pt = torch.Tensor(1, 1, 1, 1).to(self.device)
                    # gradient penalty loss
                    self.cri_gp = GradientPenaltyLoss(device=self.device).to(self.device)
                    self.l_gp_w = train_opt['gp_weight']
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
            # if os.path.isfile(os.path.join(self.log_path,'lr.npz')):
            #     lr_G = np.load(os.path.join(self.log_path,'lr.npz'))['lr_G']
            #     lr_D = np.load(os.path.join(self.log_path, 'lr.npz'))['lr_D']
            # else:
            self.lr_G = train_opt['lr_G']
            self.lr_D = train_opt['lr_D']
            self.optimizer_G = torch.optim.Adam(optim_params, lr=self.lr_G,weight_decay=wd_G, betas=(train_opt['beta1_G'],
                train_opt['beta2_G'] if train_opt['beta2_G'] is not None else 0.999))
            self.optimizers.append(self.optimizer_G)
            # D
            if self.D_exists:
                wd_D = train_opt['weight_decay_D'] if train_opt['weight_decay_D'] else 0
                self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=self.lr_D, \
                    weight_decay=wd_D, betas=(train_opt['beta1_D'], train_opt['beta2_D'] if train_opt['beta2_D'] is not None else 0.999))
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
        if self.is_train:
            self.D_verified,self.verified_D_saved = bool(self.D_verified),bool(self.verified_D_saved)
            if self.D_exists:
                for param_group in self.optimizer_D.param_groups:
                    param_group['lr'] = self.lr_D
                if self.verified_D_saved:# When already started utilizing the adversarial loss term, using the same lr for both D and G and using a different number of Z-iterations:
                    self.lr_G = 1*self.lr_D
                    if 'Z_optimizer' in self.__dict__.keys(): #If MAP loss is calculated:
                        self.Z_optimizer.max_iters = self.opt['train']['Num_Z_iterations'][-1]
            for param_group in self.optimizer_G.param_groups:
                param_group['lr'] = self.lr_G

        print('---------- Model initialized ------------------')
        self.print_network()
        print('-----------------------------------------------')

    def Output_Batch(self,within_0_1):
        if within_0_1:
            # return torch.clamp(self.JPEG['extractor'](self.fake_H) / 255, 0, 1)
            if self.output_image.size(1) == 3:
                return torch.clamp(util.Tensor_YCbCR2RGB(self.output_image/ 255) , 0, 1)
            else:
                return torch.clamp(self.output_image / 255, 0, 1)
        else:
            # return self.JPEG['extractor'](self.fake_H)
            return self.output_image

    def Enforce_Consistency(self,input_im,inconsostent_output):
        if self.DCT_generator:
            return inconsostent_output
        else:
            chroma_input = input_im.shape[1] == 3 # If chroma_input, enforcing consistency ONLY ON CHROMA CHANNELS!
            use_Y_JPEG_interface = self.chroma_mode and not chroma_input
            # assert not chroma_input,'Unsupported yet'
            input_DCT = self.JPEG['compressor_Y'](input_im) if use_Y_JPEG_interface else self.JPEG['compressor'](input_im)
            inconsistent_DCT = self.JPEG['non_quantized_compressor_Y'](inconsostent_output) if use_Y_JPEG_interface else self.JPEG['non_quantized_compressor'](inconsostent_output)
            if chroma_input:
                Y_DCT = inconsistent_DCT[:,:256,...]
                inconsistent_DCT = inconsistent_DCT[:,256:,...]
                input_DCT = input_DCT[:,256:,...]
            assert (input_DCT.round()==input_DCT).all(),'Expected to get integer DCT coefficients for input image'
            EPSILON = 0.00001 # Using this EPSILON value to avoid rounding issues that may cause output image to be inconsistent with the input JPEG code, once re-compressed. (e.g. the round to nearest even issue: https://en.wikipedia.org/wiki/Rounding#Round_half_to_even)
            consistent_DCT = input_DCT+torch.clamp(inconsistent_DCT-input_DCT,-0.5+EPSILON,0.5-EPSILON)
            # Removing the following consistecy check after having verified consistency enforcing works:
            # assert ((input_DCT-self.JPEG['compressor'](self.JPEG['extractor'](consistent_DCT))).abs()==0).all().item(),'Output is inconsistent with the input JPEG code. Something went wrong... (probably rounding issues)'
            if chroma_input:
                consistent_DCT = torch.cat([Y_DCT,consistent_DCT],1)
            return self.JPEG['extractor_Y'](consistent_DCT) if use_Y_JPEG_interface else self.JPEG['extractor'](consistent_DCT)

    def Prepare_Input(self,im_input,latent_input,compressed_input=False):
        if compressed_input:
            # assert self.DCT_generator,'Unsupported yet'
            self.var_Comp = im_input
        else:
            chroma_input = im_input.size(1) == 3
            if self.chroma_mode and not chroma_input:
                self.var_Comp = self.JPEG['compressor_Y'](im_input)
            else:
                self.var_Comp = self.JPEG['compressor'](im_input)
            if not self.DCT_generator:
                if self.chroma_mode and not chroma_input:
                    self.var_Comp = self.JPEG['extractor_Y'](self.var_Comp)
                else:
                    self.var_Comp = self.JPEG['extractor'](self.var_Comp)
        if latent_input is not None and latent_input.numel()>0:
            if self.var_Comp.size()[2:]!=latent_input.size()[2:]:
                assert np.all(2*np.array(self.var_Comp.size()[2:])==np.array(latent_input.size()[2:])),'dimensions are not as expected' #Should happen due to differences in block size between chroma and Y models - There are twice as many blocks in Y, so spatial dimensions should be doubled.
                latent_input = nn.functional.interpolate(latent_input, size=self.var_Comp.size()[2:], mode='bilinear', align_corners=True)
            self.model_input = torch.cat([latent_input.type(self.var_Comp.type()),self.var_Comp],dim=1)
        else:
            self.model_input = 1*self.var_Comp

    def GetLatent(self):
        latent = 1*self.model_input[:,:self.num_latent_channels,...]
        return latent

    def Repeat_Z_3_channels(self,input_Z):
        if self.num_latent_channels>3:
            return torch.cat([input_Z.repeat([1, self.num_latent_channels // 3, 1, 1]), input_Z[:, :self.num_latent_channels % 3, ...]], 1)
        else:
            return input_Z

    def Z_2_three_channels(self,multi_channel_Z):
        if self.num_latent_channels > 3:
            ndims = multi_channel_Z.dim()
            if ndims==3:
                multi_channel_Z = multi_channel_Z.unsqueeze(0)
            input_Z_size = np.array(multi_channel_Z.size())
            multi_channel_Z = 1*nn.functional.pad(multi_channel_Z,(0,0,0,0,0,int(np.ceil(input_Z_size[1]/3)*3-input_Z_size[1])))
            multi_channel_Z = multi_channel_Z.view(input_Z_size[0],-1,3,input_Z_size[2],input_Z_size[3])
            num_repeats = input_Z_size[1]//3
            per_channel_normalizer = num_repeats+((input_Z_size[1]/3-num_repeats)*torch.tensor([3.,2.,1.])>=1)
            t_2_return = multi_channel_Z.sum(1)/per_channel_normalizer.view(1,3,1,1).type(multi_channel_Z.type())
            if ndims==3:
                t_2_return = t_2_return.squeeze(0)
            return t_2_return
        else:
            return multi_channel_Z

    def Enforce_pair_Consistency(self,compressed_im,desired_im):
        def im_2_tensor(im):
            return torch.from_numpy(rgb2ycbcr(255*im,only_y=False).transpose((2,0,1))).unsqueeze(0)
        compressed_im,desired_im = im_2_tensor(compressed_im),im_2_tensor(desired_im)
        compressed_Y_DCT = self.JPEG['compressor_Y'](compressed_im[:,0,...].unsqueeze(1))
        def Consistent_Correction(desired_coeffs,constraining_coeffs):
            return torch.clamp(desired_coeffs-constraining_coeffs,-0.5,0.5)+constraining_coeffs
        Y = Consistent_Correction(self.JPEG['compressor_Y_non_quantized'](desired_im[:,0,...].unsqueeze(1)),compressed_Y_DCT)
        Y = self.JPEG['extractor_Y'](Y)
        compressed_chroma_DCT = self.JPEG['compressor'](compressed_im)
        num_coeffs = self.JPEG['compressor'].block_size
        chroma = self.JPEG['compressor_non_quantized'](desired_im).view(1,3,num_coeffs//8,8,num_coeffs//8,8,compressed_chroma_DCT.size(2),compressed_chroma_DCT.size(3))
        chroma[0, 1, 0, :, 0, ...] = Consistent_Correction(chroma[0, 1, 0, :, 0, ...],
            compressed_chroma_DCT[0, num_coeffs ** 2:num_coeffs ** 2 + 8 ** 2,...].view(8, 8, compressed_chroma_DCT.size(2),compressed_chroma_DCT.size(3)))
        chroma[0, 2, 0, :, 0, ...] = Consistent_Correction(chroma[0, 2, 0, :, 0, ...],
            compressed_chroma_DCT[0, num_coeffs ** 2+ 8 ** 2:num_coeffs ** 2 + 2*(8 ** 2),...].view(8, 8, compressed_chroma_DCT.size(2),compressed_chroma_DCT.size(3)))
        chroma = chroma.view(1,3*num_coeffs**2,compressed_chroma_DCT.size(2),compressed_chroma_DCT.size(3))
        chroma = self.JPEG['extractor'](chroma[:,num_coeffs**2:,...])
        return torch.clamp(util.Tensor_YCbCR2RGB(torch.cat([Y,chroma],1))/255,0,1)[0].data.cpu().numpy().transpose((1,2,0))

    def feed_data(self, data, need_GT=True,detach_Y=True,mixed_Y=False):
        self.QF = data['QF']
        for module in self.JPEG.values():
            module.Set_Q_Table(self.QF)
        # self.JPEG['compressor'].Set_Q_Table(self.QF)
        # self.JPEG['extractor'].Set_Q_Table(self.QF)
        # if self.DCT_discriminator or not self.DCT_generator:
        #     self.JPEG['non_quantized_compressor'].Set_Q_Table(self.QF)
        #     if self.chroma_mode:
        #         self.JPEG['non_quantized_compressor_Y'].Set_Q_Table(self.QF)
        if self.latent_input is not None:
            input_size = np.array(data['Uncomp'].size()) if 'Uncomp' in data.keys() else [1,1,8,8]*np.array(data['Comp'].size())
            DCT_dims = list(input_size[2:]//8 if self.DCT_generator else input_size[2:])
            # DCT_dims = list(np.array(self.var_Comp.size())[2:]) # Spatial dimensions of the latent channel correspond to those of the Y channel DCT coefficients.
            # Z is downsampled for the chroma channels generator
            if 'Z' in data.keys():
                cur_Z = data['Z']
            else:
                SPATIALLY_UNIFORM_SAMPLED_Z = self.cri_latent is not None
                if SPATIALLY_UNIFORM_SAMPLED_Z:
                    cur_Z = torch.rand([input_size[0], self.num_latent_channels, 1, 1])
                else:
                    cur_Z = torch.rand([input_size[0], self.num_latent_channels]+DCT_dims)
                if self.opt['network_G']['latent_channels'] in ['SVD_structure_tensor','SVDinNormedOut_structure_tensor']:
                    cur_Z = cur_Z[:,:3,...]
                    cur_Z[:,-1,...] = 2*np.pi*cur_Z[:,-1,...]
                    self.SVD = {'theta':cur_Z[:,-1,...],'lambda0_ratio':1*cur_Z[:,0,...],'lambda1_ratio':1*cur_Z[:,1,...]}
                    cur_Z = SVD_2_LatentZ(cur_Z).detach()
                    cur_Z =self.Repeat_Z_3_channels(cur_Z)
                else:
                    cur_Z = 2*cur_Z-1

            if isinstance(cur_Z,int) or isinstance(cur_Z,float) or len(cur_Z.shape)<4 or (cur_Z.shape[2]==1 and not torch.is_tensor(cur_Z)):
                cur_Z = cur_Z*np.ones([1,self.num_latent_channels]+DCT_dims)
            elif torch.is_tensor(cur_Z) and cur_Z.size(dim=2)==1:
                cur_Z = (cur_Z*torch.ones([1,1]+DCT_dims))#.type(self.var_Comp.type())
            if not torch.is_tensor(cur_Z):
                cur_Z = torch.from_numpy(cur_Z)#.type(self.var_Comp.type())
            self.Y_channel_is_fake = torch.ones([self.QF.size(0)]).byte()
        else:
            cur_Z = None
        if 'Comp' in data.keys():
            self.var_Comp = data['Comp']
            self.Prepare_Input(self.var_Comp, latent_input=cur_Z,compressed_input=True)
        else:
            if self.chroma_mode:
                for key in ['compressor_Y','extractor_Y']:
                    self.JPEG[key].Set_Q_Table(self.QF)
                # self.JPEG['compressor_Y'].Set_Q_Table(self.QF)
                # self.JPEG['extractor_Y'].Set_Q_Table(self.QF)
                GT_Y_channel = data['Uncomp'][:,0,...].unsqueeze(1)
                self.Prepare_Input(GT_Y_channel,cur_Z)
                self.test_Y(detach=detach_Y,prevent_grads_calc=detach_Y) # Use detach_Y=False here when, e.g., computing gradients with respect to Z, which should take into account the path going through netG_Y as well, so it should not be detached.
                if self.mixed_Y_4_training and mixed_Y:#When training a chroma Discriminator (D), I want to prevent it from distinguishing based on the Y channel. To this end, Y channel of fake batches is a mix of real Y channels and the output of the Y generator, with arbitrary 1:1 ratio.
                    # self.Y_channel_is_fake = torch.ones([self.batch_size]).byte()
                    self.Y_channel_is_fake[torch.randperm(self.batch_size)[:self.batch_size//2]] = 0
                    self.y_channel_input[~self.Y_channel_is_fake] = GT_Y_channel[~self.Y_channel_is_fake].cuda()
                data['Uncomp'][:,0,...] = self.y_channel_input.squeeze(1)
            self.Prepare_Input(data['Uncomp'],latent_input=cur_Z)
        if need_GT:  # train or val
            if self.is_train and self.add_quantization_noise:
                assert not self.chroma_mode,'Unsupported yet'
                data['Uncomp'] = torch.clamp(data['Uncomp']+torch.rand_like(data['Uncomp'])-0.5,16,235) # Adding quantization noise to real images to avoid discriminating based on quantization differences between real and fake
            self.var_Uncomp = data['Uncomp'].to(self.device)
            input_ref = data['ref'] if 'ref' in data else data['Uncomp']
            if self.DCT_discriminator:
                input_ref = self.JPEG['non_quantized_compressor'](input_ref)
                if self.chroma_mode:
                    input_ref = input_ref[:,self.opt['scale']**2:,...]
                    # Even when not in concat_inpiut mode, I'm supplying D with channel Y, so it does not need to determine realness based only on the chroma channels
                    if self.concatenated_D_input:
                        input_ref = torch.cat([self.var_Comp, input_ref], 1)
                    else:
                        input_ref = torch.cat([self.var_Comp[:,:self.opt['scale']**2,...], input_ref], 1)
                elif self.concatenated_D_input:
                    input_ref = torch.cat([self.var_Comp, input_ref], 1)
                if self.Z_injected_2_D:
                    input_ref = torch.cat([cur_Z.type(input_ref.type()), input_ref], 1)
            elif self.D_exists and self.concatenated_D_input:
                input_ref = torch.cat([self.var_Comp, input_ref.to(self.device)], 1)
            self.var_ref = input_ref.to(self.device)

    def Prepare_D_input(self,fake_H):
        self.D_fake_input = fake_H
        if not self.DCT_discriminator:
            if self.chroma_mode:
                # raise Exception('Unsupported yet')
                self.D_fake_input = torch.cat([self.output_image[:,:1,...],torch.clamp(self.output_image[:,1:,...], min=16, max=240)],1)
            else:
                self.D_fake_input = torch.clamp(self.output_image,min=16,max=235)
        if self.concatenated_D_input:
            self.D_fake_input = torch.cat([self.var_Comp, self.D_fake_input], 1)
        if self.Z_injected_2_D:
            self.D_fake_input = torch.cat([self.GetLatent().type(self.D_fake_input.type()), self.D_fake_input], 1)

    def optimize_parameters(self):
        Z_OPTIMIZATION_WHEN_D_UNVERIFIED = len(self.opt['train']['Num_Z_iterations'])>1
        SINGLE_D_UPDATE_UNTIL_FIRST_G_STEP = True
        first_grad_accumulation_step_G = self.step%self.grad_accumulation_steps_G==0
        last_grad_accumulation_step_G = self.step % self.grad_accumulation_steps_G == (self.grad_accumulation_steps_G-1)
        first_grad_accumulation_step_D = self.step%self.grad_accumulation_steps_D==0
        last_grad_accumulation_step_D = self.step % self.grad_accumulation_steps_D == (self.grad_accumulation_steps_D-1)

        if first_grad_accumulation_step_G:
            G_step_batch = self.separate_G_D_batches_counter.max_val == 1 or self.separate_G_D_batches_counter.counter == 1
            self.generator_step = False
            if G_step_batch:
                self.generator_step = self.gradient_step_num > self.D_init_iters
                if self.generator_step and self.D_exists:
                    # if self.auto_GD_update_ratio:
                    self.generator_step = self.GD_update_controller.Step_query(True)
                    # else:
                    #     self.generator_step = self.gradient_step_num % max([1, self.global_D_update_ratio]) == 0
                    #     # When D batch is larger than G batch, run G iter on final D iter steps, to avoid updating G in the middle of calculating D gradients.
                    #     self.generator_step = self.generator_step and self.step % self.grad_accumulation_steps_D >= self.grad_accumulation_steps_D - self.grad_accumulation_steps_G

        # self.discriminator_step = False
        if self.D_exists and first_grad_accumulation_step_D:
            D_step_batch = self.separate_G_D_batches_counter.counter == 0
            if D_step_batch:
                self.discriminator_step = self.gradient_step_num >= -self.D_init_iters
                if self.discriminator_step:
                    # if self.GD_update_controller is None:
                    #     if SINGLE_D_UPDATE_UNTIL_FIRST_G_STEP and not self.verified_D_saved:
                    #         self.discriminator_step = True
                    #     else:
                    #         self.discriminator_step = self.gradient_step_num % max([1, np.ceil(1 / self.global_D_update_ratio)]) == 0
                    # else:
                    self.discriminator_step = self.GD_update_controller.Step_query(False)

        # if first_grad_accumulation_step_D or self.generator_step:
        #     G_grads_retained = True
        #     self.Set_Require_Grad_Status(self.netG,True)
        # else:
        #     G_grads_retained = False
        #     self.Set_Require_Grad_Status(self.netG, False)
        performing_optimized_Z_steps = self.cri_optimalZ is not None and (self.generator_started_learning or Z_OPTIMIZATION_WHEN_D_UNVERIFIED)
        performing_non_optimized_Z_steps = any([getattr(self,a) is not None for a in ['cri_gan','cri_pix','cri_latent','cri_fea']])
        actual_dual_step_steps = int(performing_optimized_Z_steps)+int(performing_non_optimized_Z_steps) # 2 if I actually have an optimized-Z step, 1 otherwise
        self.separate_G_D_batches_counter.advance()
        for possible_dual_step_num in range(actual_dual_step_steps):
            # optimized_Z_step = possible_dual_step_num==(actual_dual_step_steps-2)#I first perform optimized Z step to avoid saving Gradients for the Z optimization, then I restore the assigned Z and perform the static Z step.
            optimized_Z_step = performing_optimized_Z_steps and possible_dual_step_num==0 #I first perform optimized Z step to avoid saving Gradients for the Z optimization, then I restore the assigned Z and perform the static Z step.
            first_dual_batch_step = possible_dual_step_num==0
            last_dual_batch_step = possible_dual_step_num==(actual_dual_step_steps-1)

            if first_dual_batch_step:
                if self.num_latent_channels>0:
                    static_Z = self.GetLatent()
                else:
                    static_Z = None
            if optimized_Z_step:
                stored_QF = 1*self.QF
                if self.chroma_mode:
                    data_dict = {'Uncomp':self.var_Uncomp[self.Y_channel_is_fake],
                             'desired':torch.clamp(util.Tensor_YCbCR2RGB(self.var_Uncomp[self.Y_channel_is_fake]/255),0,1),'QF':self.QF[self.Y_channel_is_fake]}
                else:
                    data_dict = {'Uncomp': self.var_Uncomp[self.Y_channel_is_fake],'desired': torch.clamp(self.var_Uncomp[self.Y_channel_is_fake] / 255, 0, 1),'QF': self.QF[self.Y_channel_is_fake]}

                self.Z_optimizer.feed_data(data_dict)
                if self.mixed_Y_4_training:
                    # In this case, fake_H has half the batch_size images, so the rest of the images (whose Y channel is real) should be loaded back.
                    Y_channel_is_fake_copy = 1*self.Y_channel_is_fake #Y_channel_is_fake is going to be overidden during Z optimization
                    optimized_Z = self.Z_optimizer.optimize()
                    self.Y_channel_is_fake = Y_channel_is_fake_copy
                    temp_Z = torch.zeros_like(optimized_Z).repeat([2,1,1,1])
                    temp_Z[self.Y_channel_is_fake] = optimized_Z
                    self.feed_data({'Uncomp':self.var_Uncomp,'QF':stored_QF,'Z':temp_Z}, need_GT=False)
                    self.fake_H = self.Enforce_Consistency(self.var_Comp,self.netG(self.model_input))
                else:
                    self.Z_optimizer.optimize()
            else:
                self.Prepare_Input(self.var_Comp, latent_input=static_Z,compressed_input=True)
                self.fake_H = self.Enforce_Consistency(self.var_Comp,self.netG(self.model_input))
            self.output_image = self.JPEG['extractor'](self.fake_H) if self.DCT_generator else self.fake_H
            if self.chroma_mode and self.DCT_generator:
                self.output_image = torch.cat([self.y_channel_input, self.output_image], 1)
            if self.D_exists:
                self.Prepare_D_input(self.fake_H)

            # D
            l_d_total = 0
            if not self.D_exists:
                self.generator_step = self.gradient_step_num>0 #Allow one first idle iteration to save initital validation results
                self.generator_Z_opt_only_step = 1*self.generator_step
            else:
                if self.discriminator_step or self.D_verification is not None:
                    if self.discriminator_step:
                        # if first_grad_accumulation_step_D and self.GD_update_controller is not None:
                        if first_grad_accumulation_step_D:
                            self.GD_update_controller.Step_performed(False)
                        self.Set_Require_Grad_Status(self.netD, True)
                        self.Set_Require_Grad_Status(self.netG, False)
                        if first_grad_accumulation_step_D and first_dual_batch_step:
                            self.optimizer_D.zero_grad()
                    if first_dual_batch_step:
                        pred_d_real = self.netD(self.var_ref)
                        if first_grad_accumulation_step_D:
                            self.l_d_real_grad_step, self.l_d_fake_grad_step, self.D_real_grad_step, self.D_fake_grad_step, self.D_logits_diff_grad_step, self.D_G_prob_ratio_grad_step,self.clamped_portion_grad_step \
                                = [], [], [], [], [], [],[]
                    pred_d_fake = self.netD(self.D_fake_input.detach())  # detach to avoid BP to G
                    if self.opt['train']['G_Dbatch_separation']=='SameD':
                        pred_g_fake = 1*pred_d_fake
                    if self.discriminator_step:
                        if self.relativistic_D:
                            assert self.opt['train']['hinge_threshold'] is None,'Unsupported yet, should think whether it reuires special adaptation of hinge loss'
                            # assert 'hinge' not in self.cri_gan.gan_type,'Unsupported yet, should think whether it reuires special adaptation of hinge loss'
                            l_d_real = self.cri_gan(pred_d_real - torch.mean(pred_d_fake), True)
                            l_d_fake = self.cri_gan(pred_d_fake - torch.mean(pred_d_real), False)
                        else:
                            if first_dual_batch_step:
                                l_d_real = 2*self.cri_gan(pred_d_real, True,self.opt['train']['hinge_threshold'])#Multiplying by 2 to be consistent with the SRGAN code, where losses are summed and not averaged.
                                # l_d_real = 2*self.cri_gan(pred_d_real, True,'hinge' in self.cri_gan.gan_type)#Multiplying by 2 to be consistent with the SRGAN code, where losses are summed and not averaged.
                            # l_d_fake = 2*self.cri_gan(pred_d_fake, False,'hinge' in self.cri_gan.gan_type)
                            l_d_fake = 2*self.cri_gan(pred_d_fake, False,self.opt['train']['hinge_threshold'])

                        l_d_total += (l_d_real + l_d_fake) / 2

                        if 'gp' in self.opt['train']['gan_type'] and self.l_gp_w>0:
                        # if self.opt['train']['gan_type'] == 'wgan-gp' and self.l_gp_w>0:
                            batch_size = self.var_ref.size(0)
                            if self.random_pt.size(0) != batch_size:
                                self.random_pt.resize_(batch_size, 1, 1, 1)
                            self.random_pt.uniform_()  # Draw random interpolation points
                            interp = self.random_pt * self.D_fake_input.detach() + (1 - self.random_pt) * self.var_ref
                            interp.requires_grad = True
                            interp_crit = self.netD(interp).mean(-1).mean(-1)
                            l_d_gp = self.l_gp_w * self.cri_gp(interp, interp_crit)  # maybe wrong in cls?
                            l_d_total += l_d_gp
                        self.l_d_real_grad_step.append(-1*l_d_real.item())
                        self.l_d_fake_grad_step.append(l_d_fake.item())
                        if self.opt['train']['hinge_threshold'] is not None:
                            self.clamped_portion_grad_step.append(0.5*((pred_d_real>self.opt['train']['hinge_threshold']).float().mean()+(pred_d_fake<-1*self.opt['train']['hinge_threshold']).float().mean()).item())
                    self.D_real_grad_step.append(torch.mean(pred_d_real.detach()).item())
                    self.D_fake_grad_step.append(torch.mean(pred_d_fake.detach()).item())
                    # self.D_logits_diff_grad_step.append(list(torch.mean(pred_d_real.detach()-pred_d_fake.detach(),dim=[d for d in range(1,pred_d_real.dim())]).data.cpu().numpy()))
                    self.D_logits_diff_grad_step.append(list((pred_d_real.detach()-pred_d_fake.detach()).view(-1).data.cpu().numpy()))
                    if first_grad_accumulation_step_D and first_dual_batch_step:
                        self.generator_Z_opt_only_step = Z_OPTIMIZATION_WHEN_D_UNVERIFIED and self.generator_step
                        if self.generator_step:
                            if self.D_verification in ['past','initial','initial_gradual'] and self.opt['train']['D_valid_Steps_4_G_update'] > 0:
                                if not self.D_verified:
                                    if len(self.log_dict['D_logits_diff']) >= self.opt['train']['D_valid_Steps_4_G_update']:
                                        self.D_G_prob_ratio_grad_step.append(
                                            np.mean([np.exp(val[1]) for val in self.log_dict['D_logits_diff'][-self.opt['train']['D_valid_Steps_4_G_update']:]]))
                                        if last_grad_accumulation_step_D and last_dual_batch_step:
                                            self.log_dict['D_G_prob_ratio'].append((self.gradient_step_num, np.mean(self.D_G_prob_ratio_grad_step)))
                                        self.generator_step = \
                                            all([val[1] > np.log(self.opt['train']['min_D_prob_ratio_4_G']) for val in self.log_dict['D_logits_diff'][-self.opt['train']['D_valid_Steps_4_G_update']:]])\
                                        and all([val[1] > self.opt['train']['min_mean_D_correct'] for val in self.log_dict['Correctly_distinguished'][-self.opt['train']['D_valid_Steps_4_G_update']:]])
                                    else:
                                        self.generator_step = False
                                    if self.generator_step:
                                        if not self.verified_D_saved:
                                            self.save(self.gradient_step_num,first_verified_D=True)  # D was approved in the current step
                                            # When already started utilizing the adversarial loss term, using the same lr for both D and G and using a different number of Z-iterations:
                                            self.lr_G = 1 * self.lr_D
                                            for param_group in self.optimizer_G.param_groups:
                                                param_group['lr'] = self.lr_G
                                            self.verified_D_saved = True
                                            if 'Z_optimizer' in self.__dict__.keys():
                                                self.Z_optimizer.max_iters = self.opt['train']['Num_Z_iterations'][-1]
                                            self.save_log()
                                        if self.D_verification=='initial':
                                            self.D_verified = True
                                        elif self.D_verification=='initial_gradual':
                                            GRAD_WIN_LENGTH_FACTOR = 100
                                            # Not saving D's state when D is verified because here this happens only after G had changed from its original state, so I cannot use this D for other models of G
                                            if len(self.log_dict['l_g_gan'])>=GRAD_WIN_LENGTH_FACTOR:
                                                self.D_verified = \
                                                    np.mean([val[1] for val in
                                                    self.log_dict['D_logits_diff'][-(GRAD_WIN_LENGTH_FACTOR*self.opt['train']['D_valid_Steps_4_G_update']):]])\
                                                    > np.log(self.opt['train']['min_D_prob_ratio_4_G']) and \
                                                    np.mean([val[1] for val in
                                                     self.log_dict['Correctly_distinguished'][-(GRAD_WIN_LENGTH_FACTOR * self.opt['train']['D_valid_Steps_4_G_update']):]])\
                                                    > self.opt['train']['min_mean_D_correct']
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
                            elif self.D_verification=='current':
                            # if self.D_verification=='current' and self.generator_step:
                                # self.generator_step = all([val > 0 for val in self.D_logits_diff_grad_step[-1]]) \
                                #     and np.mean(self.D_logits_diff_grad_step[-1])>np.log(self.opt['train']['min_D_prob_ratio_4_G'])
                                self.generator_step = (np.mean(self.D_logits_diff_grad_step[-1]) > np.log(self.opt['train']['min_D_prob_ratio_4_G']))\
                                    and (np.mean([val > 0 for val in self.D_logits_diff_grad_step[-1]])>=self.opt['train']['min_mean_D_correct'])
                            if not self.generator_step:
                            # if G_grads_retained and not self.generator_step:# Freeing up the unnecessary gradients memory and forcing controller to switch to D step, if generator step was canceled since D was not verified:
                            #     self.D_fake_input = self.D_fake_input.detach()
                                # if self.GD_update_controller is not None:
                                print('G step skipped since D was not verified at step %d'%self.gradient_step_num)
                                self.GD_update_controller.Force_D_step()
                                # G_grads_retained = False
                    if self.discriminator_step:
                        l_d_total /= (self.grad_accumulation_steps_D*actual_dual_step_steps)
                        l_d_total.backward(retain_graph=self.generator_step or ('wgan' in self.opt['train']['gan_type']))

                    if last_grad_accumulation_step_D and last_dual_batch_step:
                        if self.discriminator_step:
                            self.optimizer_D.step()
                            # set log
                            self.log_dict['l_d_real'].append((self.gradient_step_num,np.mean(self.l_d_real_grad_step)))
                            self.log_dict['l_d_fake'].append((self.gradient_step_num,np.mean(self.l_d_fake_grad_step)))
                            self.log_dict['clamped_portion'].append((self.gradient_step_num,np.mean(self.clamped_portion_grad_step)))
                            self.log_dict['l_d_real_fake'].append((self.gradient_step_num,np.mean(self.l_d_fake_grad_step)+np.mean(self.l_d_real_grad_step)))
                            if 'gp' in self.opt['train']['gan_type'] and self.l_gp_w>0:
                                self.log_dict['l_d_gp'].append((self.gradient_step_num,l_d_gp.item()))
                        # D outputs
                        self.log_dict['D_real'].append((self.gradient_step_num,np.mean(self.D_real_grad_step)))
                        self.log_dict['D_fake'].append((self.gradient_step_num,np.mean(self.D_fake_grad_step)))
                        self.log_dict['D_logits_diff'].append((self.gradient_step_num,np.mean(np.concatenate(self.D_logits_diff_grad_step))))
                        min_outliers_threshold = util.Min_Outliers_Threshold(torch.cat([pred_d_real.view(-1),pred_d_fake.view(-1)]),torch.cat([1*torch.ones_like(pred_d_real.view(-1)),-1*torch.ones_like(pred_d_fake.view(-1))]))
                        self.log_dict['Correctly_distinguished'].append((self.gradient_step_num,(pred_d_real>min_outliers_threshold).float().mean().item()))
                        # self.log_dict['Correctly_distinguished'].append((self.gradient_step_num,np.mean([val0>0 for val1 in self.D_logits_diff_grad_step for val0 in val1])))

            # G step:
            if self.generator_step or self.generator_Z_opt_only_step:
                l_g_total = 0
                self.generator_started_learning = True
                if self.D_exists:
                    self.Set_Require_Grad_Status(self.netD, False)
                self.Set_Require_Grad_Status(self.netG, True)
                if first_grad_accumulation_step_G and first_dual_batch_step:
                    # if self.GD_update_controller is not None:
                    if self.D_exists:
                        self.log_dict['D_update_ratio'].append((self.gradient_step_num, self.GD_update_controller.Query_update_ratio()))
                    self.optimizer_G.zero_grad()
                    self.l_g_pix_grad_step,self.l_g_fea_grad_step,self.l_g_gan_grad_step,self.l_g_range_grad_step,\
                        self.l_g_latent_grad_step,self.l_g_optimalZ_grad_step,self.Z_effect_grad_step = [],[],[],[],[],[],[]
                if optimized_Z_step:
                    l_g_optimalZ = self.cri_optimalZ(self.output_image, self.var_Uncomp)
                    l_g_total += self.l_g_optimalZ_w * l_g_optimalZ/self.grad_accumulation_steps_G
                    self.l_g_optimalZ_grad_step.append(l_g_optimalZ.item())
                    self.Z_effect_grad_step.append(np.diff(self.Z_optimizer.loss_values)[0])
                if self.generator_step:
                    # if first_grad_accumulation_step_G and first_dual_batch_step and self.GD_update_controller is not None:
                    if self.D_exists and first_grad_accumulation_step_G and first_dual_batch_step:
                        self.GD_update_controller.Step_performed(True)
                    if self.cri_pix and not optimized_Z_step:  # pixel loss
                        l_g_pix = self.cri_pix(self.output_image, self.crop_2_output_shape(self.var_Uncomp,DCT=False))
                        l_g_total += self.l_pix_w(self.gradient_step_num) * l_g_pix/(self.grad_accumulation_steps_G*actual_dual_step_steps)
                    if self.cri_fea and not optimized_Z_step:  # feature loss
                        real_fea = self.netF(self.var_Uncomp).detach()
                        fake_fea = self.netF(self.output_image)
                        l_g_fea = self.cri_fea(fake_fea, real_fea)
                        l_g_total += self.l_fea_w * l_g_fea/(self.grad_accumulation_steps_G*actual_dual_step_steps)
                    if self.cri_range: #range loss
                        l_g_range = self.cri_range(self.output_image)
                        l_g_total += self.l_range_w * l_g_range/(self.grad_accumulation_steps_G*actual_dual_step_steps)
                    # if self.cri_latent and last_dual_batch_step:
                    if self.cri_latent and not optimized_Z_step:
                        latent_loss_dict = {'Decomp':self.output_image,'Uncomp':self.var_Uncomp,'Z':static_Z}
                        if self.opt['network_G']['latent_channels'] == 'SVD_structure_tensor':
                            latent_loss_dict['SVD'] = self.SVD
                        l_g_latent = self.cri_latent(latent_loss_dict)[self.Y_channel_is_fake]
                        l_g_total += self.l_latent_w * l_g_latent.mean()/self.grad_accumulation_steps_G
                        self.l_g_latent_grad_step.append([l.item() for l in l_g_latent.mean(0)])

                    # G gan + cls loss
                    if not self.D_exists:
                        l_g_gan = 0
                    else:
                        if self.opt['train']['G_Dbatch_separation']!='SameD':
                            pred_g_fake = self.netD(self.D_fake_input)

                        if self.relativistic_D:
                            pred_d_real = self.netD(self.var_ref).detach()
                            l_g_gan = (self.cri_gan(pred_d_real - torch.mean(pred_g_fake), False) +
                                                      self.cri_gan(pred_g_fake - torch.mean(pred_d_real), True)) / 2/(self.grad_accumulation_steps_G*actual_dual_step_steps)
                        else:
                            l_g_gan = self.cri_gan(pred_g_fake, True)/(self.grad_accumulation_steps_G*actual_dual_step_steps)

                    l_g_total += self.l_gan_w *l_g_gan
                else:
                    last_dual_batch_step = True # When self.generator_step==False but self.generator_Z_opt_only_step==True, I don't have the second part of the generator step without the Z-optimization, so optimization step and logs saving should be done in the first part.
                    self.generator_Z_opt_only_step = False#Avoid entering the generator step part in the second dual-batch step, because there is no loss there so optimizer step() and backward() commands shoudn't be called
                l_g_total.backward()
                if self.generator_step:
                    if not optimized_Z_step:
                        if self.cri_pix:
                            quantized_l_pix = self.cri_pix(self.crop_2_output_shape(self.JPEG['extractor'](self.var_Comp),DCT=False) if self.DCT_generator else self.var_Comp, self.crop_2_output_shape(self.var_Uncomp,DCT=False))
                            self.l_g_pix_grad_step.append((l_g_pix/quantized_l_pix).log().item())
                        if self.cri_fea:
                            self.l_g_fea_grad_step.append(l_g_fea.item())
                    if self.cri_gan:
                        self.l_g_gan_grad_step.append(l_g_gan.item())
                    if self.cri_range: #range loss
                        self.l_g_range_grad_step.append(l_g_range.item())
                if last_grad_accumulation_step_G and last_dual_batch_step:
                    self.optimizer_G.step()
                    self.generator_changed = True
                    # set log
                    if self.generator_step:
                        if self.cri_pix:
                            self.log_dict['l_g_pix_log_rel'].append((self.gradient_step_num,np.mean(self.l_g_pix_grad_step)))
                        if self.cri_fea:
                            self.log_dict['l_g_fea'].append((self.gradient_step_num,np.mean(self.l_g_fea_grad_step)))
                        if self.cri_range:
                            self.log_dict['l_g_range'].append((self.gradient_step_num,np.mean(self.l_g_range_grad_step)))
                        if self.cri_latent:
                            for channel_num in range(3):
                            # for channel_num in range(self.num_latent_channels):
                                self.log_dict['l_g_latent_%d'%(channel_num)].append((self.gradient_step_num, np.mean([val[channel_num] for val in self.l_g_latent_grad_step])))
                        if self.cri_gan:
                            self.log_dict['l_g_gan'].append((self.gradient_step_num,np.mean(self.l_g_gan_grad_step)))
                            # if self.GD_update_controller is not None or self.gradient_step_num%self.opt['train']['val_freq']==0: # Following Tamar's idea, recomputing G's output after its training step, to see if it is able to follow D:
                            if True:
                                # with torch.no_grad():
                                # assert not self.chroma_mode,'Unsupported yet'
                                self.test()
                                self.Prepare_D_input(self.fake_H)
                                # self.Prepare_D_input(self.netG(self.model_input))
                                # I'm performing averaging in two steps to allow measuring the correctly distinguished portion in the future:
                                post_G_step_D_scores = self.netD(self.D_fake_input.detach()).detach()
                                if self.opt['train']['G_Dbatch_separation'] != 'SeparateBatch': # I can't compute this without pred_d_real (hence the discriminator_step condition), and it doesn't make sense to compare with pred_d_real in this case, because it was computed on a different batch.
                                    if not self.discriminator_step:
                                        pred_d_real = self.netD(self.var_ref)
                                    self.log_dict['post_train_D_diff'].append((self.gradient_step_num,np.mean([v.item() for v in list(
                                        torch.mean(pred_d_real.detach() - post_G_step_D_scores,
                                                   dim=[d for d in range(1, pred_d_real.dim())]).data.cpu().numpy())])))
                                if self.opt['train']['G_Dbatch_separation'] != 'SameD': #It doesn't make sense to compare with pred_g_fake in this case, because it was computed with D prior its update.
                                    self.log_dict['G_step_D_gain'].append((self.gradient_step_num,np.mean([v.item() for v in list(
                                        torch.mean(post_G_step_D_scores-pred_g_fake.detach(),
                                                   dim=[d for d in range(1, pred_g_fake.dim())]).data.cpu().numpy())])))
                    if self.cri_gan and self.auto_GD_update_ratio:
                        self.GD_update_controller.Update_ratio(np.mean([v[1] for v in self.log_dict[self.opt['train']['D_update_measure']]\
                            if (v[0]>=self.gradient_step_num-self.opt['train']['steps_4_loss_std'] and v[0]>self.opt['train']['D_init_iters'])]))

                    if self.cri_optimalZ:
                        self.log_dict['l_g_optimalZ'].append((self.gradient_step_num,np.mean(self.l_g_optimalZ_grad_step)))
                        self.log_dict['Z_effect'].append((self.gradient_step_num, np.mean(self.Z_effect_grad_step)))
        self.step += 1

    def test_Y(self,prevent_grads_calc=True,**kwargs):
        if prevent_grads_calc:
            with torch.no_grad():
                self.test_Y_(**kwargs)
        else:
            self.test_Y_(**kwargs)

    def test_Y_(self,detach=False):
        self.y_channel_input = self.netG_Y(self.model_input)
        self.y_channel_input = torch.clamp(self.JPEG['extractor_Y'](self.y_channel_input) if self.DCT_generator else
            self.Enforce_Consistency(self.var_Comp,self.y_channel_input),0,255)
        if detach:
            self.y_channel_input = self.y_channel_input.detach()

    def test_(self,uncompressed_chroma=None,detach_Y=False,chroma_Z=None):
        self.netG.eval()
        # if self.chroma_mode and not Y_already_computed:
        if uncompressed_chroma is not None:
            self.test_Y(detach=detach_Y,prevent_grads_calc=False) #Passing prevent_grads_calc=False because this the decision to calc grads is already taken when test_ was called.
            if uncompressed_chroma.size(0)!=self.y_channel_input.size(0):
                uncompressed_chroma = uncompressed_chroma.repeat([self.y_channel_input.size(0)]+[1]*(uncompressed_chroma.ndimension()-1))
            self.Prepare_Input(torch.cat([self.y_channel_input,uncompressed_chroma],1),self.GetLatent() if chroma_Z is None else chroma_Z)
        self.fake_H = self.Enforce_Consistency(self.var_Comp,self.netG(self.model_input))
        self.output_image = self.JPEG['extractor'](self.fake_H) if self.DCT_generator else self.fake_H
        if self.chroma_mode and self.DCT_generator:
            self.output_image = torch.cat([self.y_channel_input,self.output_image],1)
        self.netG.train()

    def Return_Compressed(self,uncompressed):
        chroma_input = uncompressed.size(1)==3
        assert self.chroma_mode or not chroma_input,'Got a color image when model is not supporting it'
        if self.chroma_mode:
            Y_channel = self.JPEG['extractor_Y'](self.JPEG['compressor_Y'](uncompressed[:,0,...].unsqueeze(1)))
            if not chroma_input:
                return Y_channel
            return self.JPEG['extractor'](self.JPEG['compressor'](torch.cat([Y_channel,uncompressed[:,1:,...]],1)))
        else:
            return self.JPEG['extractor'](self.JPEG['compressor'](uncompressed))

    def test(self,prevent_grads_calc=True,**kwargs):
        if prevent_grads_calc:
            with torch.no_grad():
                self.test_(**kwargs)
        else:
            self.test_(**kwargs)

    def reset_err_collection(self):
        self.average_err_collection_counter = 0
        self.average_abs_err_estimates = np.zeros([8, 8])
        self.average_abs_GT_err = np.zeros([8, 8])

    def return_collected_err_avg(self,first_eval):
        if first_eval:
            self.err_normalizer = self.average_abs_GT_err/self.average_err_collection_counter
        return self.average_abs_err_estimates/self.average_err_collection_counter/self.err_normalizer

    def crop_2_output_shape(self,image,DCT):
        if self.G_margins==0:
            return image
        margins = self.G_margins if DCT else self.opt['scale']*self.G_margins
        if isinstance(image,torch.Tensor):
            assert image.dim()==4
            return image[...,margins:-margins,margins:-margins]

        else:#Numpy
            if image.ndim==3 and image.shape[2] in [1,3] and not DCT:
                return image[margins:-margins,margins:-margins,:]

    def perform_validation(self,data_loader,cur_Z,print_rlt,first_eval,save_images,collect_avg_err_est=True):
        SAVE_IMAGE_COLLAGE = True
        avg_psnr, avg_quantized_psnr,avg_niqe,avg_quantized_niqe,avg_GT_niqe = [], [],[],[],[]
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
        chroma_mode = 'YCbCr' if self.chroma_mode else 'Y'
        if collect_avg_err_est:
            # assert not self.chroma_mode,'Unsupported yet'
            self.reset_err_collection()
            # self.netG.module.reset_err_collectiion()
        for val_data in tqdm.tqdm(data_loader):
            if save_images:
                if idx % val_images_collage_rows == 0:  image_collage.append([]);   GT_image_collage.append([]);    quantized_image_collage.append([])
            idx += 1
            QF = val_data['QF'].item()
            img_name = os.path.splitext(os.path.basename(val_data['Uncomp_path'][0]))[0]
            val_data['Z'] = cur_Z
            self.feed_data(val_data)
            self.test()
            if collect_avg_err_est:
                if self.chroma_mode:
                    assert self.DCT_generator
                    DCT_diffs = (self.var_Comp[:,256:,...]-self.fake_H).view(2,64,self.var_Comp.shape[2],self.var_Comp.shape[3]).abs()
                else:
                    DCT_diffs = ((self.var_Comp-self.fake_H) if self.DCT_generator else (self.JPEG['compressor'](self.var_Comp)-self.JPEG['non_quantized_compressor'](self.fake_H))).abs()
                self.average_abs_err_estimates += torch.mean(DCT_diffs,dim=(0,2,3)).view([8,8]).detach().cpu().numpy()
                self.average_err_collection_counter += 1
                if first_eval:
                    GT_DCT_diffs = ((self.var_Comp if self.DCT_generator else self.JPEG['compressor'](self.var_Comp))-self.JPEG['non_quantized_compressor'](self.var_Uncomp)).abs()
                    if self.chroma_mode:
                        GT_DCT_diffs = GT_DCT_diffs[:, 256:, ...].view(2, 64, GT_DCT_diffs.shape[2], GT_DCT_diffs.shape[3])
                    self.average_abs_GT_err += torch.mean(GT_DCT_diffs,dim=(0,2,3)).view([8,8]).detach().cpu().numpy()
            # self.test(Y_already_computed=True) # I pass Y_already_computed because Y was computed inside feed_data, and self.model_input now comprises the computed Y generator output.
            visuals = self.get_current_visuals()
            sr_img = util.tensor2img(visuals['Decomp'], out_type=np.uint8, min_max=[0, 255],chroma_mode=chroma_mode)  # float32
            gt_img = util.tensor2img(visuals['Uncomp'], out_type=np.uint8, min_max=[0, 255],chroma_mode=chroma_mode)  # float32
            avg_psnr.append(util.calculate_psnr(sr_img, gt_img))
            if not self.chroma_mode:
                avg_niqe.append(niqe(np.expand_dims(sr_img[...,0],0)))
            if save_images:
                if SAVE_IMAGE_COLLAGE:
                    margins2crop = ((np.array(sr_img.shape[:2]) - per_image_saved_patch) / 2).astype(np.int32)
                    image_collage[-1].append(np.clip(sr_img[margins2crop[0]:-margins2crop[0], margins2crop[1]:-margins2crop[1], ...], 0,255).astype(np.uint8))
                    if first_eval:  # Save GT Uncomp images
                        GT_image_collage[-1].append(np.clip(gt_img[margins2crop[0]:-margins2crop[0], margins2crop[1]:-margins2crop[1], ...], 0,255).astype(np.uint8))
                        quantized_image = util.tensor2img(self.JPEG['extractor'](self.var_Comp) if self.DCT_generator else self.var_Comp,out_type=np.uint8, min_max=[0, 255],chroma_mode=chroma_mode)
                        quantized_image_collage[-1].append(quantized_image[margins2crop[0]:-margins2crop[0], margins2crop[1]:-margins2crop[1], ...])
                        avg_quantized_psnr.append(util.calculate_psnr(quantized_image, gt_img))
                        if not self.chroma_mode:
                            avg_quantized_niqe.append(niqe(np.expand_dims(quantized_image[...,0],0)))
                            avg_GT_niqe.append(niqe(np.expand_dims(gt_img[...,0],0)))
                        quantized_image_collage[-1][-1] = cv2.putText(quantized_image_collage[-1][-1].copy(), str(QF), (0, 50),cv2.FONT_HERSHEY_PLAIN, fontScale=4.0,
                                    color=np.mod(255 / 2 + quantized_image_collage[-1][-1][:25, :25].mean(), 255),thickness=2)
                        # if self.chroma_mode: # In this case cv2.putText returns cv2.Umat instead of an ndarray, so it should be converted:
                        #     quantized_image_collage[-1][-1] = quantized_image_collage[-1][-1].get()
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
                if first_eval:
                    self.log_dict['per_im_quantized_psnr_QF%d' % (QF)] = [(0, 0)]
                    if not self.chroma_mode:
                        self.log_dict['per_im_quantized_niqe_QF%d' % (QF)] = [(0, 0)]
                        self.log_dict['per_im_GT_niqe_QF%d' % (QF)] = [(0, 0)]
            if first_eval:
                self.log_dict['per_im_quantized_psnr_QF%d' % (QF)][0] = \
                    (0,((QF_images_counter[QF]-1)*self.log_dict['per_im_quantized_psnr_QF%d' % (QF)][0][1] + avg_quantized_psnr[-1])/QF_images_counter[QF])
                if not self.chroma_mode:
                    self.log_dict['per_im_quantized_niqe_QF%d' % (QF)][0] = \
                        (0, ((QF_images_counter[QF] - 1) * self.log_dict['per_im_quantized_niqe_QF%d' % (QF)][0][1] +avg_quantized_niqe[-1]) / QF_images_counter[QF])
                    self.log_dict['per_im_GT_niqe_QF%d' % (QF)][0] = \
                        (0, ((QF_images_counter[QF] - 1) * self.log_dict['per_im_GT_niqe_QF%d' % (QF)][0][1] +avg_GT_niqe[-1]) / QF_images_counter[QF])
        if save_images:
            self.generator_changed = False

        if collect_avg_err_est:
            import matplotlib.pyplot as plt
            plt.clf()
            # plt.imshow(np.log(self.netG.module.return_collected_err_avg()))
            self.avg_estimated_err = np.concatenate([self.avg_estimated_err,np.expand_dims(self.return_collected_err_avg(first_eval=first_eval),-1)],-1)
            self.avg_estimated_err_step.append(self.gradient_step_num)
            plt.imshow(np.log(self.avg_estimated_err[...,-1]))
            plt.colorbar()
            plt.title('Log(|estimated errors|/|GT errors|)')
            plt.savefig(os.path.join(self.log_path,'Est_quantization_errors.png'))
        avg_psnr = [51.14 if np.isinf(v) else v for v in avg_psnr] # Replacing inf values with PSNR corresponding to the error being the quantization error (0.5), to prevent contaminating the average
        for i, QF in enumerate(data_loader.dataset.per_index_QF):
            print_rlt['psnr_gain_QF%d' % (QF)] += (avg_psnr[i] - self.log_dict['per_im_quantized_psnr_QF%d' % (QF)][0][1])/QF_images_counter[QF]
        avg_psnr = 1 * np.mean(avg_psnr)
        if not self.chroma_mode:
            avg_niqe = 1*np.mean(avg_niqe)
        if SAVE_IMAGE_COLLAGE and save_images:
            save_img_path = os.path.join(os.path.join(self.opt['path']['val_images']),'{:d}_{}PSNR{:.3f}.png'.format(self.gradient_step_num,
                ('Z' + str(cur_Z)) if self.opt['network_G']['latent_input'] else '', avg_psnr))
            self.im_collages.append(np.concatenate([np.concatenate(col, 0) for col in image_collage], 1))
            util.save_img(self.im_collages[-1], save_img_path)
            if first_eval:  # Save GT Uncomp images
                util.save_img(np.concatenate([np.concatenate(col, 0) for col in GT_image_collage], 1),
                              os.path.join(os.path.join(self.opt['path']['val_images']), 'GT_Uncomp.png'))
                avg_quantized_psnr = 1 * np.mean(avg_quantized_psnr)
                print_rlt['quantized_psnr'] = avg_quantized_psnr
                util.save_img(np.concatenate([np.concatenate(col, 0) for col in quantized_image_collage], 1),os.path.join(os.path.join(self.opt['path']['val_images']),
                       'Quantized_{}_PSNR{:.3f}.png'.format(('Z' + str(cur_Z)) if self.opt['network_G']['latent_input'] else '',avg_quantized_psnr)))
                self.log_dict['quantized_psnr_val'] = [(self.gradient_step_num, print_rlt['quantized_psnr'])]
                if not self.chroma_mode:
                    avg_quantized_niqe = 1 * np.mean(avg_quantized_niqe)
                    print_rlt['quantized_niqe'] = avg_quantized_niqe
                    self.log_dict['quantized_niqe_val'] = [(self.gradient_step_num, print_rlt['quantized_niqe'])]
                    avg_GT_niqe = 1 * np.mean(avg_GT_niqe)
                    print_rlt['GT_niqe'] = avg_GT_niqe
                    self.log_dict['GT_niqe_val'] = [(self.gradient_step_num, print_rlt['GT_niqe'])]

        print_rlt['psnr'] += avg_psnr
        if not self.chroma_mode:
            print_rlt['niqe'] += avg_niqe

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
            # np.savez(os.path.join(self.log_path,'lr.npz'),step_num=cur_step,lr_G =self.optimizer_G.param_groups[0]['lr'],lr_D =self.optimizer_D.param_groups[0]['lr'])
            self.lr_G,self.lr_D = self.optimizer_G.param_groups[0]['lr'],self.optimizer_D.param_groups[0]['lr']
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
        for attr in ADDITIONALLY_SAVED_ATTRIBUTES:
            dict_2_save[attr] = getattr(self,attr)
        np.savez(os.path.join(self.log_path,'logs.npz'), ** dict_2_save)
        if self.cri_latent is not None and 'collected_ratios' in self.cri_latent.__dir__():
            np.savez(os.path.join(self.log_path,'collected_stats.npz'),*self.cri_latent.collected_ratios)
        if 'avg_estimated_err' in self.__dir__():
            np.savez(os.path.join(self.log_path,'avg_estimated_err.npz'),avg_estimated_err=self.avg_estimated_err,avg_estimated_err_step=self.avg_estimated_err_step)

    def load_log(self,max_step=None):
        PREPEND_OLD_LOG = False
        loaded_log = np.load(os.path.join(self.log_path,'logs.npz'),allow_pickle=True)
        if PREPEND_OLD_LOG:
            old_log = np.load(os.path.join(self.log_path, 'old_logs.npz'))
        self.log_dict = OrderedDict([val for val in zip(self.log_dict.keys(),[[] for i in self.log_dict.keys()])])
        for key in loaded_log.files:
            if key=='psnr_val':
                self.log_dict[key] = ([tuple(val) for val in old_log[key]] if PREPEND_OLD_LOG else [])+[tuple(val) for val in loaded_log[key]]
            elif key in ADDITIONALLY_SAVED_ATTRIBUTES:
                setattr(self,key,loaded_log[key])
                continue
            # elif key=='D_verified':
            #     self.D_verified = bool(loaded_log[key])
            #     continue
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
            # out_dict['Decomp'] = self.fake_H.detach().float().cpu()
            out_dict['Decomp'] = self.Output_Batch(within_0_1=False).detach().float().cpu()
            if need_Uncomp:
                out_dict['Uncomp'] = self.var_Uncomp.detach().float().cpu()
        else:
            out_dict['Comp'] = self.var_Comp.detach()[0].float().cpu()
            # out_dict['Decomp'] = self.fake_H.detach()[0].float().cpu()
            out_dict['Decomp'] = self.Output_Batch(within_0_1=False).detach()[0].float().cpu()
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
                net_desc = self.get_network_description(self.netF)
                # s, n,receptive_field = self.get_network_description(self.netF)
                # print('Number of parameters in F: {:,d}. Receptive field size: ({:,d},{:,d})'.format(n, *receptive_field))
                # s, n = self.get_network_description(self.netF)
                print('Number of parameters in F: {:,d}'.format(net_desc['n']))
                message = '\n\n\n-------------- Perceptual Network --------------\n' + s + '\n'
                if not self.opt['train']['resume']:
                    with open(network_path, 'a') as f:
                        f.write(message)

    def load(self,max_step=None,resume_train=None):
        resume_training = resume_train if resume_train is not None else (self.opt['is_train'] and self.opt['train']['resume'])
        if max_step is not None or (resume_training is not None and resume_training) or (not self.opt['is_train'] and not self.opt['Y_model']):
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
                    # self.load_network(os.path.join(self.opt['path']['models'],model_name), self.netD)
            else:
                print('Testing model for G [{:s}] ...'.format(os.path.join(self.opt['path']['models'],model_name)))
                self.load_network(os.path.join(self.opt['path']['models'],model_name), self.netG)
                if 'netD' in self.__dict__.keys(): #When running from GUI
                    model_name = model_name.replace('_G','_D')
                    print('Loading also model for D [{:s}] ...'.format(os.path.join(self.opt['path']['models'], model_name)))
                    self.load_network(os.path.join(self.opt['path']['models'], model_name), self.netD)
                self.gradient_step_num = loaded_model_step

        else:
            load_path_G = self.opt['path']['pretrained_model_G']
            if load_path_G is not None:
                print('loading model for G [{:s}] ...'.format(load_path_G))
                self.load_network(load_path_G, self.netG)
            load_path_D = self.opt['path']['pretrained_model_D']
            if self.opt['is_train'] and load_path_D is not None and self.D_exists:
                print('loading model for D [{:s}] ...'.format(load_path_D))
                self.load_network(load_path_D, self.netD)
                # self.load_network(load_path_D, self.netD,optimizer=self.optimizer_D)
        if self.chroma_mode and USE_Y_GENERATOR_4_CHROMA:
            print('loading model for G of channel Y [{:s}] ...'.format(self.opt['path']['Y_channel_model_G']))
            self.load_network(self.opt['path']['Y_channel_model_G'], self.netG_Y)

    def save(self, iter_label,first_verified_D=False):
        # if first_verified_D:
        #     self.save_network(self.save_dir, self.netD, 'D_verified', iter_label, self.optimizer_D)
        #     return
        if self.D_exists:
            self.save_network(self.save_dir, self.netD, 'D'+('_verified' if first_verified_D else ''), iter_label,self.optimizer_D)
        saving_path = self.save_network(self.save_dir, self.netG, 'G'+('_verified' if first_verified_D else ''), iter_label,self.optimizer_G)
        return saving_path

