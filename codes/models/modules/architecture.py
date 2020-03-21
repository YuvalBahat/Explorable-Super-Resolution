import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision
from . import block as B
from . import spectral_norm as SN
import functools
import numpy as np
import os
import models.modules.archs_util as arch_util
import torch.nn.functional as F
import re

####################
# Generator
####################
class MSRResNet(nn.Module):
    ''' modified SRResNet'''

    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=16, upscale=4):
        super(MSRResNet, self).__init__()
        self.upscale = upscale

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        basic_block = functools.partial(arch_util.ResidualBlock_noBN, nf=nf)
        self.recon_trunk = arch_util.make_layer(basic_block, nb)

        # upsampling
        if self.upscale == 2:
            self.upconv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(2)
        elif self.upscale == 3:
            self.upconv1 = nn.Conv2d(nf, nf * 9, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(3)
        elif self.upscale == 4:
            self.upconv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
            self.upconv2 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
            self.pixel_shuffle = nn.PixelShuffle(2)

        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # initialization
        arch_util.initialize_weights([self.conv_first, self.upconv1, self.HRconv, self.conv_last],
                                     0.1)
        if self.upscale == 4:
            arch_util.initialize_weights(self.upconv2, 0.1)

    def forward(self, x):
        fea = self.lrelu(self.conv_first(x))
        out = self.recon_trunk(fea)

        if self.upscale == 4:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
            out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))
        elif self.upscale == 3 or self.upscale == 2:
            out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))

        out = self.conv_last(self.lrelu(self.HRconv(out)))
        base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        out += base
        return out


class SRResNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, upscale=4, norm_type='batch', act_type='relu', \
            mode='NAC', res_scale=1, upsample_mode='upconv'):
        super(SRResNet, self).__init__()
        n_upscale = int(math.log(upscale, 2))
        if upscale == 3:
            n_upscale = 1

        fea_conv = B.conv_block(in_nc, nf, kernel_size=3, norm_type=None, act_type=None)
        resnet_blocks = [B.ResNetBlock(nf, nf, nf, norm_type=norm_type, act_type=act_type,\
            mode=mode, res_scale=res_scale) for _ in range(nb)]
        LR_conv = B.conv_block(nf, nf, kernel_size=3, norm_type=norm_type, act_type=None, mode=mode)

        if upsample_mode == 'upconv':
            upsample_block = B.upconv_blcok
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.pixelshuffle_block
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))
        if upscale == 3:
            upsampler = upsample_block(nf, nf, 3, act_type=act_type)
        else:
            upsampler = [upsample_block(nf, nf, act_type=act_type) for _ in range(n_upscale)]
        HR_conv0 = B.conv_block(nf, nf, kernel_size=3, norm_type=None, act_type=act_type)
        HR_conv1 = B.conv_block(nf, out_nc, kernel_size=3, norm_type=None, act_type=None)

        self.model = B.sequential(fea_conv, B.ShortcutBlock(B.sequential(*resnet_blocks, LR_conv)),\
            *upsampler, HR_conv0, HR_conv1)

    def forward(self, x):
        x = self.model(x)
        return x

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class DnCNN(nn.Module):
    def __init__(self, n_channels, depth, kernel_size = 3, in_nc=64, out_nc=64, norm_type='batch', act_type='leakyrelu',
                 latent_input=None,num_latent_channels=None,discriminator=False,expected_input_size=None):
        super(DnCNN, self).__init__()
        assert in_nc==64 and out_nc==64,'Currently only supporting 64 DCT channels'
        assert act_type=='leakyrelu'
        assert norm_type in ['batch','instance','layer',None]
        assert latent_input is None and num_latent_channels is None
        self.average_err_collection_counter = 0
        self.average_abs_err_estimates = np.zeros([8,8])
        self.discriminator_net = discriminator
        padding = 0 if self.discriminator_net else kernel_size//2
        layers = []

        layers.append(nn.Conv2d(in_channels=in_nc, out_channels=n_channels, kernel_size=kernel_size, padding=padding,bias=True))
        if self.discriminator_net:
            expected_input_size -= (kernel_size - 1)
        layers.append(nn.ReLU(inplace=True))
        for _ in range(depth - 2):
            layers.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding,bias=False))
            if self.discriminator_net:
                expected_input_size -= (kernel_size-1)
            if norm_type=='batch':
                layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum=0.95))
            elif norm_type=='layer':
                layers.append(nn.LayerNorm(normalized_shape=[n_channels,expected_input_size,expected_input_size],elementwise_affine=False))
            elif norm_type=='instance':
                layers.append(nn.InstanceNorm2d(n_channels))
            layers.append(nn.LeakyReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=n_channels, out_channels=out_nc, kernel_size=kernel_size, padding=padding,bias=False))
        if self.discriminator_net:
            expected_input_size -= (kernel_size - 1)
            layers.append(Flatten())
            layers.append(nn.Linear(in_features=out_nc*(expected_input_size**2),out_features=1))
            # layers.append(nn.Linear(in_features=64, out_features=1))
        layers.append(nn.Sigmoid())
        self.dncnn = nn.Sequential(*layers)
        # self._initialize_weights()

    def forward(self, x):
        if self.discriminator_net:
            return self.dncnn(x)
        else:
            quantization_err_estimation = self.dncnn(x)-0.5
            if not next(self.modules()).training:
                self.average_err_collection_counter += 1
                self.average_abs_err_estimates = ((self.average_err_collection_counter-1)*self.average_abs_err_estimates+
                    quantization_err_estimation.abs().mean(-1).mean(-1).mean(0).view(8,8).data.cpu().numpy())/self.average_err_collection_counter
            return x+quantization_err_estimation

    def return_collected_err_avg(self):
        self.average_err_collection_counter = 0
        natrix_2_return = 1*self.average_abs_err_estimates
        self.average_abs_err_estimates = np.zeros([8,8])
        return natrix_2_return

    def save_estimated_errors_fig(self,quantization_err_batch):
        import matplotlib.pyplot as plt
        plt.clf()
        plt.imshow(quantization_err_batch.abs().mean(-1).mean(-1).mean(0).view(8,8).data.cpu().numpy())
        plt.colorbar()
        plt.savefig('Est_quantization_errors_0iters_95Kiters.png')

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                print('init weight')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)


class RRDBNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, gc=32, upscale=4, norm_type=None, \
            act_type='leakyrelu', mode='CNA', upsample_mode='upconv',latent_input=None,num_latent_channels=None):
        super(RRDBNet, self).__init__()
        self.latent_input = latent_input
        if num_latent_channels is not None and num_latent_channels>0:
            num_latent_channels_HR = 1 * num_latent_channels
            if 'HR_rearranged' in latent_input:
                num_latent_channels *= upscale**2
        self.num_latent_channels = 1*num_latent_channels
        self.upscale = upscale
        n_upscale = int(math.log(upscale, 2))
        if upscale == 3:
            n_upscale = 1
        if latent_input is not None:
            in_nc += num_latent_channels
        if latent_input is None or 'all_layers' not in latent_input:
            num_latent_channels,num_latent_channels_HR = 0,0

        USE_MODULE_LISTS = True
        fea_conv = B.conv_block(in_nc, nf, kernel_size=3, norm_type=None, act_type=None,return_module_list=USE_MODULE_LISTS)
        rb_blocks = [B.RRDB(nf, kernel_size=3, gc=32, stride=1, bias=True, pad_type='zero', \
            norm_type=norm_type, act_type=act_type, mode='CNA',latent_input_channels=num_latent_channels) for _ in range(nb)]
        LR_conv = B.conv_block(nf+num_latent_channels, nf, kernel_size=3, norm_type=norm_type, act_type=None, mode=mode,return_module_list=USE_MODULE_LISTS)

        if upsample_mode == 'upconv':
            upsample_block = B.upconv_blcok
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.pixelshuffle_block
        else:
            raise NotImplementedError('upsample mode [{:s}] is not found'.format(upsample_mode))
        if upscale == 3:
            upsampler = upsample_block(nf, nf, 3, act_type=act_type)
        else:
            upsampler = [upsample_block(nf, nf, act_type=act_type) for _ in range(n_upscale)]
        if latent_input is not None and 'all_layers' in latent_input:
            if 'LR' in latent_input:
                self.latent_upsampler = nn.Upsample(scale_factor=upscale if upscale==3 else 2)
        HR_conv0 = B.conv_block(nf+num_latent_channels_HR, nf, kernel_size=3, norm_type=None, act_type=act_type,return_module_list=USE_MODULE_LISTS)
        HR_conv1 = B.conv_block(nf+num_latent_channels_HR, out_nc, kernel_size=3, norm_type=None, act_type=None,return_module_list=USE_MODULE_LISTS)

        if USE_MODULE_LISTS:
            self.model = nn.ModuleList(fea_conv+\
                [B.ShortcutBlock(B.sequential(*(rb_blocks+LR_conv),return_module_list=USE_MODULE_LISTS),latent_input_channels=num_latent_channels,use_module_list=True)]+\
                                       upsampler+HR_conv0+HR_conv1)
        else:
            self.model = B.sequential(fea_conv, B.ShortcutBlock(B.sequential(*rb_blocks, LR_conv)),\
                *upsampler, HR_conv0, HR_conv1)

    def forward(self, x):
        if self.latent_input is not None:
            if 'HR_downscaled' in self.latent_input:
                # latent_input_HR = 1*self.Z
                latent_input_HR,x = torch.split(x,split_size_or_sections=[x.size(1)-3,3],dim=1)
                latent_input_HR = latent_input_HR.view([latent_input_HR.size(0)]+[-1]+[self.upscale*val for val in list(latent_input_HR.size()[2:])])
                latent_input = torch.nn.functional.interpolate(input=latent_input_HR,scale_factor=1/self.upscale,mode='bilinear',align_corners=False)
            else:
                latent_input = 1*self.Z
            x = torch.cat([latent_input, x], dim=1)
        for i,module in enumerate(self.model):
            module_children = [str(type(m)) for m in module.children()]
            if i>0 and self.latent_input is not None and 'all_layers' in self.latent_input:
                if len(module_children)>0 and 'Upsample' in module_children[0]:
                    if 'LR' in self.latent_input:
                        latent_input = self.latent_upsampler(latent_input)
                    elif 'HR_rearranged' in self.latent_input:
                        raise Exception('Unsupported yet')
                        latent_input = latent_input.view()
                    elif 'HR_downscaled' in self.latent_input:
                        latent_input = 1*latent_input_HR
                elif 'ReLU' not in str(type(module)):
                    x = torch.cat([latent_input,x],1)
            x = module(x)
        return x


####################
# Discriminator
####################

class PatchGAN_Discriminator(nn.Module):
    DEFAULT_N_LAYERS = 3

    def __init__(self, input_nc, opt_net,ndf=64, n_layers=DEFAULT_N_LAYERS, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(PatchGAN_Discriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        self.decomposed_input = bool(opt_net['decomposed_input'])
        self.pre_clipping = bool(opt_net['pre_clipping'])
        projected_component_sequences = []
        in_ch_addition = input_nc if self.decomposed_input else 0
        kw = 4
        padw = 1
        max_out_channels = 512
        sequences = [nn.Sequential(*[nn.Conv2d(input_nc+in_ch_addition, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)])]
        # if self.decomposed_input:
        #     projected_component_sequences = [nn.Conv2d(input_nc, input_nc, kernel_size=kw, stride=2, padding=padw)]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            # nf_mult_prev = nf_mult
            # nf_mult = min(2 ** max(0,n-n_layers+self.DEFAULT_N_LAYERS), 8)
            nf_mult_prev = min(max_out_channels, ndf * nf_mult) // ndf
            nf_mult = min(2 ** n, 8)
            sequences.append(nn.Sequential(*[
                nn.Conv2d(ndf * nf_mult_prev+in_ch_addition, min(max_out_channels, ndf * nf_mult), kernel_size=kw,
                          stride=2 if n > n_layers - self.DEFAULT_N_LAYERS else 1,
                          padding=padw, bias=use_bias), norm_layer(ndf * nf_mult), nn.LeakyReLU(0.2, True)]))
            # if self.decomposed_input:
            #     projected_component_sequences.append(
            #         nn.Conv2d(input_nc,input_nc, kernel_size=kw,
            #                   stride=2 if n > n_layers - self.DEFAULT_N_LAYERS else 1,
            #                   padding=padw, bias=use_bias))

        # nf_mult_prev = nf_mult
        nf_mult_prev = min(max_out_channels, ndf * nf_mult) // ndf
        nf_mult = min(2 ** n_layers, 8)
        sequences.append(nn.Sequential(*[
            nn.Conv2d(ndf * nf_mult_prev+in_ch_addition, min(max_out_channels, ndf * nf_mult), kernel_size=kw, stride=1,
                      padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)]))
        # if self.decomposed_input:
        #     projected_component_sequences.append(
        #     nn.Conv2d(input_nc,input_nc, kernel_size=kw, stride=1,
        #               padding=padw, bias=use_bias))
        sequences.append(nn.Sequential(*[
            nn.Conv2d(min(max_out_channels, ndf * nf_mult)+in_ch_addition, 1, kernel_size=kw, stride=1,
                      padding=padw)]))  # output 1 channel prediction map
        self.num_modules = len(sequences)
        if self.decomposed_input:
            for seq in sequences:
                conv_stride = [child.stride[0] for child in seq.children() if 'Conv2d' in str(child.__class__)]
                assert len(conv_stride)<=1,'More than one conv layer in seq?'
                if len(conv_stride)>0:
                    projected_component_sequences.append(nn.Conv2d(input_nc,input_nc, kernel_size=kw, stride=conv_stride[0],
                      padding=padw, bias=use_bias))
        self.model = nn.ModuleList(sequences+projected_component_sequences)

    def forward(self, input):
        # pre-clipping:
        # 1.Making D oblivious to pixel values range, by clipping values to be within valid range
        # 2.Making D oblivious to quantization issues, by quantizing its inputs to 256 possible values
        if self.decomposed_input:
            projected_component = input[0]
            input = input[1]
            if self.pre_clipping:
                input = torch.max(input=torch.min(input=input,other=1-projected_component),other=-projected_component)
                # input = (255*(input+projected_component)).round()/255-projected_component
        elif self.pre_clipping:
            input = torch.clamp(input=input,min=0,max=1)
            # input = (255*input).round()/255
        for i,seq in enumerate(self.model[:self.num_modules]):
            if self.decomposed_input:
                if i > 0:
                    projected_component = self.model[self.num_modules + i - 1](projected_component)
                input = seq(torch.cat([projected_component,input],dim=1))
            else:
                input = seq(input)
        return input

class Discriminator_VGG_128_nonModified(nn.Module):
    def __init__(self, in_nc, nf):
        super(Discriminator_VGG_128_nonModified, self).__init__()
        # [64, 128, 128]
        self.conv0_0 = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.conv0_1 = nn.Conv2d(nf, nf, 4, 2, 1, bias=False)
        self.bn0_1 = nn.BatchNorm2d(nf, affine=True)
        # [64, 64, 64]
        self.conv1_0 = nn.Conv2d(nf, nf * 2, 3, 1, 1, bias=False)
        self.bn1_0 = nn.BatchNorm2d(nf * 2, affine=True)
        self.conv1_1 = nn.Conv2d(nf * 2, nf * 2, 4, 2, 1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(nf * 2, affine=True)
        # [128, 32, 32]
        self.conv2_0 = nn.Conv2d(nf * 2, nf * 4, 3, 1, 1, bias=False)
        self.bn2_0 = nn.BatchNorm2d(nf * 4, affine=True)
        self.conv2_1 = nn.Conv2d(nf * 4, nf * 4, 4, 2, 1, bias=False)
        self.bn2_1 = nn.BatchNorm2d(nf * 4, affine=True)
        # [256, 16, 16]
        self.conv3_0 = nn.Conv2d(nf * 4, nf * 8, 3, 1, 1, bias=False)
        self.bn3_0 = nn.BatchNorm2d(nf * 8, affine=True)
        self.conv3_1 = nn.Conv2d(nf * 8, nf * 8, 4, 2, 1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(nf * 8, affine=True)
        # [512, 8, 8]
        self.conv4_0 = nn.Conv2d(nf * 8, nf * 8, 3, 1, 1, bias=False)
        self.bn4_0 = nn.BatchNorm2d(nf * 8, affine=True)
        self.conv4_1 = nn.Conv2d(nf * 8, nf * 8, 4, 2, 1, bias=False)
        self.bn4_1 = nn.BatchNorm2d(nf * 8, affine=True)

        self.linear1 = nn.Linear(512 * 4 * 4, 100)
        self.linear2 = nn.Linear(100, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.lrelu(self.conv0_0(x))
        fea = self.lrelu(self.bn0_1(self.conv0_1(fea)))

        fea = self.lrelu(self.bn1_0(self.conv1_0(fea)))
        fea = self.lrelu(self.bn1_1(self.conv1_1(fea)))

        fea = self.lrelu(self.bn2_0(self.conv2_0(fea)))
        fea = self.lrelu(self.bn2_1(self.conv2_1(fea)))

        fea = self.lrelu(self.bn3_0(self.conv3_0(fea)))
        fea = self.lrelu(self.bn3_1(self.conv3_1(fea)))

        fea = self.lrelu(self.bn4_0(self.conv4_0(fea)))
        fea = self.lrelu(self.bn4_1(self.conv4_1(fea)))

        fea = fea.view(fea.size(0), -1)
        fea = self.lrelu(self.linear1(fea))
        out = self.linear2(fea)
        return out

# VGG style Discriminator with input size 128*128
class Discriminator_VGG_128(nn.Module):
    def __init__(self, in_nc, base_nf, norm_type='batch', act_type='leakyrelu', mode='CNA',input_patch_size=128,num_2_strides=5,nb=10):
        super(Discriminator_VGG_128, self).__init__()
        assert num_2_strides<=5,'Can be modified by adding more stridable layers, if needed.'
        self.num_2_strides = 1*num_2_strides
        # features
        # hxw, c
        # 128, 64
        FC_end_patch_size = 1*input_patch_size
        conv0 = B.conv_block(in_nc, base_nf, kernel_size=3, norm_type=None, act_type=act_type,mode=mode)
        conv1 = B.conv_block(base_nf, base_nf, kernel_size=4, stride=2 if num_2_strides>0 else 1, norm_type=norm_type,act_type=act_type, mode=mode)
        FC_end_patch_size = np.ceil((FC_end_patch_size-1)/(2 if num_2_strides>0 else 1))
        num_2_strides -= 1
        # 64, 64
        conv2 = B.conv_block(base_nf, base_nf*2, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv3 = B.conv_block(base_nf*2, base_nf*2, kernel_size=4, stride=2 if num_2_strides>0 else 1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        FC_end_patch_size = np.ceil((FC_end_patch_size-1)/(2 if num_2_strides>0 else 1))
        num_2_strides -= 1
        # 32, 128
        conv4 = B.conv_block(base_nf*2, base_nf*4, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv5 = B.conv_block(base_nf*4, base_nf*4, kernel_size=4, stride=2 if num_2_strides>0 else 1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        FC_end_patch_size = np.ceil((FC_end_patch_size-1)/(2 if num_2_strides>0 else 1))
        num_2_strides -= 1
        # 16, 256
        conv6 = B.conv_block(base_nf*4, base_nf*8, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv7 = B.conv_block(base_nf*8, base_nf*8, kernel_size=4, stride=2 if num_2_strides>0 else 1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        FC_end_patch_size = np.ceil((FC_end_patch_size-1)/(2 if num_2_strides>0 else 1))
        num_2_strides -= 1
        # 8, 512
        conv8 = B.conv_block(base_nf*8, base_nf*8, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv9 = B.conv_block(base_nf*8, base_nf*8, kernel_size=4, stride=2 if num_2_strides>0 else 1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        FC_end_patch_size = np.ceil((FC_end_patch_size-1)/(2 if num_2_strides>0 else 1))
        num_2_strides -= 1
        # 4, 512
        self.features = B.sequential(*([conv0, conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8,conv9][:nb]))

        self.last_FC_layers = self.num_2_strides==5 #Replacing the FC layers with convolutions, which means using a patch discriminator:
        self.last_FC_layers = False
        # classifier
        # FC_end_patch_size = input_patch_size//(2**self.num_2_strides)
        if self.last_FC_layers:
            self.classifier = nn.Sequential(nn.Linear(base_nf*8 * int(FC_end_patch_size)**2, 100), nn.LeakyReLU(0.2, True), nn.Linear(100, 1))
        else:
            # num_feature_channels = base_nf*8
            num_feature_channels = [l for l in self.features.children()][-2].num_features
            pseudo_FC_conv0 = B.conv_block(num_feature_channels,min(100,num_feature_channels),kernel_size=8,stride=1,norm_type=norm_type,act_type=act_type, mode=mode,pad_type=None)
            pseudo_FC_conv1 = B.conv_block(min(100,num_feature_channels),1,kernel_size=1,stride=1,norm_type=norm_type,act_type=act_type, mode=mode)
            self.classifier = nn.Sequential(pseudo_FC_conv0, nn.LeakyReLU(0.2, False),pseudo_FC_conv1) # Changed the LeakyRelu inplace arg to False here, because it caused a bug for some reason.

    def forward(self, x):
        x = self.features(x)
        if self.last_FC_layers:
            x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# VGG style Discriminator with input size 128*128, Spectral Normalization
class Discriminator_VGG_128_SN(nn.Module):
    def __init__(self):
        super(Discriminator_VGG_128_SN, self).__init__()
        # features
        # hxw, c
        # 128, 64
        self.lrelu = nn.LeakyReLU(0.2, True)

        self.conv0 = SN.spectral_norm(nn.Conv2d(3, 64, 3, 1, 1))
        self.conv1 = SN.spectral_norm(nn.Conv2d(64, 64, 4, 2, 1))
        # 64, 64
        self.conv2 = SN.spectral_norm(nn.Conv2d(64, 128, 3, 1, 1))
        self.conv3 = SN.spectral_norm(nn.Conv2d(128, 128, 4, 2, 1))
        # 32, 128
        self.conv4 = SN.spectral_norm(nn.Conv2d(128, 256, 3, 1, 1))
        self.conv5 = SN.spectral_norm(nn.Conv2d(256, 256, 4, 2, 1))
        # 16, 256
        self.conv6 = SN.spectral_norm(nn.Conv2d(256, 512, 3, 1, 1))
        self.conv7 = SN.spectral_norm(nn.Conv2d(512, 512, 4, 2, 1))
        # 8, 512
        self.conv8 = SN.spectral_norm(nn.Conv2d(512, 512, 3, 1, 1))
        self.conv9 = SN.spectral_norm(nn.Conv2d(512, 512, 4, 2, 1))
        # 4, 512

        # classifier
        self.linear0 = SN.spectral_norm(nn.Linear(512 * 4 * 4, 100))
        self.linear1 = SN.spectral_norm(nn.Linear(100, 1))

    def forward(self, x):
        x = self.lrelu(self.conv0(x))
        x = self.lrelu(self.conv1(x))
        x = self.lrelu(self.conv2(x))
        x = self.lrelu(self.conv3(x))
        x = self.lrelu(self.conv4(x))
        x = self.lrelu(self.conv5(x))
        x = self.lrelu(self.conv6(x))
        x = self.lrelu(self.conv7(x))
        x = self.lrelu(self.conv8(x))
        x = self.lrelu(self.conv9(x))
        x = x.view(x.size(0), -1)
        x = self.lrelu(self.linear0(x))
        x = self.linear1(x)
        return x


class Discriminator_VGG_96(nn.Module):
    def __init__(self, in_nc, base_nf, norm_type='batch', act_type='leakyrelu', mode='CNA'):
        super(Discriminator_VGG_96, self).__init__()
        # features
        # hxw, c
        # 96, 64
        conv0 = B.conv_block(in_nc, base_nf, kernel_size=3, norm_type=None, act_type=act_type, \
            mode=mode)
        conv1 = B.conv_block(base_nf, base_nf, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 48, 64
        conv2 = B.conv_block(base_nf, base_nf*2, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv3 = B.conv_block(base_nf*2, base_nf*2, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 24, 128
        conv4 = B.conv_block(base_nf*2, base_nf*4, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv5 = B.conv_block(base_nf*4, base_nf*4, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 12, 256
        conv6 = B.conv_block(base_nf*4, base_nf*8, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv7 = B.conv_block(base_nf*8, base_nf*8, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 6, 512
        conv8 = B.conv_block(base_nf*8, base_nf*8, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv9 = B.conv_block(base_nf*8, base_nf*8, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 3, 512
        self.features = B.sequential(conv0, conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8,\
            conv9)

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(512 * 3 * 3, 100), nn.LeakyReLU(0.2, True), nn.Linear(100, 1))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class Discriminator_VGG_192(nn.Module):
    def __init__(self, in_nc, base_nf, norm_type='batch', act_type='leakyrelu', mode='CNA'):
        super(Discriminator_VGG_192, self).__init__()
        # features
        # hxw, c
        # 192, 64
        conv0 = B.conv_block(in_nc, base_nf, kernel_size=3, norm_type=None, act_type=act_type, \
            mode=mode)
        conv1 = B.conv_block(base_nf, base_nf, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 96, 64
        conv2 = B.conv_block(base_nf, base_nf*2, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv3 = B.conv_block(base_nf*2, base_nf*2, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 48, 128
        conv4 = B.conv_block(base_nf*2, base_nf*4, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv5 = B.conv_block(base_nf*4, base_nf*4, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 24, 256
        conv6 = B.conv_block(base_nf*4, base_nf*8, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv7 = B.conv_block(base_nf*8, base_nf*8, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 12, 512
        conv8 = B.conv_block(base_nf*8, base_nf*8, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv9 = B.conv_block(base_nf*8, base_nf*8, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 6, 512
        conv10 = B.conv_block(base_nf*8, base_nf*8, kernel_size=3, stride=1, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        conv11 = B.conv_block(base_nf*8, base_nf*8, kernel_size=4, stride=2, norm_type=norm_type, \
            act_type=act_type, mode=mode)
        # 3, 512
        self.features = B.sequential(conv0, conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8,\
            conv9, conv10, conv11)

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(512 * 3 * 3, 100), nn.LeakyReLU(0.2, True), nn.Linear(100, 1))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


####################
# Perceptual Network
####################
RETRAINING_OBLIGING_MODIFICATIONS = ['num_channel_factor_\d(\.\d)?$','patches_init_first']

# Assume input range is [0, 1]
class VGGFeatureExtractor(nn.Module):
    def __init__(self,feature_layer=34,use_bn=False,use_input_norm=True,
                 device=torch.device('cpu'),state_dict=None,arch='vgg19',arch_config='',**kwargs):
        super(VGGFeatureExtractor, self).__init__()
        if arch_config!='':
            assert all([re.search(pattern,arch_config) is None for pattern in RETRAINING_OBLIGING_MODIFICATIONS]) or 'untrained_' in arch_config
            # assert (re.search('patches_init_(first|all)',arch_config) is None) or 'untrained' not in arch_config,'Relying on trained weights statistics when setting model weights'
        if arch=='SegNetAE':
            from models.modules import SegNet
            model = nn.DataParallel(SegNet.SegNet(3,encode_only=True))
            loaded_state_dict = torch.load('/home/tiras/ybahat/Autoencoder/models/BEST_checkpoint.tar')['model']
            modified_state_dict = {}
            for key in model.state_dict().keys():
                modified_state_dict[key] = loaded_state_dict[key.replace('.features.0','.down1').replace('.features.1','.down2').replace('.features.2','.down3').replace('.features.3','.down4').replace('.features.4','.down5')]
            model.load_state_dict(modified_state_dict)
            model = model.module
        elif use_bn:
            model = torchvision.models.__dict__[arch+'_bn'](pretrained='untrained' not in arch_config)
        else:
            model = torchvision.models.__dict__[arch](pretrained='untrained' not in arch_config)
        # I now remove all unnecessary layers before changing the model configuration, because this change may make alter the number of layers, thus necessitating changing the feature_layer parameter.
        if state_dict is not None:
            state_dict = dict(zip([key.replace('module.','') for key in state_dict.keys()],[value for value in state_dict.values()]))
            model.load_state_dict(state_dict,strict=False)
        model.features = nn.Sequential(*list(model.features.children())[:(feature_layer + 1)])
        arch_config = arch_config.replace('untrained_','').replace('untrained','')
        if arch_config!='':
            import sys
            sys.path.append(os.path.abspath('../../RandomPooling'))
            from model_modification import Modify_Model
            saved_config_params = kwargs['saved_config_params'] if 'saved_config_params' in kwargs.keys() else None
            model = Modify_Model(model,arch_config,classification_mode=False,saved_config_params=saved_config_params)
        self.use_input_norm = use_input_norm
        if self.use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            # [0.485-1, 0.456-1, 0.406-1] if input in range [-1,1]
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            # [0.229*2, 0.224*2, 0.225*2] if input in range [-1,1]
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
        #     Moved the next line to appear earlier, before altering the number of layers in the model
        # self.features = nn.Sequential(*list(model.features.children())[:(feature_layer + 1)])
        self.features = model.features
        # No need to BP to variable
        for k, v in self.features.named_parameters():
            v.requires_grad = False

    def _initialize_weights(self):#This function was copied from the torchvision.models.vgg code:
        for m in self.features.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        output = self.features(x)
        return output


# Assume input range is [0, 1]
class ResNet101FeatureExtractor(nn.Module):
    def __init__(self, use_input_norm=True, device=torch.device('cpu')):
        super(ResNet101FeatureExtractor, self).__init__()
        model = torchvision.models.resnet101(pretrained=True)
        self.use_input_norm = use_input_norm
        if self.use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            # [0.485-1, 0.456-1, 0.406-1] if input in range [-1,1]
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            # [0.229*2, 0.224*2, 0.225*2] if input in range [-1,1]
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
        self.features = nn.Sequential(*list(model.children())[:8])
        # No need to BP to variable
        for k, v in self.features.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        output = self.features(x)
        return output


class MINCNet(nn.Module):
    def __init__(self):
        super(MINCNet, self).__init__()
        self.ReLU = nn.ReLU(True)
        self.conv11 = nn.Conv2d(3, 64, 3, 1, 1)
        self.conv12 = nn.Conv2d(64, 64, 3, 1, 1)
        self.maxpool1 = nn.MaxPool2d(2, stride=2, padding=0, ceil_mode=True)
        self.conv21 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv22 = nn.Conv2d(128, 128, 3, 1, 1)
        self.maxpool2 = nn.MaxPool2d(2, stride=2, padding=0, ceil_mode=True)
        self.conv31 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv32 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv33 = nn.Conv2d(256, 256, 3, 1, 1)
        self.maxpool3 = nn.MaxPool2d(2, stride=2, padding=0, ceil_mode=True)
        self.conv41 = nn.Conv2d(256, 512, 3, 1, 1)
        self.conv42 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv43 = nn.Conv2d(512, 512, 3, 1, 1)
        self.maxpool4 = nn.MaxPool2d(2, stride=2, padding=0, ceil_mode=True)
        self.conv51 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv52 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv53 = nn.Conv2d(512, 512, 3, 1, 1)

    def forward(self, x):
        out = self.ReLU(self.conv11(x))
        out = self.ReLU(self.conv12(out))
        out = self.maxpool1(out)
        out = self.ReLU(self.conv21(out))
        out = self.ReLU(self.conv22(out))
        out = self.maxpool2(out)
        out = self.ReLU(self.conv31(out))
        out = self.ReLU(self.conv32(out))
        out = self.ReLU(self.conv33(out))
        out = self.maxpool3(out)
        out = self.ReLU(self.conv41(out))
        out = self.ReLU(self.conv42(out))
        out = self.ReLU(self.conv43(out))
        out = self.maxpool4(out)
        out = self.ReLU(self.conv51(out))
        out = self.ReLU(self.conv52(out))
        out = self.conv53(out)
        return out

# Encoder:
def conv3x3(in_planes, out_planes):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,padding=1, bias=True)

def meanpoolConv(inplanes, outplanes):
    sequence = []
    sequence += [nn.AvgPool2d(kernel_size=2, stride=2)]
    sequence += [nn.Conv2d(inplanes, outplanes,
                           kernel_size=1, stride=1, padding=0, bias=True)]
    return nn.Sequential(*sequence)


def convMeanpool(inplanes, outplanes):
    sequence = []
    sequence += [conv3x3(inplanes, outplanes)]
    sequence += [nn.AvgPool2d(kernel_size=2, stride=2)]
    return nn.Sequential(*sequence)

class BasicBlock(nn.Module):
    def __init__(self, inplanes, outplanes, norm_layer=None, nl_layer=None):
        super(BasicBlock, self).__init__()
        layers = []
        if norm_layer is not None:
            layers += [norm_layer(inplanes)]
        layers += [nl_layer()]
        layers += [conv3x3(inplanes, inplanes)]
        if norm_layer is not None:
            layers += [norm_layer(inplanes)]
        layers += [nl_layer()]
        layers += [convMeanpool(inplanes, outplanes)]
        self.conv = nn.Sequential(*layers)
        self.shortcut = meanpoolConv(inplanes, outplanes)

    def forward(self, x):
        out = self.conv(x) + self.shortcut(x)
        return out

class E_ResNet(nn.Module):
    def __init__(self, input_nc=3, output_nc=1, ndf=64, n_blocks=4,
                 norm_layer=None, nl_layer=None, vaeLike=False):
        super(E_ResNet, self).__init__()
        self.vaeLike = vaeLike
        max_ndf = 4
        conv_layers = [
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1, bias=True)]
        for n in range(1, n_blocks):
            input_ndf = ndf * min(max_ndf, n)
            output_ndf = ndf * min(max_ndf, n + 1)
            conv_layers += [BasicBlock(input_ndf,
                                       output_ndf, norm_layer, nl_layer)]
        conv_layers += [nl_layer(), nn.AvgPool2d(8)]
        if vaeLike:
            self.fc = nn.Sequential(*[nn.Linear(output_ndf, output_nc)])
            self.fcVar = nn.Sequential(*[nn.Linear(output_ndf, output_nc)])
        else:
            self.fc = nn.Sequential(*[nn.Linear(output_ndf, output_nc)])
        self.conv = nn.Sequential(*conv_layers)

    def forward(self, x):
        x_conv = self.conv(x)
        conv_flat = x_conv.view(x.size(0), -1)
        output = self.fc(conv_flat)
        if self.vaeLike:
            outputVar = self.fcVar(conv_flat)
            return output, outputVar
        else:
            return output
        return output

# Assume input range is [0, 1]
class MINCFeatureExtractor(nn.Module):
    def __init__(self, feature_layer=34, use_bn=False, use_input_norm=True, \
                device=torch.device('cpu')):
        super(MINCFeatureExtractor, self).__init__()

        self.features = MINCNet()
        self.features.load_state_dict(
            torch.load('../experiments/pretrained_models/VGG16minc_53.pth'), strict=True)
        self.features.eval()
        # No need to BP to variable
        for k, v in self.features.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        output = self.features(x)
        return output
