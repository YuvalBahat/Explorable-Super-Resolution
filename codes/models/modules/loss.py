import torch
import torch.nn as nn
import numpy as np
import cv2
from collections import deque

class FilterLoss(nn.Module):
    def __init__(self,latent_channels):
        super(FilterLoss,self).__init__()
        self.latent_channels = latent_channels
        self.NOISE_STD = 1/255
        if latent_channels == 'STD_1dir':#Channel 0 controls STD, channel 1 controls horizontal Sobel
            self.num_channels = 2
            DELTA_SIZE = 7
            delta_im = np.zeros([DELTA_SIZE,DELTA_SIZE]); delta_im[DELTA_SIZE//2,DELTA_SIZE//2] = 1;
            dir_filter = cv2.Sobel(delta_im,ddepth=cv2.CV_64F,dx=1,dy=0)
            filter_margins = np.argwhere(np.any(dir_filter!=0,0))[0][0]
            dir_filter = dir_filter[filter_margins:-filter_margins,filter_margins:-filter_margins]
            self.filter = nn.Conv2d(in_channels=3,out_channels=3,kernel_size=dir_filter.shape,bias=False,groups=3)
            self.filter.weight = nn.Parameter(data=torch.from_numpy(np.tile(np.expand_dims(np.expand_dims(dir_filter, 0), 0), reps=[3, 1, 1, 1])).type(torch.cuda.FloatTensor), requires_grad=False)
            self.filter.filter_layer = True
        elif latent_channels=='STD_directional':#Channel 1-2 control the energy of a specific directional derivative, channel 0 controls the remainder of energy (all other directions)
            self.num_channels = 3
        elif self.latent_channels == 'structure_tensor':
            self.num_channels = 3
            gradient_filters = [[[1,-1],[0,0]],[[1,0],[-1,0]]]
            self.filters = []
            for filter in gradient_filters:
                conv_layer = nn.Conv2d(in_channels=3,out_channels=3,kernel_size=filter.shape,bias=False,groups=3)
                conv_layer.weight = nn.Parameter(data=torch.from_numpy(np.tile(np.expand_dims(np.expand_dims(filter, 0), 0), reps=[3, 1, 1, 1])).type(torch.cuda.FloatTensor), requires_grad=False)
                conv_layer.filter_layer = True
                self.filters.append(conv_layer)
        else:
            raise Exception('Unknown latent channel setting %s' % (opt['network_G']['latent_channels']))
        # self.dynamic_range = torch.ones([2,self.num_channels]).cuda()
        self.collected_ratios = [deque(maxlen=10000) for i in range(self.num_channels)]

        # if filter is not None:
        # else:
        #     self.filter = None

    def forward(self, data):
        image_shape = list(data['HR'].size())
        LOWER_PERCENTILE,HIGHER_PERCENTILE = 5,95
        cur_Z = data['Z'].mean(dim=(2,3))
        if self.latent_channels == 'STD_1dir':
            dir_filter_output_SR = self.filter(data['SR'])
            dir_filter_output_HR = self.filter(data['HR'])
            dir_magnitude_ratio = dir_filter_output_SR.abs().mean(dim=(1, 2, 3)) / (
            dir_filter_output_HR.abs().mean(dim=(1, 2, 3)) + self.NOISE_STD)
            STD_ratio = data['SR'].contiguous().view(tuple(image_shape[:2] + [-1])).std(dim=-1).mean(1) /\
                        (data['HR'].contiguous().view(tuple(image_shape[:2] + [-1])).std(dim=-1).mean(1) + self.NOISE_STD)
            normalized_Z = []
            for ch_num in range(self.num_channels):
                self.collected_ratios[ch_num] += [val.item() for val in list(actual_ratios[:, ch_num])]
                upper_bound = np.percentile(self.collected_ratios[ch_num], HIGHER_PERCENTILE)
                lower_bound = np.percentile(self.collected_ratios[ch_num], LOWER_PERCENTILE)
                normalized_Z.append(
                    (cur_Z[:, ch_num]) / 2 * (upper_bound - lower_bound) + np.mean([upper_bound, lower_bound]))
            normalized_Z = torch.stack(normalized_Z, 1)
        elif self.latent_channels == 'STD_directional':
            horizontal_derivative_SR = (data['SR'][:,:,:,2:]-data['SR'][:,:,:,:-2])[:,:,1:-1,:].unsqueeze(1)/2
            vertical_derivative_SR = (data['SR'][:, :, 2:,:] - data['SR'][:, :, :-2, :])[:,:,:,1:-1].unsqueeze(1)/2
            horizontal_derivative_HR = (data['HR'][:, :, :, 2:] - data['HR'][:, :, :, :-2])[:,:,1:-1,:].unsqueeze(1)/2
            vertical_derivative_HR = (data['HR'][:, :, 2:, :] - data['HR'][:, :, :-2, :])[:,:,:,1:-1].unsqueeze(1)/2
            dir_normal = cur_Z[:,1:3]
            dir_normal = dir_normal/torch.sqrt(torch.sum(dir_normal**2,dim=1,keepdim=True))
            dir_filter_output_SR = (dir_normal.unsqueeze(2).unsqueeze(3).unsqueeze(4)*torch.cat([horizontal_derivative_SR,vertical_derivative_SR],dim=1)).sum(1)
            dir_filter_output_HR = (dir_normal.unsqueeze(2).unsqueeze(3).unsqueeze(4) * torch.cat([horizontal_derivative_HR, vertical_derivative_HR],dim=1)).sum(1)
            dir_magnitude_ratio = dir_filter_output_SR.abs().mean(dim=(1,2,3))/(dir_filter_output_HR.abs().mean(dim=(1,2,3))+self.NOISE_STD)
            self.collected_ratios[1] += [val.item() for val in list(dir_magnitude_ratio)]
            STD_ratio = (data['SR'][:,:,1:-1,1:-1]-dir_filter_output_SR).abs().mean(dim=(1,2,3))/((data['HR'][:,:,1:-1,1:-1]-dir_filter_output_HR).abs().mean(dim=(1,2,3))+self.NOISE_STD)
            self.collected_ratios[0] += [val.item() for val in list(STD_ratio)]
            STD_upper_bound = np.percentile(self.collected_ratios[0], HIGHER_PERCENTILE)
            STD_lower_bound = np.percentile(self.collected_ratios[0],LOWER_PERCENTILE)
            dir_magnitude_upper_bound = np.percentile(self.collected_ratios[1], HIGHER_PERCENTILE)
            dir_magnitude_lower_bound = np.percentile(self.collected_ratios[1], LOWER_PERCENTILE)
            mag_normal = (cur_Z[:,1:3]**2).sum(1).sqrt()
            normalized_Z = torch.stack([cur_Z[:,0]*(STD_upper_bound-STD_lower_bound)+np.mean([STD_upper_bound,STD_lower_bound]),
                                        mag_normal/np.sqrt(2)*(dir_magnitude_upper_bound-dir_magnitude_lower_bound)+np.mean([dir_magnitude_upper_bound,dir_magnitude_lower_bound])],1)
        elif self.latent_channels == 'structure_tensor':
            pass
        actual_ratios = torch.stack([STD_ratio,dir_magnitude_ratio],1)

        # self.dynamic_range[0,:] = torch.min(input=self.dynamic_range[0,:],other=torch.min(actual_ratios,dim=0)[0]).detach()
        # self.dynamic_range[1,:] = torch.max(input=self.dynamic_range[1,:],other=torch.max(actual_ratios,dim=0)[0]).detach()
        # normalized_Z = []
        # for ch_num in range(self.num_channels):
        #     self.collected_ratios[ch_num] += [val.item() for val in list(actual_ratios[:,ch_num])]
        #     upper_bound = np.percentile(self.collected_ratios[ch_num],HIGHER_PERCENTILE)
        #     lower_bound = np.percentile(self.collected_ratios[ch_num],LOWER_PERCENTILE)
        #     normalized_Z.append((cur_Z[:,ch_num])/2*(upper_bound-lower_bound)+np.mean([upper_bound,lower_bound]))
        # normalized_Z = torch.stack(normalized_Z,1)
        # normalized_Z = data['Z'].mean(dim=(2,3))/2*((self.dynamic_range[1]-self.dynamic_range[0]).unsqueeze(0))+self.dynamic_range.mean(dim=0,keepdim=True) # Assuming uniform D per image per channel, in range [-1,1]. Normalizing to the range [0.9,1.1]
        return torch.mean((actual_ratios-normalized_Z).abs())


# Define GAN loss: [vanilla | lsgan | wgan-gp]
class GANLoss(nn.Module):
    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type.lower()
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan-gp':

            def wgan_loss(input, target):
                # target is boolean
                return -1 * input.mean() if target else input.mean()

            self.loss = wgan_loss
        else:
            raise NotImplementedError('GAN type [{:s}] is not found'.format(self.gan_type))

    def get_target_label(self, input, target_is_real):
        if self.gan_type == 'wgan-gp':
            return target_is_real
        if target_is_real:
            return torch.empty_like(input).fill_(self.real_label_val)
        else:
            return torch.empty_like(input).fill_(self.fake_label_val)

    def forward(self, input, target_is_real):
        target_label = self.get_target_label(input, target_is_real)
        loss = self.loss(input, target_label)
        return loss

def CreateRangeLoss(legit_range):
    dtype = torch.cuda.FloatTensor
    legit_range = torch.FloatTensor(legit_range).type(dtype)
    def RangeLoss(x):
        return torch.max(torch.max(x-legit_range[1],other=torch.zeros(size=[1]).type(dtype)),other=torch.max(legit_range[0]-x,other=torch.zeros(size=[1]).type(dtype))).mean()
    return RangeLoss

class GradientPenaltyLoss(nn.Module):
    def __init__(self, device=torch.device('cpu')):
        super(GradientPenaltyLoss, self).__init__()
        self.register_buffer('grad_outputs', torch.Tensor())
        self.grad_outputs = self.grad_outputs.to(device)

    def get_grad_outputs(self, input):
        if self.grad_outputs.size() != input.size():
            self.grad_outputs.resize_(input.size()).fill_(1.0)
        return self.grad_outputs

    def forward(self, interp, interp_crit):
        grad_outputs = self.get_grad_outputs(interp_crit)
        grad_interp = torch.autograd.grad(outputs=interp_crit, inputs=interp, \
            grad_outputs=grad_outputs, create_graph=True, retain_graph=True, only_inputs=True)[0]
        grad_interp = grad_interp.view(grad_interp.size(0), -1)
        grad_interp_norm = grad_interp.norm(2, dim=1)

        loss = ((grad_interp_norm - 1)**2).mean()
        return loss
