import torch
import torch.nn as nn
import numpy as np
import cv2
from collections import deque

EPSILON = 1e-30


def ValidStructTensorIndicator(a,d,b):
    # return 1-(a**2==d**2)*(b*(a+d)==0)
    return ((2 * b * (a + d))**2+(a ** 2 - d ** 2)**2)>EPSILON

def SVD_Symmetric_2x2(a,d,b):
    ATAN2_FACTOR = 10000
    theta = 0.5 * torch.atan2(ATAN2_FACTOR*2 * b * (a + d),ATAN2_FACTOR*(a ** 2 - d ** 2))
    FACTOR_4_NUMERIC_ISSUE = 10
    a,d,b = FACTOR_4_NUMERIC_ISSUE*(a.type(torch.cuda.DoubleTensor)),FACTOR_4_NUMERIC_ISSUE*(d.type(torch.cuda.DoubleTensor)),FACTOR_4_NUMERIC_ISSUE*(b.type(torch.cuda.DoubleTensor))
    S_1 = a ** 2 + d ** 2 + 2 * (b ** 2)
    S_2 = (a+d)*torch.sqrt((a-d)**2 + (2 * b) ** 2+EPSILON)
    S_1,S_2 = S_1/(FACTOR_4_NUMERIC_ISSUE**2),S_2/(FACTOR_4_NUMERIC_ISSUE**2)
    # S_2 = torch.min(S_1,S_2) # A patchy solution to a super-odd problem. I analitically showed S_1>=S_2 for ALL a,d,b, but for some reason ~20% of cases do not satisfy this. I suspect numerical reasons, so I enforce this here.
    lambda0 = torch.sqrt((S_1 + S_2) / 2+EPSILON).type(torch.cuda.FloatTensor)
    lambda1 = torch.sqrt((S_1 - S_2) / 2+EPSILON).type(torch.cuda.FloatTensor)
    # lambda0,lambda1 = torch.ones_like(lambda0),0.5*torch.ones_like(lambda1)
    return lambda0,lambda1,theta

class FilterLoss(nn.Module):
    def __init__(self,latent_channels,constant_Z=None,reference_images=None,masks=None):
        super(FilterLoss,self).__init__()
        self.LOCAL_LOSS_4_TEST = True

        self.latent_channels = latent_channels
        self.NOISE_STD = 1e-15#1/255
        self.model_training = constant_Z is None
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
        elif 'structure_tensor' in self.latent_channels:
            self.num_channels = 3
            self.NOISE_STD = 1e-7
            gradient_filters = [[[-1,1],[0,0]],[[-1,0],[1,0]]]
            self.filters = []
            for filter in gradient_filters:
                filter = np.array(filter)
                conv_layer = nn.Conv2d(in_channels=3,out_channels=3,kernel_size=filter.shape,bias=False,groups=3)
                conv_layer.weight = nn.Parameter(data=torch.from_numpy(np.tile(np.expand_dims(np.expand_dims(filter, 0), 0), reps=[3, 1, 1, 1])).type(torch.cuda.FloatTensor), requires_grad=False)
                conv_layer.filter_layer = True
                self.filters.append(conv_layer)
        else:
            raise Exception('Unknown latent channel setting %s' % (latent_channels))
        if self.model_training:
            self.collected_ratios = [deque(maxlen=10000) for i in range(self.num_channels)]
        else:
            self.HR_mask,self.LR_mask = masks['HR'],masks['LR']
            if self.LOCAL_LOSS_4_TEST:
                self.constant_Z = torch.nn.functional.interpolate(input=constant_Z,scale_factor=int(self.HR_mask.size(0)/self.LR_mask.size(0)),mode='bilinear',align_corners=False).to(self.LR_mask.device)[:,:,:-1,:-1]
            else:
                self.constant_Z = (constant_Z.to(self.LR_mask.device)*self.LR_mask).sum(dim=(2,3))/self.LR_mask.sum()
            reference_derivatives = {}
            for ref_image in reference_images.keys():
                reference_derivatives[ref_image] = []
                for filter in self.filters:
                    reference_derivatives[ref_image].append(filter(reference_images[ref_image]))
                reference_derivatives[ref_image] = torch.stack(reference_derivatives[ref_image], 0)
                reference_derivatives[ref_image] = torch.cat([reference_derivatives[ref_image]**2,torch.prod(reference_derivatives[ref_image],dim=0,keepdim=True)],0)
                if self.LOCAL_LOSS_4_TEST:
                    reference_derivatives[ref_image] = reference_derivatives[ref_image].mean(dim=2)
                else:
                    reference_derivatives[ref_image] = (reference_derivatives[ref_image].mean(dim=2)*self.HR_mask[:-1,:-1]).sum(dim=(2,3))/self.HR_mask[:-1,:-1].sum()
            if self.LOCAL_LOSS_4_TEST:
                reference_derivatives['tensor_normalizer'] = torch.sqrt(torch.prod(torch.stack([torch.mean(
                    torch.cat([reference_derivatives[ref_image][i] for ref_image in reference_images.keys()]),dim=0) for i in
                                                                                                range(2)]),dim=0))
            else:
                reference_derivatives['tensor_normalizer'] = torch.sqrt(torch.prod(torch.stack([torch.mean(torch.cat([reference_derivatives[ref_image][i] for ref_image in reference_images.keys()])) for i in range(2)]))).item()
            for ref_image in reference_images.keys():
                if self.LOCAL_LOSS_4_TEST:
                    reference_derivatives[ref_image] = [(val / (reference_derivatives['tensor_normalizer'] + self.NOISE_STD)) for val in reference_derivatives[ref_image]]
                else:
                    reference_derivatives[ref_image] = [(val/(reference_derivatives['tensor_normalizer']+self.NOISE_STD)).item() for val in reference_derivatives[ref_image]]
            self.reference_derivatives = reference_derivatives

    def forward(self, data):
        image_shape = list(data['SR'].size())
        LOWER_PERCENTILE,HIGHER_PERCENTILE = 5,95
        if self.model_training:
            cur_Z = data['Z'].mean(dim=(2,3))
        else:
            cur_Z = self.constant_Z
        if self.latent_channels == 'STD_1dir':
            dir_filter_output_SR = self.filter(data['SR'])
            dir_filter_output_HR = self.filter(data['HR'])
            dir_magnitude_ratio = dir_filter_output_SR.abs().mean(dim=(1, 2, 3)) / (
            dir_filter_output_HR.abs().mean(dim=(1, 2, 3)) + self.NOISE_STD)
            STD_ratio = data['SR'].contiguous().view(tuple(image_shape[:2] + [-1])).std(dim=-1).mean(1) /\
                        (data['HR'].contiguous().view(tuple(image_shape[:2] + [-1])).std(dim=-1).mean(1) + self.NOISE_STD)
            normalized_Z = []
            for ch_num in range(self.num_channels):
                self.collected_ratios[ch_num] += [val.item() for val in list(measured_values[:, ch_num])]
                upper_bound = np.percentile(self.collected_ratios[ch_num], HIGHER_PERCENTILE)
                lower_bound = np.percentile(self.collected_ratios[ch_num], LOWER_PERCENTILE)
                normalized_Z.append(
                    (cur_Z[:, ch_num]) / 2 * (upper_bound - lower_bound) + np.mean([upper_bound, lower_bound]))
            normalized_Z = torch.stack(normalized_Z, 1)
            measured_values = torch.stack([STD_ratio, dir_magnitude_ratio], 1)
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
            measured_values = torch.stack([STD_ratio, dir_magnitude_ratio], 1)
        elif'structure_tensor' in self.latent_channels:
            ZERO_CENTERED_IxIy = False
            if not self.model_training:
                RATIO_LOSS = 'No'
            elif self.latent_channels == 'SVD_structure_tensor':
                RATIO_LOSS = 'OnlyDiagonals'
            elif self.latent_channels=='SVDinNormedOut_structure_tensor':
                RATIO_LOSS = 'SingleNormalizer'
            else:
                RATIO_LOSS = 'OnlyDiagonals' #'No','All','OnlyDiagonals','Diagonals_IxIyRelative'
                assert not (RATIO_LOSS=='All' and ZERO_CENTERED_IxIy),'Do I want to combine these two flags?'
            derivatives_SR,derivatives_HR = [],[]
            for filter in self.filters:
                derivatives_SR.append(filter(data['SR']))
                if RATIO_LOSS!='No':
                    derivatives_HR.append(filter(data['HR']))
            non_squared_derivatives_SR = torch.stack(derivatives_SR,0)
            derivatives_SR = torch.cat([non_squared_derivatives_SR**2,torch.prod(non_squared_derivatives_SR,dim=0,keepdim=True)],0)
            if self.model_training:
                derivatives_SR = derivatives_SR.mean(dim=(2, 3,4))  # In ALL configurations, I also average all values before taking ratios - should think whether it might be a problem.
            else:
                if self.LOCAL_LOSS_4_TEST:
                    derivatives_SR = derivatives_SR.mean(dim=2)
                else:
                    derivatives_SR = (derivatives_SR.mean(dim=2)*self.HR_mask[:-1,:-1]).sum(dim=(2,3))/self.HR_mask[:-1,:-1].sum()
            if self.latent_channels == 'SVD_structure_tensor':
                lambda0_SR,lambda1_SR,theta_SR = SVD_Symmetric_2x2(*derivatives_SR)
                images_validity_4_backprop = ValidStructTensorIndicator(*derivatives_SR)
            else:
                measured_values = [derivatives_SR[i] for i in range(derivatives_SR.size(0))]
            if RATIO_LOSS!='No':
                non_squared_derivatives_HR = torch.stack(derivatives_HR, 0)
                derivatives_HR = torch.cat([non_squared_derivatives_HR**2,torch.prod(non_squared_derivatives_HR,dim=0,keepdim=True)],0)
                derivatives_HR = derivatives_HR.mean(dim=(2, 3, 4))
                if self.latent_channels == 'SVD_structure_tensor':
                    lambda0_HR, lambda1_HR, theta_HR = SVD_Symmetric_2x2(*derivatives_HR)
                    images_validity_4_backprop = images_validity_4_backprop*ValidStructTensorIndicator(*derivatives_HR)
                    measured_values = [lambda0_SR/(lambda0_HR+self.NOISE_STD),lambda1_SR/(lambda1_HR+self.NOISE_STD),theta_SR]
                    # measured_values = [val.mean(dim=(1,2,3)).to() for val in measured_values]
                elif self.latent_channels=='SVDinNormedOut_structure_tensor':
                    tensor_normalizer = torch.prod(torch.sqrt(derivatives_HR[:2]),dim=0)
                    measured_values = [measured_val/(tensor_normalizer+self.NOISE_STD) for measured_val in measured_values]
                else:
                    measured_values = [measured_values[i]/((derivatives_HR[i]+torch.sign(measured_values[i])*self.NOISE_STD)
                        if (i<2 or RATIO_LOSS=='All') else 1) for i in range(derivatives_SR.size(0))]
            elif not self.model_training:
                measured_values = [measured_val / (self.reference_derivatives['tensor_normalizer'] + self.NOISE_STD) for measured_val in measured_values]
            normalized_Z = []
            for i in range(len(measured_values)):
                if self.model_training:
                    self.collected_ratios[i] += [val.item() for val in measured_values[i]]
                    upper_bound = np.percentile(self.collected_ratios[i], HIGHER_PERCENTILE)
                    lower_bound = np.percentile(self.collected_ratios[i], LOWER_PERCENTILE)
                    if self.latent_channels in ['structure_tensor','SVDinNormedOut_structure_tensor']:
                        if i==2 and ZERO_CENTERED_IxIy:
                            upper_bound = np.max(np.abs([upper_bound, lower_bound]))
                            lower_bound = -1 * upper_bound
                        normalized_Z.append((cur_Z[:, i]) / 2 * (upper_bound - lower_bound) + np.mean([upper_bound, lower_bound]))
                    elif self.latent_channels == 'SVD_structure_tensor':
                        if i<2:
                            measured_values[i] = (measured_values[i]-np.mean([upper_bound, lower_bound]))/(upper_bound - lower_bound+EPSILON)+0.5
                            normalized_Z.append(data['SVD']['lambda%d_ratio'%(i)])
                        else:
                            measured_values[i] = measured_values[i]/np.pi
                            normalized_Z.append((torch.fmod(data['SVD']['theta'],torch.tensor(np.pi))-np.pi/2)/np.pi)
                else:
                    if self.LOCAL_LOSS_4_TEST:
                        normalized_Z.append((cur_Z[:, i]) / 2 * (self.reference_derivatives['max'][i] - self.reference_derivatives['min'][i]) +
                            torch.mean(torch.cat([self.reference_derivatives['max'][i], self.reference_derivatives['min'][i]],0),0,keepdim=True))
                    else:
                        normalized_Z.append((cur_Z[:, i]) / 2 * (self.reference_derivatives['max'][i] - self.reference_derivatives['min'][i]) + np.mean([self.reference_derivatives['max'][i],self.reference_derivatives['min'][i]]))

            if self.latent_channels in ['structure_tensor','SVDinNormedOut_structure_tensor']:
                measured_values = torch.stack(measured_values,1)
                normalized_Z = torch.stack(normalized_Z, 1)
        if self.latent_channels == 'SVD_structure_tensor':
            normalized_Z = [item.mean(dim=(1,2)).to(measured_values[0].device) for item in normalized_Z]
            abs_differences = []
            for i in range(2):
                abs_differences.append((measured_values[i]-normalized_Z[i]).abs())
            abs_differences.append(torch.min(torch.min((measured_values[2]-normalized_Z[2]).abs(),(measured_values[2]-normalized_Z[2]+np.pi).abs()),(measured_values[2]-normalized_Z[2]-np.pi).abs()))
            if images_validity_4_backprop.float().sum()>=1:
                return torch.mean(torch.stack(abs_differences,1)[images_validity_4_backprop],dim=0)
            else:
                return torch.zeros([3]).type(abs_differences[0].dtype).view([-1])
        else:
            return torch.mean((measured_values-normalized_Z).abs(),dim=0)


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
