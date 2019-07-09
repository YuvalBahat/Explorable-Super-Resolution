import os
import math
from datetime import datetime
import numpy as np
import cv2
from torchvision.utils import make_grid
import GPUtil
import time
from skimage.transform import resize
from scipy.signal import convolve2d
from scipy.ndimage.morphology import binary_opening
from sklearn.feature_extraction.image import extract_patches_2d
import torch
from models.modules.loss import GANLoss,FilterLoss

####################
# miscellaneous
####################


def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def mkdirs(paths):
    if isinstance(paths, str):
        mkdir(paths)
    else:
        for path in paths:
            mkdir(path)

def mkdir_and_rename(path):
    if os.path.exists(path):
        # new_name = path + '_archived_' + get_timestamp()
        # print('Path already exists. Rename it to [{:s}]'.format(new_name))
        # os.rename(path, new_name)
        renamed_path = path + '_Renamed' + get_timestamp()
        os.rename(path,renamed_path)
        print('Path already exists. Changing to [{:s}]'.format(renamed_path))
    os.makedirs(path)

def Assign_GPU(max_GPUs=1):
    excluded_IDs = []
    GPU_2_use = GPUtil.getAvailable(order='memory',excludeID=excluded_IDs,limit=max_GPUs if max_GPUs is not None else 100)
    if len(GPU_2_use)==0:
        print('No available GPUs. waiting...')
        while len(GPU_2_use)==0:
            time.sleep(10)
            GPU_2_use = GPUtil.getAvailable(order='memory', excludeID=excluded_IDs)
    assert len(GPU_2_use)>0,'No available GPUs...'
    if max_GPUs is not None:
        print('Using GPU #%d'%(GPU_2_use[0]))
        os.environ["CUDA_VISIBLE_DEVICES"] = "%d"%(GPU_2_use[0]) # Limit to 1 GPU when using an interactive session
        return [GPU_2_use[0]]
    else:
        return GPU_2_use

def SVD_Symmetric_2x2(a,d,b):
    EPSILON = 1e-30
    ATAN2_FACTOR = 10000
    theta = 0.5 * torch.atan2(ATAN2_FACTOR*2 * b * (a + d),ATAN2_FACTOR*(a ** 2 - d ** 2)).type(torch.cuda.FloatTensor)
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

####################
# image convert
####################
def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)


def save_img(img, img_path, mode='RGB'):
    cv2.imwrite(img_path, img)

def Convert_Im_2_Zinput(Z_image,im_size,Z_range,single_channel=False):
    SMOOTHING_WIN_SIZE = 5
    Z_image = resize(Z_image,im_size)
    if single_channel:
        Z_image = np.mean(Z_image,2,keepdims=True)
    if np.any(np.std(Z_image,(0,1))>0):
        Z_image = (Z_image-np.min(Z_image))/(np.max(Z_image)-np.min(Z_image))*2*Z_range-Z_range
        pad_size = SMOOTHING_WIN_SIZE//2
        for c_num in range(Z_image.shape[2]):
            Z_image[:,:,c_num] = convolve2d(np.pad(Z_image[:,:,c_num],[[pad_size,pad_size],[pad_size,pad_size]],mode='edge'),
                                            np.ones([SMOOTHING_WIN_SIZE,SMOOTHING_WIN_SIZE]),mode='valid')/SMOOTHING_WIN_SIZE**2
    else:
        Z_image = Z_image*2*Z_range-Z_range
    return np.expand_dims(Z_image.transpose((2,0,1)),0)

def crop_center(image,margins):
    if margins[0]>0:
        image = image[margins[0]:-margins[0],...]
    if margins[1]>0:
        image = image[:,margins[1]:-margins[1],...]
    return  image

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def SVD_2_LatentZ(SVD_values,max_lambda=1):
    # Given SVD values, returns corresponding structural tensor values.
    # SVD values: lambda0 in [0,max_lambda], lambda1 in [0,max_lambda], theta in [0,2*pi]
    # Returned values: Signa I_x^2 in [-max_lambda,max_lambda], Signa I_y^2 in [-max_lambda,max_lambda], Sigma I_x*I_y in (not sure, should calculate, but a symmetric range).
    return torch.stack([2*max_lambda*(SVD_values[:,1,...]*(torch.sin(SVD_values[:,-1,...])**2)+SVD_values[:,0,...]*(torch.cos(SVD_values[:,-1,...])**2))-max_lambda,
                                  2*max_lambda*(SVD_values[:,0,...]*(torch.sin(SVD_values[:,-1,...])**2)+SVD_values[:,1,...]*(torch.cos(SVD_values[:,-1,...])**2))-max_lambda,#Normalizing range to have negative values as well,trying to match [-1,1]
                                  2*(SVD_values[:,0,...]-SVD_values[:,1,...])*torch.sin(SVD_values[:,-1,...])*torch.cos(SVD_values[:,-1,...])],1)

class Optimizable_Temperature(torch.nn.Module):
    def __init__(self,initial_temperature=None):
        super(Optimizable_Temperature,self).__init__()
        self.log_temperature = torch.nn.Parameter(data=torch.zeros([1]).type(torch.cuda.DoubleTensor))
        if initial_temperature is not None:
            self.log_temperature.data = torch.log(torch.tensor(initial_temperature).type(torch.cuda.DoubleTensor))

    def forward(self):
        return torch.exp(self.log_temperature)

class SoftHistogramLoss(torch.nn.Module):
    def __init__(self,bins,min,max,desired_hist_image_mask=None,desired_hist_image=None,gray_scale=True,input_im_HR_mask=None,patch_size=1,automatic_temperature=False,
            image_Z=None,temperature=0.05,dictionary_not_histogram=False):
        OVERLAPPING_PATCHES = False
        self.temperature = temperature#0.05**2#0.006**6
        self.exp_power = 2#6
        self.SQRT_EPSILON = 1e-7

        super(SoftHistogramLoss,self).__init__()
        # min correspond to the CENTER of the first bin, and max to the CENTER of the last bin
        self.device = torch.device('cuda')
        self.bin_width = (max-min)/(bins-1)
        self.max = max
        # self.temperature = torch.tensor(self.temperature,requires_grad=True).type(torch.cuda.DoubleTensor)
        self.temperature_optimizer = automatic_temperature
        if automatic_temperature:
            self.optimizable_temperature = Optimizable_Temperature(self.temperature)
            self.image_Z = image_Z
        else:
            self.temperature = torch.tensor(self.temperature).type(torch.cuda.DoubleTensor)
            # self.temperature = self.temperature*self.bin_width*85 # 0.006 was calculated for pixels range [0,1] with 85 bins. So I adjust it acording to actual range and n_bins, manifested in self.bin_width
        self.bin_centers = torch.linspace(min,max,bins)
        self.gray_scale = gray_scale
        self.patch_size = patch_size
        self.num_dims = 3
        self.KDE = not gray_scale or patch_size>1 # Using Kernel Density Estimation rather than histogram
        if gray_scale:
            self.num_dims = self.num_dims//3
            self.bins = 1. * self.bin_centers.view([1] + list(self.bin_centers.size())).type(torch.cuda.DoubleTensor)
            if desired_hist_image is not None:
                desired_hist_image = desired_hist_image.mean(1, keepdim=True).view([-1,1])
        if patch_size>1:
            assert gray_scale and (desired_hist_image is not None),'Not supporting color images or patch histograms for model training loss for now'
            self.num_dims = patch_size**2
            desired_im_patch_extraction_mat = self.ReturnPatchExtractionMat(desired_hist_image_mask).to(self.device)
            desired_hist_image = torch.sparse.mm(desired_im_patch_extraction_mat,desired_hist_image).view([self.num_dims,-1,1])
            self.desired_hist_image_mask = None
        else:
            self.desired_hist_image_mask = torch.from_numpy(desired_hist_image_mask).to(self.device).view([-1]).type(torch.ByteTensor) if desired_hist_image_mask is not None else None
            if desired_hist_image is not None:
                desired_hist_image = 1 * desired_hist_image.view([self.num_dims, -1, 1])
        if self.KDE:
            # The bins are now simply the multi-dimensional pixels/patches. So now I remove redundant bins, by checking if there is duplicacy:
            if self.desired_hist_image_mask is not None:
                desired_hist_image = desired_hist_image[:,self.desired_hist_image_mask,:]
            self.bins = self.Desired_Im_2_Bins(desired_hist_image)
        if not dictionary_not_histogram:
            self.loss = torch.nn.KLDivLoss()
        if patch_size>1:
            self.patch_extraction_mat = self.ReturnPatchExtractionMat(input_im_HR_mask.data.cpu().numpy(),patches_overlap=OVERLAPPING_PATCHES).to(self.device)
            self.image_mask = None
        else:
            self.image_mask = input_im_HR_mask.view([-1]).type(torch.ByteTensor) if input_im_HR_mask is not None else None
        self.dictionary_not_histogram = dictionary_not_histogram
        if not dictionary_not_histogram:
            if not automatic_temperature and desired_hist_image is not None:
                with torch.no_grad():
                    self.desired_hists_list = [self.ComputeSoftHistogram(desired_hist_image,image_mask=self.desired_hist_image_mask,return_log_hist=False,
                                                                  reshape_image=False,compute_hist_normalizer=True).detach()]
            else:
                self.desired_hist_image = desired_hist_image

    def Feed_Desired_Hist_Im(self,desired_hist_image):
        self.desired_hists_list = []
        for desired_im in desired_hist_image:
            if self.gray_scale:
                desired_im = desired_im.mean(0, keepdim=True).view([1,-1, 1])
            with torch.no_grad():
                self.desired_hists_list.append(self.ComputeSoftHistogram(desired_im,image_mask=self.desired_hist_image_mask,return_log_hist=False,
                                                                  reshape_image=False,compute_hist_normalizer=True).detach())

    def Desired_Im_2_Bins(self,desired_im):
        bins = 1*desired_im
        repeated_elements_mat = (desired_im.view([self.num_dims, -1, 1]) - desired_im.view([desired_im.size(0)] + [1, -1])).abs()
        repeated_elements_mat = (repeated_elements_mat < self.bin_width / 2).all(0)
        repeated_elements_mat = torch.mul(repeated_elements_mat,(1 - torch.diag(torch.ones([repeated_elements_mat.size(0)]))).type(
                                              repeated_elements_mat.dtype).to(repeated_elements_mat.device))
        repeated_elements_mat = torch.triu(repeated_elements_mat).any(1) ^ 1
        bins = bins[:, repeated_elements_mat]
        bins = bins.view([desired_im.size(0), 1, -1]).type(torch.cuda.DoubleTensor)
        return bins

    def TemperatureSearch(self,desired_image,initial_image,desired_KL_div):
        log_temperature_range = [0.1,1]
        STEP_SIZE = 10
        KL_DIV_TOLERANCE = 0.1
        cur_KL_div = []
        desired_temp_within_range = False
        with torch.no_grad():
            while True:
                next_temperature = np.exp(np.mean(log_temperature_range))
                if np.isinf(next_temperature) or next_temperature==0:
                    print('KL div. is %.3e even for temperature of %.3e, aborting temperature search with that.'%(cur_KL_div[-1],self.temperature))
                    break
                self.temperature = 1*next_temperature
                desired_im_hist = self.ComputeSoftHistogram(desired_image,image_mask=self.desired_hist_image_mask,return_log_hist=False,reshape_image=False,compute_hist_normalizer=True)
                initial_image_hist = self.ComputeSoftHistogram(initial_image,image_mask=self.image_mask,return_log_hist=True,reshape_image=False,compute_hist_normalizer=False)
                cur_KL_div.append(self.loss(initial_image_hist,desired_im_hist).item())
                KL_div_too_big = cur_KL_div[-1] > desired_KL_div
                if np.abs(np.log(max([0,cur_KL_div[-1]])/desired_KL_div))<=np.log(1+KL_DIV_TOLERANCE):
                    print('Automatically set histogram temperature to %.3e'%(self.temperature))
                    break
                elif not desired_temp_within_range:
                    if len(cur_KL_div)==1:
                        initial_KL_div_too_big = KL_div_too_big
                    else:
                        desired_temp_within_range = initial_KL_div_too_big^KL_div_too_big
                    if not desired_temp_within_range:
                        if KL_div_too_big:
                            log_temperature_range[1] += STEP_SIZE
                        else:
                            log_temperature_range[0] -= STEP_SIZE
                if desired_temp_within_range:
                    if KL_div_too_big:
                        log_temperature_range[0] = 1*np.log(self.temperature)
                    else:
                        log_temperature_range[1] = 1*np.log(self.temperature)

    def ReturnPatchExtractionMat(self,mask,patches_overlap=True):
        mask = binary_opening(mask,np.ones([self.patch_size, self.patch_size]).astype(np.bool))
        patches_indexes = extract_patches_2d(np.multiply(mask,1 + np.arange(mask.size).reshape(mask.shape)),(self.patch_size, self.patch_size)).reshape([-1, self.num_dims])
        patches_indexes = patches_indexes[np.all(patches_indexes > 0, 1), :] - 1
        if not patches_overlap:
            unique_indexes = list(set(list(patches_indexes.reshape([-1]))))
            min_index = min(unique_indexes)
            index_taken_indicator = np.zeros([max(unique_indexes)-min(unique_indexes)]).astype(np.bool)
            valid_patches = np.ones([patches_indexes.shape[0]]).astype(np.bool)
            for patch_num,patch in enumerate(patches_indexes):
                if np.any(index_taken_indicator[patch-min_index-1]):
                    valid_patches[patch_num] = False
                    continue
                index_taken_indicator[patch - min_index-1] = True
            patches_indexes = patches_indexes[valid_patches]
        corresponding_mat_rows = np.arange(patches_indexes.size).reshape([-1])
        patch_extraction_mat = torch.sparse.FloatTensor(torch.LongTensor([corresponding_mat_rows, patches_indexes.transpose().reshape([-1])]),
            torch.FloatTensor(np.ones([corresponding_mat_rows.size])),torch.Size([patches_indexes.size, mask.size]))
        return patch_extraction_mat

    def ComputeSoftHistogram(self,image,image_mask,return_log_hist,reshape_image,compute_hist_normalizer,temperature=None):
        CANONICAL_KDE_4_DICTIONARY = True
        if temperature is None:
            temperature = 1*self.temperature
        if not reshape_image:
            image = image.type(torch.cuda.DoubleTensor)
        else:
            if self.patch_size > 1:
                image = torch.sparse.mm(self.patch_extraction_mat, image.view([-1, 1])).view([self.num_dims, -1])
            else:
                image = image.contiguous().view([self.num_dims,-1])
                if image_mask is not None:
                    image = image[:, image_mask]
            image = image.unsqueeze(-1).type(torch.cuda.DoubleTensor)
        hist = (image-self.bins).abs()
        hist = torch.min(hist,(image-self.bins-self.max).abs())
        hist = torch.min(hist,(image-self.bins+self.max).abs())
        if not self.dictionary_not_histogram or CANONICAL_KDE_4_DICTIONARY:
            hist = -((hist+self.SQRT_EPSILON)**self.exp_power)/temperature
        hist = hist.mean(0)
        if self.dictionary_not_histogram and not CANONICAL_KDE_4_DICTIONARY:
            # return torch.exp(self.bin_width/(hist+self.bin_width/2))
            return hist.min(dim=1)[0].view([1,-1])
            # return hist.min(dim=1)[0].view([1, -1])
        if self.dictionary_not_histogram and CANONICAL_KDE_4_DICTIONARY:
            return -1*torch.log(torch.exp(hist).mean(1)).view([1, -1])
        hist = torch.exp(hist).mean(0)
        if compute_hist_normalizer or not self.KDE:
            self.normalizer = hist.sum()/image.size(1)
        hist = (hist/self.normalizer/image.size(1)).type(torch.cuda.FloatTensor)
        if self.KDE: # Adding another "bin" to account for all other missing bins
            hist = torch.cat([hist,(1-torch.min(torch.tensor(1).type(hist.dtype).to(hist.device),hist.sum())).view([1])])
        if return_log_hist:
            return torch.log(hist+torch.finfo(hist.dtype).eps).view([1,-1])
        else:
            return hist.view([1,-1])

    def forward(self,cur_images):
        cur_images_hists,KLdiv_grad_sizes = [],[]
        for i,cur_image in enumerate(cur_images):
            if self.gray_scale:
                cur_image = cur_image.mean(0, keepdim=True)
            if self.temperature_optimizer:
                self.temperature = self.optimizable_temperature()
                self.desired_hists_list.append(self.ComputeSoftHistogram(self.desired_hist_image, image_mask=self.desired_hist_image_mask,return_log_hist=False,
                                                              reshape_image=False, compute_hist_normalizer=True))
            else:
                temperature = self.temperature*(1 if len(cur_images)==1 else 5**(i-1))
            cur_images_hists.append(self.ComputeSoftHistogram(cur_image, self.image_mask, return_log_hist=True,reshape_image=True, compute_hist_normalizer=False,temperature=temperature))
            if self.temperature_optimizer:
                KLdiv_grad_sizes.append(-1*(torch.autograd.grad(outputs=self.loss(cur_images_hists[-1],self.desired_hists_list[-1]),inputs=self.image_Z,create_graph=True)[0]).norm(p=2))
        if self.temperature_optimizer:
            return self.loss(torch.cat(cur_images_hists,0),torch.cat(self.desired_hists_list,0)),torch.stack(KLdiv_grad_sizes).mean()
        elif self.dictionary_not_histogram:
            return torch.cat(cur_images_hists,0).mean(1)
        else:
            return self.loss(torch.cat(cur_images_hists,0),torch.cat(self.desired_hists_list,0))

class Optimizable_Z(torch.nn.Module):
    def __init__(self,Z_shape,Z_range=None,initial_pre_tanh_Z=None,Z_mask=None,random_perturbations=False):
        super(Optimizable_Z, self).__init__()
        # self.device = torch.device('cuda')
        self.Z = torch.nn.Parameter(data=torch.zeros(Z_shape).type(torch.cuda.FloatTensor))
        if Z_mask is not None and not np.all(Z_mask):
            self.mask = torch.from_numpy(Z_mask).type(torch.cuda.FloatTensor).to(self.Z.data.device)
            self.initial_pre_tanh_Z = 1*initial_pre_tanh_Z.type(torch.cuda.FloatTensor).to(self.Z.data.device)
        else:
            self.mask = None
        if initial_pre_tanh_Z is not None:
            assert initial_pre_tanh_Z.size()==self.Z.data.size(),'Initilizer size does not match desired Z size'
            if random_perturbations:
                initial_pre_tanh_Z += torch.normal(mean=torch.zeros_like(initial_pre_tanh_Z), std=0.001 * torch.ones_like(initial_pre_tanh_Z))
            self.Z.data = initial_pre_tanh_Z.to(self.Z.data.device)
        self.Z_range = Z_range
        if Z_range is not None:
            self.tanh = torch.nn.Tanh()


    def forward(self):
        if self.Z_range is not None:
            self.Z.data = torch.min(torch.max(self.Z,torch.tensor(-torch.finfo(self.Z.dtype).max).type(self.Z.dtype).to(self.Z.device)),torch.tensor(torch.finfo(self.Z.dtype).max).type(self.Z.dtype).to(self.Z.device))
        if self.mask is not None:
            self.Z.data = self.mask * self.Z.data + (1 - self.mask) * self.initial_pre_tanh_Z
        if self.Z_range is not None:
            return self.Z_range*self.tanh(self.Z)
        else:
            return self.Z

    def PreTanhZ(self):
        if self.mask is not None:
            return self.mask * self.Z.data + (1 - self.mask) * self.initial_pre_tanh_Z
        else:
            return self.Z.data

    def Randomize_Z(self):
        torch.nn.init.xavier_uniform_(self.Z.data,gain=100)

    def Return_Detached_Z(self):
        return self.forward().detach()

def ArcTanH(input_tensor):
    return 0.5*torch.log((1+input_tensor+torch.finfo(input_tensor.dtype).eps)/(1-input_tensor+torch.finfo(input_tensor.dtype).eps))

class Z_optimizer():
    MIN_LR = 1e-5
    def __init__(self,objective,Z_size,model,Z_range,max_iters,data=None,loggers=None,image_mask=None,Z_mask=None,initial_Z=None,initial_LR=None,existing_optimizer=None,
                 batch_size=1,HR_unpadder=None,auto_set_hist_temperature=False,random_Z_inits=False):
        if (initial_Z is not None or 'cur_Z' in model.__dict__.keys()):
            if initial_Z is None:
                initial_Z = 1*model.GetLatent()
            initial_pre_tanh_Z = initial_Z/Z_range
            initial_pre_tanh_Z = torch.clamp(initial_pre_tanh_Z,min=-1+torch.finfo(initial_pre_tanh_Z.dtype).eps,max=1.-torch.finfo(initial_pre_tanh_Z.dtype).eps)
            initial_pre_tanh_Z = ArcTanH(initial_pre_tanh_Z)
            # if random_Z_inits and 'random' not in objective or ('random' in objective and 'limited' in objective):
            #     initial_pre_tanh_Z += torch.normal(mean=torch.zeros_like(initial_pre_tanh_Z),std=0.01*torch.ones_like(initial_pre_tanh_Z))
        else:
            initial_pre_tanh_Z = None
        self.Z_model = Optimizable_Z(Z_shape=[batch_size,model.num_latent_channels] + list(Z_size), Z_range=Z_range,initial_pre_tanh_Z=initial_pre_tanh_Z,Z_mask=Z_mask,
            random_perturbations=(random_Z_inits and 'random' not in objective) or ('random' in objective and 'limited' in objective))
        assert (initial_LR is not None) or (existing_optimizer is not None),'Should either supply optimizer from previous iterations or initial LR for new optimizer'
        self.objective = objective
        self.data = data
        self.device = torch.device('cuda')
        self.model = model
        if image_mask is None:
            if 'fake_H' in model.__dict__.keys():
                self.image_mask = torch.ones(list(model.fake_H.size()[2:])).type(model.fake_H.dtype).to(self.device)
            else:
                self.image_mask = None
            self.Z_mask = None#torch.ones(Z_size).type(model.fake_H.dtype).to(self.device)
        else:
            assert Z_mask is not None,'Should either supply both masks or niether'
            self.image_mask = torch.from_numpy(image_mask).type(model.fake_H.dtype).to(self.device)
            self.Z_mask = torch.from_numpy(Z_mask).type(model.fake_H.dtype).to(self.device)
            self.initial_Z = 1.*model.GetLatent()
            self.image_mask.requires_grad = False
            self.Z_mask.requires_grad = False
        if 'l1' in objective and 'random' not in objective:
            if data is not None and 'HR' in data.keys():
                self.GT_HR = data['HR']
            if self.image_mask is None:
                self.loss = torch.nn.L1Loss().to(torch.device('cuda'))
            else:
                def Masked_L1_loss(produced_im,GT_im):
                    return torch.nn.functional.l1_loss(input=produced_im*(self.image_mask).to(self.device),target=GT_im*(self.image_mask).to(self.device)).to(torch.device('cuda'))
                self.loss = Masked_L1_loss
            # scheduler_threshold = 1e-2
        elif 'desired_SVD' in objective:
            self.loss = FilterLoss(latent_channels='SVDinNormedOut_structure_tensor',constant_Z=data['desired_Z'],
                                   reference_images={'min':data['reference_image_min'],'max':data['reference_image_max']},masks={'LR':self.Z_mask,'HR':self.image_mask})
        elif 'STD' in objective:
            assert self.objective in ['max_STD', 'min_STD']
            # scheduler_threshold = 0.9999
        elif 'VGG' in objective and 'random' not in objective:
            self.GT_HR_VGG = model.netF(self.GT_HR).detach().to(self.device)
            self.loss = torch.nn.L1Loss().to(torch.device('cuda'))
        elif any([phrase in objective for phrase in ['hist','dict']]):
            self.automatic_temperature = auto_set_hist_temperature
            if self.automatic_temperature:
                assert 'hist' in objective,'Unsupported  for dictionary'
                self.data['Z'] = self.Z_model()
                pre_tanh_Z = self.Z_model.Z
                pre_tanh_Z.requires_grad = True
                model.feed_data(self.data, need_HR=False)
                d_KLdiv_2_d_temperature = SoftHistogramLoss(bins=256,min=0,max=1,desired_hist_image=self.data['HR'].detach(),desired_hist_image_mask=data['Desired_Im_Mask'],
                    input_im_HR_mask=self.image_mask,gray_scale=True,patch_size=3 if 'patch' in objective else 1,automatic_temperature=True,image_Z=pre_tanh_Z)
                temperature_optimizer = torch.optim.Adam(d_KLdiv_2_d_temperature.optimizable_temperature.parameters(), lr=0.5)
                temperature_optimizer.zero_grad()
                initial_image = model.netG(model.var_L).to(self.device)
                temperatures,gradient_sizes,KL_divs = [],[],[]
                NUM_ITERS = 50
                for tempertaure_seeking_iter in range(NUM_ITERS):
                    cur_KL_div,temperature_gradients_size = d_KLdiv_2_d_temperature(initial_image)
                    temperature_gradients_size.backward(retain_graph=(tempertaure_seeking_iter<(NUM_ITERS-1)))
                    temperature_optimizer.step()
                    KL_divs.append(cur_KL_div.item())
                    temperatures.append(d_KLdiv_2_d_temperature.temperature.item())
                    gradient_sizes.append(temperature_gradients_size.item())
                optimal_temperature = temperatures[np.argmin(gradient_sizes)]
            else:
                optimal_temperature = 5e-4 if 'hist' in objective else 1e-3
            self.loss = SoftHistogramLoss(bins=256,min=0,max=1,desired_hist_image=self.data['HR'] if self.data is not None else None,
                desired_hist_image_mask=data['Desired_Im_Mask'] if self.data is not None else None,input_im_HR_mask=self.image_mask,
                gray_scale=True,patch_size=7 if 'patch' in objective else 1,temperature=optimal_temperature,dictionary_not_histogram='dict' in objective)
        elif 'Adversarial' in objective:
            self.netD = model.netD
            self.loss = GANLoss('wgan-gp', 1.0, 0.0).to(self.device)
        elif 'limited' in objective:
            self.initial_image = 1*model.fake_H.detach()
            self.rmse_weight = data['rmse_weight']
        if existing_optimizer is None:
            self.optimizer = torch.optim.Adam(self.Z_model.parameters(), lr=initial_LR)
        else:
            self.optimizer = existing_optimizer
        self.LR = initial_LR
        self.scheduler = None#torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer,verbose=True,threshold=1e-2,min_lr=self.MIN_LR,cooldown=10)
        self.loggers = loggers
        self.cur_iter = 0
        self.max_iters = max_iters
        self.model_training = HR_unpadder is not None
        self.random_Z_inits = random_Z_inits or self.model_training
        self.HR_unpadder = HR_unpadder

    def feed_data(self,data):
        self.data = data
        self.cur_iter = 0
        if 'l1' in self.objective:
            self.GT_HR = data['HR'].to(self.device)
        elif 'hist' in self.objective:
            self.loss.Feed_Desired_Hist_Im(data['HR'].to(self.device))

    def Manage_Model_Grad_Requirements(self,disable):
        if disable:
            self.original_requires_grad_status = []
            for p in self.model.netG.parameters():
                self.original_requires_grad_status.append(p.requires_grad)
                p.requires_grad = False
        else:
            for i, p in enumerate(self.model.netG.parameters()):
                p.requires_grad = self.original_requires_grad_status[i]

    def optimize(self):
        if 'Adversarial' in self.objective:
            self.model.netG.train(True) # Preventing image padding in the DTE code, to have the output fitD's input size
        self.Manage_Model_Grad_Requirements(disable=True)
        self.loss_values = []
        if self.random_Z_inits and self.cur_iter==0:
            self.Z_model.Randomize_Z()
        z_iter = self.cur_iter
        while True:
        # for z_iter in range(self.cur_iter,self.cur_iter+self.max_iters):
            if self.max_iters>0:
                if z_iter==(self.cur_iter+self.max_iters):
                    break
            elif len(self.loss_values)>=-self.max_iters:
                if z_iter==(self.cur_iter-2*self.max_iters):
                    break
                if (self.loss_values[self.max_iters] - self.loss_values[-1]) / np.abs(self.loss_values[self.max_iters]) < 1e-2 * self.LR:
                    break
            self.optimizer.zero_grad()
            self.data['Z'] = self.Z_model()
            # self.data['Z'] = self.Z_mask*self.Z_model()+(1-self.Z_mask)*self.model.cur_Z
            self.model.feed_data(self.data, need_HR=False)
            self.model.fake_H = self.model.netG(self.model.var_L)
            if self.model_training:
                self.model.fake_H = self.HR_unpadder(self.model.fake_H)
            if 'random' in self.objective:
                if 'l1' in self.objective:
                    data_in_loss_domain = self.model.fake_H
                elif 'VGG' in self.objective:
                    data_in_loss_domain = self.model.netF(self.model.fake_H)
                Z_loss = torch.min((data_in_loss_domain.unsqueeze(0) - data_in_loss_domain.unsqueeze(1)).abs() + torch.eye(
                        data_in_loss_domain.size(0)).unsqueeze(2).unsqueeze(3).unsqueeze(4).to(data_in_loss_domain.device), dim=0)[0]
                if 'limited' in self.objective:
                    rmse = (data_in_loss_domain - self.initial_image).abs()
                    if z_iter==0:
                        rmse_weight = 1*self.rmse_weight#*Z_loss.mean().item()/rmse.mean().item()
                    Z_loss = Z_loss-rmse_weight*rmse
                if self.Z_mask is not None:
                    Z_loss = Z_loss*self.Z_mask
                Z_loss = -1*Z_loss.mean(dim=(1,2,3))
            elif 'l1' in self.objective:
                Z_loss = self.loss(self.model.fake_H.to(self.device), self.GT_HR.to(self.device))
            elif 'desired_SVD' in self.objective:
                Z_loss = self.loss({'SR':self.model.fake_H.to(self.device)}).mean()
            elif any([phrase in self.objective for phrase in ['hist','dict']]):
                Z_loss = self.loss(self.model.fake_H.to(self.device))
            elif 'Adversarial' in self.objective:
                Z_loss = self.loss(self.netD(self.model.DTE_net.HR_unpadder(self.model.fake_H).to(self.device)),True)
            elif 'STD' in self.objective:
                Z_loss = torch.std(self.model.fake_H*self.image_mask,dim=(1,2,3)).to(self.device)
            elif 'VGG' in self.objective:
                Z_loss = self.loss(self.model.netF(self.model.fake_H).to(self.device),self.GT_HR_VGG)
            if 'max' in self.objective:
                Z_loss = -1*Z_loss
            # Z_loss.backward(retain_graph=(self.HR_unpadder is not None))
            cur_LR = self.optimizer.param_groups[0]['lr']
            if self.loggers is not None:
                for logger_num,logger in enumerate(self.loggers):
                    cur_value = Z_loss[logger_num].item() if Z_loss.dim()>0 else Z_loss.item()
                    logger.print_format_results('val', {'epoch': 0, 'iters': z_iter, 'time': time.time(), 'model': '','lr': cur_LR, 'Z_loss': cur_value}, dont_print=True)
            Z_loss = Z_loss.mean()
            Z_loss.backward()
            self.loss_values.append(Z_loss.item())
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step(Z_loss)
                if cur_LR<=1.2*self.MIN_LR:
                    break
            z_iter += 1
        if 'Adversarial' in self.objective:
            self.model.netG.train(False) # Preventing image padding in the DTE code, to have the output fitD's input size
        if 'random' in self.objective and 'limited' in self.objective:
            self.loss_values = self.loss_values[1:] #Removing the first loss values which is close to 0 in this case, to prevent discarfing optimization because loss increased compared to it.
        self.cur_iter = z_iter+1
        Z_2_return = self.Z_model.Return_Detached_Z()
        self.Manage_Model_Grad_Requirements(disable=False)
        # if self.Z_mask is not None and self.ONLY_MODIFY_MASKED_AREA:
        #     Z_2_return = self.Z_mask * self.Z_model() + (1 - self.Z_mask) * self.initial_Z
        #     self.data['Z'] = Z_2_return
        #     self.model.feed_data(self.data, need_HR=False)
        if self.model_training:# Results of all optimization iterations were cropped, so I do another one without cropping and with Gradients computation (for model training)
            self.data['Z'] = Z_2_return
            self.model.feed_data(self.data, need_HR=False)
            self.model.fake_H = self.model.netG(self.model.var_L)
        return Z_2_return

    def ReturnStatus(self):
        return self.Z_model.PreTanhZ(),self.optimizer


####################
# metric
####################


def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')
