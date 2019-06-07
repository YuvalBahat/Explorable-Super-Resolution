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
import torch
from models.modules.loss import GANLoss

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

class SoftHistogramLoss(torch.nn.Module):
    def __init__(self,bins,min,max,desired_hist_image,desired_hist_image_mask,gray_scale=True,input_im_HR_mask=None):
        super(SoftHistogramLoss,self).__init__()
        assert gray_scale,'Multi-dimensional histogram is not yet supported'
        bin_width = (max-min)/bins
        self.max = max
        self.bin_centers = torch.linspace(min+bin_width/2,max-bin_width/2,bins)
        self.gray_scale = gray_scale
        self.temperature = 3e-13
        self.exp_power = 0.15
        self.SQRT_EPSILON = 1e-7
        self.image_mask = None
        if gray_scale:
            self.bins = 1.*self.bin_centers.view([1]+list(self.bin_centers.size())).to(desired_hist_image.device)
            self.image_mask = desired_hist_image_mask.view([-1] + [1] * (self.bins.dim() - 1)).type(torch.ByteTensor)
            self.desired_hist = self.ComputeSoftHistogram(desired_hist_image.mean(1,keepdim=True),return_log_hist=False).detach()
            self.image_mask = input_im_HR_mask.view([-1] + [1] * (self.bins.dim() - 1)).type(torch.ByteTensor) if input_im_HR_mask is not None else None
        self.KLdiv_loss = torch.nn.KLDivLoss()

    def ComputeSoftHistogram(self,image,return_log_hist,wrap_hist=True):
        if self.gray_scale:
            image = image.view([-1]+[1]*(self.bins.dim()-1))
            if self.image_mask is not None:
                image = image[self.image_mask].view([-1]+[1]*(self.bins.dim()-1))
            if wrap_hist:
                hist = torch.min(torch.min((image-self.bins).abs(),(image-self.bins-self.max).abs()),(image-self.bins+self.max).abs())
            else:
                hist = (image-self.bins).abs()
            hist = -((hist/self.temperature+self.SQRT_EPSILON)**self.exp_power).mean(0,keepdim=True)
            if return_log_hist:
                return hist
            hist = torch.exp(hist)
            return hist/hist.sum()

    def forward(self,cur_image):
        if self.gray_scale:
            cur_image = cur_image.mean(1,keepdim=True)
        cur_image_hist = self.ComputeSoftHistogram(cur_image,return_log_hist=True)
        return self.KLdiv_loss(cur_image_hist,self.desired_hist)

class Optimizable_Z(torch.nn.Module):
    def __init__(self,Z_shape,Z_range=None,initial_Z=None):
        super(Optimizable_Z, self).__init__()
        # self.device = torch.device('cuda')
        self.Z = torch.nn.Parameter(data=torch.zeros(Z_shape).type(torch.cuda.FloatTensor))
        if initial_Z is not None:
            self.Z.data = initial_Z
        self.Z_range = Z_range
        if Z_range is not None:
            self.tanh = torch.nn.Tanh()
        # self.loss = torch.nn.L1Loss().to(torch.device('cuda'))


    def forward(self):
        if self.Z_range is not None:
            return self.Z_range*self.tanh(self.Z)
        else:
            return self.Z

    def PreTanhZ(self):
        return self.Z.data
    # def Loss(self,im1,im2):
    #     return self.loss(im1.to(self.device),im2.to(self.device))

class Z_optimizer():
    MIN_LR = 1e-5
    def __init__(self,objective,LR_size,model,Z_range,logger,max_iters,data,image_mask=None,Z_mask=None,initial_pre_tanh_Z=None,initial_LR=None,existing_optimizer=None):
        self.Z_model = Optimizable_Z(Z_shape=[1,model.num_latent_channels] + list(LR_size), Z_range=Z_range,initial_Z=initial_pre_tanh_Z)
        assert (initial_LR is not None) or (existing_optimizer is not None),'Should either supply optimizer from previous iterations or initial LR for new optimizer'
        if existing_optimizer is None:
            self.optimizer = torch.optim.Adam(self.Z_model.parameters(), lr=initial_LR)
        else:
            self.optimizer = existing_optimizer
        self.objective = objective
        self.data = data
        self.device = torch.device('cuda')
        if image_mask is None:
            self.image_mask = torch.ones(list(model.fake_H.size()[2:])).type(model.fake_H.dtype).to(self.device)
            self.Z_mask = None#torch.ones(LR_size).type(model.fake_H.dtype).to(self.device)
        else:
            assert Z_mask is not None,'Should either supply both masks or niether'
            self.image_mask = torch.from_numpy(image_mask).type(model.fake_H.dtype).to(self.device)
            self.Z_mask = torch.from_numpy(Z_mask).type(model.fake_H.dtype).to(self.device)
            self.initial_Z = 1.*model.cur_Z
        self.Z_mask.requires_grad = False
        self.image_mask.requires_grad = False
        if 'L1' in objective:
            self.loss = torch.nn.L1Loss().to(torch.device('cuda'))
            # scheduler_threshold = 1e-2
        elif 'STD' in objective:
            assert self.objective in ['max_STD', 'min_STD']
            # scheduler_threshold = 0.9999
        elif 'VGG' in objective:
            self.GT_HR_VGG = model.netF(self.GT_HR).detach().to(self.device)
            self.loss = torch.nn.L1Loss().to(torch.device('cuda'))
        elif 'Hist' in objective:
            # def HistogramLoss(model_output,desired_Im):
            #     num_pixels = model_output.numel()/model_output.size(1)
            #     output_hist = torch.histc(model_output.mean(1),bins=255,min=0,max=1).type(model_output.dtype)/num_pixels
            #     desired_hist = torch.histc(desired_Im.mean(1), bins=255,min=0,max=1).type(model_output.dtype)/num_pixels
            #     return torch.nn.functional.kl_div(torch.log(output_hist+1e-9),desired_hist)
            self.loss = SoftHistogramLoss(bins=255//3,min=0,max=1,desired_hist_image=self.data['HR'],desired_hist_image_mask=data['Desired_Im_Mask'],
                input_im_HR_mask=self.image_mask)
        elif 'Adversarial' in objective:
            self.netD = model.netD
            self.loss = GANLoss('wgan-gp', 1.0, 0.0).to(self.device)

        self.scheduler = None#torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer,verbose=True,threshold=1e-2,min_lr=self.MIN_LR,cooldown=10)
        self.model = model
        self.logger = logger
        self.cur_iter = 0
        self.max_iters = max_iters
    def optimize(self):
        if 'Adversarial' in self.objective:
            self.model.netG.train(True) # Preventing image padding in the DTE code, to have the output fitD's input size
        for z_iter in range(self.cur_iter,self.cur_iter+self.max_iters):
            self.optimizer.zero_grad()
            self.data['Z'] = self.Z_model()
            # self.data['Z'] = self.Z_mask*self.Z_model()+(1-self.Z_mask)*self.model.cur_Z
            self.model.feed_data(self.data, need_HR=False)
            self.model.fake_H = self.model.netG(self.model.var_L)
            # self.model.test(prevent_grads_calc=False)
            if 'L1' in self.objective:
                Z_loss = self.loss(self.model.fake_H.to(self.device), self.GT_HR.to(self.device))
            elif 'Hist' in self.objective:
                Z_loss = self.loss(self.model.fake_H.to(self.device))
            elif 'Adversarial' in self.objective:
                Z_loss = self.loss(self.netD(self.model.DTE_net.HR_unpadder(self.model.fake_H).to(self.device)),True)
            elif 'STD' in self.objective:
                Z_loss = (self.model.fake_H*self.image_mask).std().to(self.device)
            elif 'VGG' in self.objective:
                Z_loss = self.loss(self.model.netF(self.model.fake_H).to(self.device),self.GT_HR_VGG)
            if 'max' in self.objective:
                Z_loss = -1*Z_loss
            Z_loss.backward()
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step(Z_loss)
            cur_LR = self.optimizer.param_groups[0]['lr']
            if cur_LR<=1.2*self.MIN_LR:
                break
            self.logger.print_format_results('val', {'epoch': 0, 'iters': z_iter, 'time': time.time(), 'model': '','lr': cur_LR, 'Z_loss': Z_loss.item()}, dont_print=True)
        if 'Adversarial' in self.objective:
            self.model.netG.train(False) # Preventing image padding in the DTE code, to have the output fitD's input size
        self.cur_iter = z_iter+1
        Z_2_return = self.Z_model()
        if self.Z_mask is not None:
            Z_2_return = self.Z_mask * self.Z_model() + (1 - self.Z_mask) * self.initial_Z
            self.data['Z'] = Z_2_return
            self.model.feed_data(self.data, need_HR=False)
            # self.model.test(prevent_grads_calc=True)
            with torch.no_grad():
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
