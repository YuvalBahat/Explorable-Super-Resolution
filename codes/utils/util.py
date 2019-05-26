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

class Optimizable_Z(torch.nn.Module):
    def __init__(self,Z_shape,Z_range=None):
        super(Optimizable_Z, self).__init__()
        # self.device = torch.device('cuda')
        self.Z = torch.nn.Parameter(data=torch.zeros(Z_shape).type(torch.cuda.FloatTensor))
        self.Z_range = Z_range
        if Z_range is not None:
            self.tanh = torch.nn.Tanh()
        # self.loss = torch.nn.L1Loss().to(torch.device('cuda'))


    def forward(self):
        if self.Z_range is not None:
            return self.Z_range*self.tanh(self.Z)
        else:
            return self.Z
    # def Loss(self,im1,im2):
    #     return self.loss(im1.to(self.device),im2.to(self.device))

class Z_optimizer():
    MIN_LR = 1e-5
    def __init__(self,objective,image_shape,model,Z_range,initial_LR,logger,max_iters,data):
        self.Z_model = Optimizable_Z(Z_shape=[1, 1] + list(image_shape), Z_range=Z_range)
        self.optimizer = torch.optim.Adam(self.Z_model.parameters(), lr=initial_LR)
        self.objective = objective
        self.data = data
        self.device = torch.device('cuda')
        if 'L1' in objective:
            self.loss = torch.nn.L1Loss().to(torch.device('cuda'))
            # scheduler_threshold = 1e-2
        elif 'STD' in objective:
            assert self.objective in ['max_STD', 'min_STD']
            # scheduler_threshold = 0.9999
        elif 'VGG' in objective:
            self.GT_HR_VGG = model.netF(self.data['HR']).detach().to(self.device)
            self.loss = torch.nn.L1Loss().to(torch.device('cuda'))
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer,verbose=True,threshold=1e-2,min_lr=self.MIN_LR,cooldown=10)
        self.model = model
        self.logger = logger
        self.max_iters = max_iters
    def optimize(self):
        for z_iter in range(self.max_iters):
            self.optimizer.zero_grad()
            self.data['Z'] = self.Z_model()
            self.model.feed_data(self.data, need_HR=False)
            self.model.test(prevent_grads_calc=False)
            if 'L1' in self.objective:
                Z_loss = self.loss(self.model.fake_H.to(self.device), self.data['HR'].to(self.device))
            elif 'STD' in self.objective:
                Z_loss = self.model.fake_H.std().to(self.device)
            elif 'VGG' in self.objective:
                Z_loss = self.loss(self.model.netF(self.model.fake_H).to(self.device),self.GT_HR_VGG)
            if 'max' in self.objective:
                Z_loss = -1*Z_loss
            Z_loss.backward()
            self.optimizer.step()
            self.scheduler.step(Z_loss)
            cur_LR = self.optimizer.param_groups[0]['lr']
            if cur_LR<=1.2*self.MIN_LR:
                break
            self.logger.print_format_results('val', {'epoch': 0, 'iters': z_iter, 'time': time.time(), 'model': '','lr': cur_LR, 'Z_loss': Z_loss.item()}, dont_print=True)
        return self.Z_model()


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
