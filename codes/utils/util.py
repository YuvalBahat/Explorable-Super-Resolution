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
    def __init__(self,bins,min,max,desired_hist_image,desired_hist_image_mask,gray_scale=True,input_im_HR_mask=None,patch_size=1,automatic_temperature=None):
        self.temperature = 0.05#0.05**2#0.006**6
        self.exp_power = 1#2#6
        self.SQRT_EPSILON = 1e-7

        super(SoftHistogramLoss,self).__init__()
        # min correspond to the CENTER of the first bin, and max to the CENTER of the last bin
        self.bin_width = (max-min)/(bins-1)
        self.max = max
        self.temperature *= self.bin_width*85 # 0.006 was calculated for pixels range [0,1] with 85 bins. So I adjust it acording to actual range and n_bins, manifested in self.bin_width
        # self.bin_centers = torch.linspace(min+self.bin_width/2,max-self.bin_width/2,bins)
        self.bin_centers = torch.linspace(min,max,bins)
        self.gray_scale = gray_scale
        self.patch_size = patch_size
        self.num_dims = desired_hist_image.size(1)
        self.KDE = not gray_scale or patch_size>1 # Using Kernel Density Estimation rather than histogram
        if gray_scale:
            self.num_dims = self.num_dims//3
            self.bins = 1. * self.bin_centers.view([1] + list(self.bin_centers.size())).type(torch.cuda.DoubleTensor)
            desired_hist_image = desired_hist_image.mean(1, keepdim=True).view([-1,1])
        if patch_size>1:
            assert gray_scale,'Not supporting color images for now'
            self.num_dims = patch_size**2
            self.patch_extraction_mat = self.ReturnPatchExtractionMat(desired_hist_image_mask).to(desired_hist_image.device)
            # patches_indexes = patches_indexes[np.all(patches_indexes>0,1),:]-1
            # corresponding_mat_rows = np.arange(patches_indexes.size).reshape([-1])
            # patch_extraction_mat = torch.sparse.FloatTensor(torch.LongTensor([corresponding_mat_rows,patches_indexes.reshape([-1])]),torch.FloatTensor(np.ones([corresponding_mat_rows.size])),
            #     torch.Size([patches_indexes.size,desired_hist_image_mask.size])).to(desired_hist_image.device)
            desired_hist_image = torch.sparse.mm(self.patch_extraction_mat,desired_hist_image).view([self.num_dims,-1,1])
            self.image_mask = None
        else:
            self.image_mask = torch.from_numpy(desired_hist_image_mask).to(desired_hist_image.device).view([-1]).type(torch.ByteTensor) if desired_hist_image_mask is not None else None
            desired_hist_image = 1 * desired_hist_image.view([self.num_dims, -1, 1])
        if self.KDE:
            # The bins are now simply the multi-dimensional pixels/patches. So now I remove redundant bins, by checking if there is duplicacy:
            if self.image_mask is not None:
                desired_hist_image = desired_hist_image[:,self.image_mask,:]
            # self.image_mask = None # I already used the mask to remove irrelevant pixels here, so should not use it when computing the desired im hist.
            self.bins = desired_hist_image
            repeated_elements_mat = (desired_hist_image.view([self.num_dims,-1,1])-desired_hist_image.view([desired_hist_image.size(0)]+[1,-1])).abs()
            repeated_elements_mat = (repeated_elements_mat < self.bin_width/ 2).all(0)
            repeated_elements_mat = torch.mul(repeated_elements_mat, (1 - torch.diag(torch.ones([repeated_elements_mat.size(0)]))).type(repeated_elements_mat.dtype).to(repeated_elements_mat.device))
            repeated_elements_mat = torch.triu(repeated_elements_mat).any(1)^1
            self.bins = self.bins[:,repeated_elements_mat]
            del repeated_elements_mat
        self.bins = self.bins.view([self.num_dims,1,-1]).type(torch.cuda.DoubleTensor)
        self.KLdiv_loss = torch.nn.KLDivLoss()
        if patch_size>1:
            self.patch_extraction_mat = self.ReturnPatchExtractionMat(input_im_HR_mask.data.cpu().numpy()).to(desired_hist_image.device)
        else:
            self.image_mask = input_im_HR_mask.view([-1]).type(torch.ByteTensor) if input_im_HR_mask is not None else None
        if automatic_temperature is not None:
            if patch_size > 1:
                initial_image = torch.sparse.mm(self.patch_extraction_mat,automatic_temperature.mean(1, keepdim=True).view([-1, 1])).view([self.num_dims, -1, 1])
            else:
                initial_image = automatic_temperature.mean(1, keepdim=True).contiguous().view([self.num_dims,-1,1])
                if self.image_mask is not None:
                    initial_image = initial_image[:,self.image_mask,:]
            self.TemperatureSearch(desired_hist_image,initial_image,1e-3)
        with torch.no_grad():
            self.desired_hist = self.ComputeSoftHistogram(desired_hist_image,return_log_hist=False,reshape_image=False,compute_hist_normalizer=True).detach()
    def TemperatureSearch(self,desired_image,initial_image,desired_KL_div):
        INITIAL_TEMPERATURE = 1
        STEP_FACTOR = 1000
        KL_DIV_TOLERANCE = 0.1
        self.temperature = INITIAL_TEMPERATURE
        cur_KL_div = []
        with torch.no_grad():
            while True:
                desired_im_hist = self.ComputeSoftHistogram(desired_image,return_log_hist=False,reshape_image=False,compute_hist_normalizer=True)
                initial_image_hist = self.ComputeSoftHistogram(initial_image,return_log_hist=True,reshape_image=False,compute_hist_normalizer=False)
                cur_KL_div.append(self.KLdiv_loss(initial_image_hist,desired_im_hist).item())
                if np.abs(np.log(max([0,cur_KL_div[-1]])/desired_KL_div))<=np.log(1+KL_DIV_TOLERANCE):
                    break
                elif len(cur_KL_div)==1:
                    if cur_KL_div[-1] > desired_KL_div:
                        temperature_range = [1*self.temperature,STEP_FACTOR*self.temperature]
                    else:
                        temperature_range = [self.temperature/STEP_FACTOR,1*self.temperature]
                else:
                    if cur_KL_div[-1]>desired_KL_div:
                        temperature_range = [self.temperature,1*temperature_range[1]]
                    else:
                        temperature_range = [1 * temperature_range[0],self.temperature]
                self.temperature = np.mean(temperature_range)

    def ReturnPatchExtractionMat(self,mask):
        mask = binary_opening(mask,np.ones([self.patch_size, self.patch_size]).astype(np.bool))
        patches_indexes = extract_patches_2d(np.multiply(mask,1 + np.arange(mask.size).reshape(mask.shape)),(self.patch_size, self.patch_size)).reshape([-1, self.num_dims])
        patches_indexes = patches_indexes[np.all(patches_indexes > 0, 1), :] - 1
        corresponding_mat_rows = np.arange(patches_indexes.size).reshape([-1])
        patch_extraction_mat = torch.sparse.FloatTensor(torch.LongTensor([corresponding_mat_rows, patches_indexes.transpose().reshape([-1])]),
            torch.FloatTensor(np.ones([corresponding_mat_rows.size])),torch.Size([patches_indexes.size, mask.size]))
        return patch_extraction_mat

    def ComputeSoftHistogram(self,image,return_log_hist,reshape_image,compute_hist_normalizer):
        if not reshape_image:
            image = image.type(torch.cuda.DoubleTensor)
        else:
            if self.patch_size > 1:
                image = torch.sparse.mm(self.patch_extraction_mat, image.view([-1, 1])).view([self.num_dims, -1])
            else:
                image = image.contiguous().view([self.num_dims,-1])
                if self.image_mask is not None:
                    image = image[:, self.image_mask]
            image = image.unsqueeze(-1).type(torch.cuda.DoubleTensor)
        hist = (image-self.bins).abs()
        hist = torch.min(hist,(image-self.bins-self.max).abs())
        hist = torch.min(hist,(image-self.bins+self.max).abs())
        hist = -((hist+self.SQRT_EPSILON)**self.exp_power)/self.temperature
        hist = hist.sum(0)
        hist = torch.exp(hist).mean(0)
        if compute_hist_normalizer or not self.KDE:
            self.normalizer = hist.sum()/image.size(1)
        hist = (hist/self.normalizer/image.size(1)).type(torch.cuda.FloatTensor)
        if self.KDE: # Adding another "bin" to account for all other missing bins
            hist = torch.cat([hist,(1-torch.min(torch.tensor(1).type(hist.dtype).to(hist.device),hist.sum())).view([1])])
        if return_log_hist:
            return torch.log(hist+torch.finfo(hist.dtype).eps)
        else:
            return hist

    def forward(self,cur_image):
        if self.gray_scale:
            cur_image = cur_image.mean(1,keepdim=True)
        cur_image_hist = self.ComputeSoftHistogram(cur_image,return_log_hist=True,reshape_image=True,compute_hist_normalizer=False)
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

    def Return_Detached_Z(self):
        return self.forward().detach()

class Z_optimizer():
    MIN_LR = 1e-5
    ONLY_MODIFY_MASKED_AREA = False
    def __init__(self,objective,LR_size,model,Z_range,max_iters,data=None,logger=None,image_mask=None,Z_mask=None,initial_pre_tanh_Z=None,initial_LR=None,existing_optimizer=None,
                 batch_size=1,HR_unpadder=None):
        if initial_pre_tanh_Z is None and 'cur_Z' in model.__dict__.keys():
            initial_pre_tanh_Z = 1.*model.cur_Z/2/(Z_range+1e-7)
            initial_pre_tanh_Z = 0.5*torch.log(1+initial_pre_tanh_Z)/(1-initial_pre_tanh_Z)
        self.Z_model = Optimizable_Z(Z_shape=[batch_size,model.num_latent_channels] + list(LR_size), Z_range=Z_range,initial_Z=initial_pre_tanh_Z)
        assert (initial_LR is not None) or (existing_optimizer is not None),'Should either supply optimizer from previous iterations or initial LR for new optimizer'
        if existing_optimizer is None:
            self.optimizer = torch.optim.Adam(self.Z_model.parameters(), lr=initial_LR)
        else:
            self.optimizer = existing_optimizer
        self.objective = objective
        self.data = data
        self.device = torch.device('cuda')
        if image_mask is None:
            if 'fake_H' in model.__dict__.keys():
                self.image_mask = torch.ones(list(model.fake_H.size()[2:])).type(model.fake_H.dtype).to(self.device)
            else:
                self.image_mask = None
            self.Z_mask = None#torch.ones(LR_size).type(model.fake_H.dtype).to(self.device)
        else:
            assert Z_mask is not None,'Should either supply both masks or niether'
            self.image_mask = torch.from_numpy(image_mask).type(model.fake_H.dtype).to(self.device)
            self.Z_mask = torch.from_numpy(Z_mask).type(model.fake_H.dtype).to(self.device)
            self.initial_Z = 1.*model.cur_Z
            self.image_mask.requires_grad = False
            self.Z_mask.requires_grad = False
        if 'l1' in objective:
            self.loss = torch.nn.L1Loss().to(torch.device('cuda'))
            # scheduler_threshold = 1e-2
        elif 'STD' in objective:
            assert self.objective in ['max_STD', 'min_STD']
            # scheduler_threshold = 0.9999
        elif 'VGG' in objective:
            self.GT_HR_VGG = model.netF(self.GT_HR).detach().to(self.device)
            self.loss = torch.nn.L1Loss().to(torch.device('cuda'))
        elif 'Hist' in objective:
            gray_scale_hist = True
            self.automatic_temperature = gray_scale_hist and 'patch' not in objective
            if self.automatic_temperature:
                self.data['Z'] = self.Z_model()
                model.feed_data(self.data, need_HR=False)
                with torch.no_grad():
                    model.fake_H = model.netG(model.var_L)
            self.loss = SoftHistogramLoss(bins=128,min=0,max=1,desired_hist_image=self.data['HR'],desired_hist_image_mask=data['Desired_Im_Mask'],
                input_im_HR_mask=self.image_mask,gray_scale=True,patch_size=5 if 'patch' in objective else 1,
                  automatic_temperature=model.fake_H.to(self.device) if self.automatic_temperature else None)
        elif 'Adversarial' in objective:
            self.netD = model.netD
            self.loss = GANLoss('wgan-gp', 1.0, 0.0).to(self.device)

        self.scheduler = None#torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer,verbose=True,threshold=1e-2,min_lr=self.MIN_LR,cooldown=10)
        self.model = model
        self.logger = logger
        self.cur_iter = 0
        self.max_iters = max_iters
        self.HR_unpadder = HR_unpadder

    def feed_data(self,data):
        self.data = data
        self.cur_iter = 0
        self.GT_HR = data['HR'].to(self.device)

    def optimize(self):
        if 'Adversarial' in self.objective:
            self.model.netG.train(True) # Preventing image padding in the DTE code, to have the output fitD's input size
        original_requires_grad_status = []
        for p in self.model.netG.parameters():
            original_requires_grad_status.append(p.requires_grad)
            p.requires_grad = False
        for z_iter in range(self.cur_iter,self.cur_iter+self.max_iters):
            self.optimizer.zero_grad()
            self.data['Z'] = self.Z_model()
            # self.data['Z'] = self.Z_mask*self.Z_model()+(1-self.Z_mask)*self.model.cur_Z
            self.model.feed_data(self.data, need_HR=False)
            self.model.fake_H = self.model.netG(self.model.var_L)
            if self.HR_unpadder is not None:
                self.model.fake_H = self.HR_unpadder(self.model.fake_H)
            # self.model.test(prevent_grads_calc=False)
            if 'l1' in self.objective:
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
            # Z_loss.backward(retain_graph=(self.HR_unpadder is not None))
            Z_loss.backward()
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step(Z_loss)
            cur_LR = self.optimizer.param_groups[0]['lr']
            if cur_LR<=1.2*self.MIN_LR:
                break
            if self.logger is not None:
                self.logger.print_format_results('val', {'epoch': 0, 'iters': z_iter, 'time': time.time(), 'model': '','lr': cur_LR, 'Z_loss': Z_loss.item()}, dont_print=True)
        if 'Adversarial' in self.objective:
            self.model.netG.train(False) # Preventing image padding in the DTE code, to have the output fitD's input size
        self.cur_iter = z_iter+1
        Z_2_return = self.Z_model.Return_Detached_Z()
        for i,p in enumerate(self.model.netG.parameters()):
            p.requires_grad = original_requires_grad_status[i]
        if self.Z_mask is not None and self.ONLY_MODIFY_MASKED_AREA:
            Z_2_return = self.Z_mask * self.Z_model() + (1 - self.Z_mask) * self.initial_Z
            self.data['Z'] = Z_2_return
            self.model.feed_data(self.data, need_HR=False)
            # self.model.test(prevent_grads_calc=True)
            with torch.no_grad():
                self.model.fake_H = self.model.netG(self.model.var_L)
        elif self.HR_unpadder is not None:# Results of all optimization iterations were cropped, so I do another one without cropping and with Gradients computation (for model training)
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
