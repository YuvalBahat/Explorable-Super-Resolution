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
# from scipy.ndimage.morphology import binary_opening
# from sklearn.feature_extraction.image import extract_patches_2d
import torch
import torch.nn as nn
# from models.modules.loss import GANLoss,FilterLoss
# from skimage.color import rgb2hsv,hsv2rgb
####################
# miscellaneous
####################
from data.util import ycbcr2rgb

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

def Assign_GPU(max_GPUs=1,**kwargs):
    excluded_IDs = []
    def getAvailable():
        return GPUtil.getAvailable(order='memory',excludeID=excluded_IDs,limit=max_GPUs if max_GPUs is not None else 100,**kwargs)
    GPU_2_use = getAvailable()
    if len(GPU_2_use)==0:
        print('No available GPUs. waiting...')
        while len(GPU_2_use)==0:
            time.sleep(5)
            GPU_2_use = getAvailable()
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
    return lambda0,lambda1,theta

####################
# image convert
####################
def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1),chroma_mode=False):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    # if self.chroma_mode:
    #     out_dict['Decomp'] = ycbcr2rgb(out_dict['Decomp'])
    tensor = tensor.squeeze().float().cpu()
    # tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    # tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    if n_dim == 4:
        assert not chroma_mode,'Unsupported yet'
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 3:
        img_np = tensor.numpy()
        if chroma_mode: #input tensor is in YCbCr color space:
            img_np = ycbcr2rgb(np.transpose(img_np, (1, 2, 0)))[:,:,[2,1,0]]
        else:
            img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    img_np = (np.clip(img_np,min_max[0],min_max[1])-min_max[0])/ (min_max[1] - min_max[0])
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

def crop_nd_array(array,desired_mask_bounding_rect):
    return array[desired_mask_bounding_rect[1]:desired_mask_bounding_rect[1] + desired_mask_bounding_rect[3],
           desired_mask_bounding_rect[0]:desired_mask_bounding_rect[0] + desired_mask_bounding_rect[2], ...]

# def zero_pad_array(array,target_im_pad_sizes,mode='constant'):
#     return np.pad(array, (tuple(target_im_pad_sizes[0]), tuple(target_im_pad_sizes[1]), (0, 0)), mode=mode)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def SVD_2_LatentZ(SVD_values,max_lambda=1):
    # Given SVD values, returns corresponding structural tensor values.
    # SVD values: lambda0 in [0,max_lambda], lambda1 in [0,max_lambda], theta in [0,2*pi]
    # Returned values: Sigma I_x^2 in [-max_lambda,max_lambda], Sigma I_y^2 in [-max_lambda,max_lambda], Sigma I_x*I_y in (not sure, should calculate, but a symmetric range).
    return torch.stack([2*max_lambda*(SVD_values[:,1,...]*(torch.sin(SVD_values[:,-1,...])**2)+SVD_values[:,0,...]*(torch.cos(SVD_values[:,-1,...])**2))-max_lambda,
                                  2*max_lambda*(SVD_values[:,0,...]*(torch.sin(SVD_values[:,-1,...])**2)+SVD_values[:,1,...]*(torch.cos(SVD_values[:,-1,...])**2))-max_lambda,#Normalizing range to have negative values as well,trying to match [-1,1]
                                  2*(SVD_values[:,0,...]-SVD_values[:,1,...])*torch.sin(SVD_values[:,-1,...])*torch.cos(SVD_values[:,-1,...])],1)

def ResizeCategorialImage(image,dsize):
    LOWER_CATEGORY_OVERRULE = True #I preefer actual scribbles to rule over brightness manipulation and TV minimization regions.
    assert 'int' in str(image.dtype),'I suspect input image is not categorial, since pixel values are not integers'
    if np.all(image.shape[:2]==dsize):
        return image
    output_image = np.zeros(shape=dsize).astype(image.dtype)
    categories_set = sorted(set(list(image.reshape([-1]))))
    if LOWER_CATEGORY_OVERRULE:
        categories_set = categories_set[::-1]
    for category in categories_set:
        cur_category_image = cv2.resize((image==category).astype(image.dtype),dsize=dsize[::-1],interpolation=cv2.INTER_LINEAR)>0.5
        output_image = np.logical_not(cur_category_image)*output_image+cur_category_image*category
    return output_image

def ResizeScribbleImage(image,dsize):
    if np.all(image.shape[:2]==dsize):
        return image
    resized = cv2.resize(image, dsize=dsize[::-1], interpolation=cv2.INTER_AREA)
    if image.ndim>resized.ndim:
        resized = np.reshape(resized,list(resized.shape[:2])+[image.shape[2]])
    return resized

def SmearMask2JpegBlocks(mask):
    # Each block in the mask is assigned with the maximal value in it. This is meant to convert each block participating in the mask to participate fully, which makes more sense in the JPEG case.
    # Note the special case of non-binary masks (when using brightness manipulation or local TV minimization) and having different non-zero values at the same block.
    # The maximal value would prevail - which is a somewhat arbitrary rule.
    mask_shape = np.array(mask.shape)
    assert np.all(mask_shape/8==np.round(mask_shape/8)),'Only supporting sizes containing integer number of 8x8 blocks'
    mask = mask.reshape([mask_shape[0]//8,8,mask_shape[1]//8,8])
    mask = np.max(np.max(mask,axis=1,keepdims=True),axis=3,keepdims=True)*np.ones([1,8,1,8]).astype(mask.dtype)
    return mask.reshape(mask_shape)

def Tensor_YCbCR2RGB(image):
    ycbcr2rgb_mat = torch.from_numpy(255*np.array([[0.00456621, 0.00456621, 0.00456621], [0, -0.00153632, 0.00791071],[0.00625893, -0.00318811, 0]]).transpose()).view(1,3,3,1,1)
    return (ycbcr2rgb_mat.type(image.type()) * image.unsqueeze(1)).sum(2) + torch.tensor([-222.921, 135.576, -276.836]).type(image.type()).view(1, 3, 1, 1) / 255

def Z_64channels2image(Z):
    return np.reshape(Z,list(Z.shape[:2])+[8,8]).transpose((0,2,1,3)).reshape(list(8*np.array(Z.shape[:2]))+[1])

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

def convert_batchNorm_2_layerNorm(model,input):
    module_layers = []
    for i,l in enumerate(model.children()):
        if isinstance(l,nn.Sequential):
            inner_module,input = convert_batchNorm_2_layerNorm(l,input=input)
            module_layers.append(inner_module)
        elif isinstance(l,nn.BatchNorm2d):
            module_layers.append(nn.LayerNorm(normalized_shape=list(input.size())[1:]))
        else:
            module_layers.append(l)
            input = l(input)
    return nn.Sequential(*module_layers),input


# import numpy as np
# import torch
# import torch.nn as nn
from torch.autograd import Variable
# import torch.nn.init as init
# import torch.nn.functional as F

def compute_RF_numerical(net,img_np):
    '''
    @param net: Pytorch network
    @param img_np: numpy array to use as input to the networks, it must be full of ones and with the correct
    shape.
    '''
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.fill_(1)
            if m.bias is not None:
                m.bias.data.fill_(0)
    net.apply(weights_init)
    # img_ = Variable(torch.from_numpy(img_np).float(),requires_grad=True).to(next(net.module.parameters()).device)
    img_ = Variable(torch.from_numpy(img_np).float(),requires_grad=True)
    out_cnn=net(img_)
    out_shape=out_cnn.size()
    ndims=len(out_cnn.size())
    # grad=torch.zeros(out_cnn.size()).to(next(net.module.parameters()).device)
    grad=torch.zeros(out_cnn.size())
    l_tmp=[]
    for i in range(ndims):
        if i==0 or i ==1:#batch or channel
            l_tmp.append(0)
        else:
            l_tmp.append(out_shape[i]//2)
    print(tuple(l_tmp))
    grad[tuple(l_tmp)]=1
    # out_cnn.backward(gradient=img_)
    out_cnn.backward(gradient=grad)
    grad_np=img_.grad[0,0].data.numpy()
    # grad_np=grad[0,0].data.cpu().numpy()
    idx_nonzeros=np.where(grad_np!=0)
    RF=[np.max(idx)-np.min(idx)+1 for idx in idx_nonzeros]

    return RF
