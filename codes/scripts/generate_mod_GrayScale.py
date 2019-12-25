import os
import sys
import numpy as np
from tqdm import tqdm
import cv2
from socket import gethostname
from DTE.imresize_DTE import imresize
from scripts.create_gaussian_upscale_kernel import create_Gaussian_Upscale_kernel

dataset_root_path = '/media/ybahat/data/Datasets' if gethostname() == 'Yuval-Technion' else '/home/tiras/datasets' if 'tiras' in os.getcwd() else '/home/ybahat/data/Databases'
# input_folder = os.path.join(dataset_root_path,'DIV2K/DIV2K_train_HR_sub')
# save_LR_folder = os.path.join(dataset_root_path,'DIV2K/DIV2K_train_HR_sub_bicLRx4')
# scale_factor = 2
dataset_name = 'LIVE_release2'#'DIV2K_train'#'Set14','BSD100'
data_category_string = '/refimgs'#'_GaussianKernel'#'_sub'#'_train'
# upscale_kernel = None#create_Gaussian_Upscale_kernel(size=16,sf=scale_factor,std=0.6)

input_folder = os.path.join(dataset_root_path,'%s/%s%s_HR'%(dataset_name,dataset_name,data_category_string))
if dataset_name=='Set14':
    input_folder = input_folder.replace('HR','HRx4')
save_Uncomp_folder = os.path.join(dataset_root_path,'%s/%s%s_Uncomp'%(dataset_name,dataset_name,data_category_string))
if dataset_name=='LIVE_release2':
    input_folder = input_folder.replace('_HR','').replace(os.path.join(dataset_name,dataset_name),dataset_name)
    save_Uncomp_folder = save_Uncomp_folder.replace(os.path.join(dataset_name,dataset_name),dataset_name)
# save_LR_folder = os.path.join(dataset_root_path,'%s/%s%s_bicLRx%d'%(dataset_name,dataset_name,data_category_string,scale_factor))

# up_scale = scale_factor
mod_scale = 8 # For JPEG

def mod_img(image,modulu):
    if image.ndim<3:
        image = image.reshape(list(image.shape)+[1])
    im_shape = image.shape[:2]
    im_shape -= np.mod(im_shape,modulu)
    return image[:im_shape[0],:im_shape[1],:]

# assert save_LR_folder is None or not os.path.exists(save_LR_folder),'Folder [{:s}] already exists. Exit...'.format(save_LR_folder)
assert save_Uncomp_folder is None or not os.path.exists(save_Uncomp_folder),'Folder [{:s}] already exists. Exit...'.format(save_Uncomp_folder)
# if save_LR_folder is not None:
#     os.makedirs(save_LR_folder)
#     print('mkdir [{:s}] ...'.format(save_LR_folder))
#     if upscale_kernel is not None:
#         np.save(os.path.join(save_LR_folder,'upscale_kernel.npy'),upscale_kernel)
if save_Uncomp_folder is not None:
    os.makedirs(save_Uncomp_folder)
    print('mkdir [{:s}] ...'.format(save_Uncomp_folder))

# img_list = []
# for file_name in os.listdir(input_folder):
#     img_list.append(os.path.join(input_folder,file_name))
progress_bar = tqdm(os.listdir(input_folder))
for file_name in progress_bar:
    cur_im = mod_img(cv2.imread(os.path.join(input_folder,file_name)),mod_scale)
    cur_im = cv2.cvtColor(cur_im,cv2.COLOR_RGB2GRAY)
    cv2.imwrite(os.path.join(save_Uncomp_folder,file_name.replace('.jpg','.png')),cur_im)
    # if save_LR_folder is not None:
    #     # cur_im = cv2.resize(cur_im,dsize=(0,0),fx=1/up_scale,fy=1/up_scale,interpolation = cv2.INTER_CUBIC)
    #     cur_im = imresize(cur_im,scale_factor=[1/float(up_scale)],kernel=upscale_kernel)
    #     cv2.imwrite(os.path.join(save_LR_folder,file_name.replace('.jpg','.png')),cur_im)



