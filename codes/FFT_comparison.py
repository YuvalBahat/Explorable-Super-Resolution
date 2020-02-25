import numpy as np
from tqdm import tqdm
import os
import cv2
from scipy.misc import imresize
import matplotlib.pyplot as plt

FOLDER_B = '/media/ybahat/data/projects/SRGAN/results/MSE_srResNet/BSD100_cubic_PSNR23.598'
FOLDER_A = '/media/ybahat/data/projects/SRGAN/results/Feat_with_MaxPool_srResNet/BSD100_cubic_PSNR23.277'

files_list = os.listdir(FOLDER_A)
FFT_ratios = []
max_dims = [0,0]
im_names = [im.split('.')[0].split('_')[0] for im in files_list]
for im_num,im_name in tqdm(enumerate(im_names)):
    file_name_B = [im for im in os.listdir(FOLDER_B) if im_name in im][0]
    image_A = np.array(cv2.imread(os.path.join(FOLDER_A,files_list[im_num]))).mean(2)/255
    image_B = np.array(cv2.imread(os.path.join(FOLDER_B,file_name_B))).mean(2)/255
    FFT_A = np.fft.fft2(image_A)
    FFT_B = np.fft.fft2(image_B)
    FFT_ratios.append(np.log(np.abs(FFT_A)/(np.finfo(FFT_A.dtype).eps+np.abs(FFT_B))))
    max_dims[0] = max(max_dims[0],FFT_B.shape[0])
    max_dims[1] = max(max_dims[1],FFT_B.shape[1])

for ratio in FFT_ratios:
    ratio = imresize(ratio,max_dims)

plt.imshow(ratio)
plt.colorbar()
title = 'Average log(|FFT(%s)|/|FFT(%s)|)'%(FOLDER_A.split('/')[-2],FOLDER_B.split('/')[-2])
plt.title(title)
plt.savefig(os.path.join('/media/ybahat/data/projects/SRGAN/results','FFT_%s_2_%s.png'%(FOLDER_A.split('/')[-2],FOLDER_B.split('/')[-2])))
print('Done')