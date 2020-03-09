import numpy as np
from tqdm import tqdm
import os
import cv2
import matplotlib.pyplot as plt

SHOW_RATIO = False

FOLDER_B = '/home/tiras/ybahat/SRGAN/results/MSE_srResNet/BSD100_cubic_PSNR23.667'
FOLDER_B = '/home/tiras/ybahat/SRGAN/results/Feat_with_MaxPool_srResNet/BSD100_cubic_PSNR23.267'
FOLDER_B = '/home/tiras/ybahat/SRGAN/results/Feat_with_VGG19Untrained_srResNet/BSD100_cubic_PSNR22.857'
# FOLDER_A = '/home/tiras/ybahat/SRGAN/results/Feat_with_untrained_randomMaxSize/BSD100_cubic_PSNR23.122'
# FOLDER_A = '/home/tiras/datasets/BSD100_test/BSD100_test_HRx4'
# FOLDER_B = '/home/tiras/ybahat/SRGAN/results/Feat_with_untrained_HalfChannels/BSD100_cubic_PSNR22.785'
# FOLDER_B = '/home/tiras/ybahat/SRGAN/results/Feat_with_untrained_TwiceChannels/BSD100_cubic_PSNR22.886'
FOLDER_A = '/media/ybahat/data/projects/SRGAN/results/Feat_with_untrained_noRelu/BSD100_cubic_PSNR22.513'

files_list = os.listdir(FOLDER_A)
FFT_ratios = []
max_dims = [0,0]
im_names = [im.split('.')[0].split('_')[0] for im in files_list]
ratio_range = [0,0]
for im_num,im_name in tqdm(enumerate(im_names)):
    image_A = np.array(cv2.imread(os.path.join(FOLDER_A,files_list[im_num]))).mean(2)/255
    FFT_A = np.fft.fft2(image_A)
    if SHOW_RATIO:
        file_name_B = [im for im in os.listdir(FOLDER_B) if im_name in im][0]
        image_B = np.array(cv2.imread(os.path.join(FOLDER_B,file_name_B))).mean(2)/255
        FFT_B = np.fft.fft2(image_B)
        FFT_ratios.append(np.log(np.abs(FFT_A)/(np.finfo(FFT_A.dtype).eps+np.abs(FFT_B))))
    else:
        FFT_ratios.append(np.log(np.abs(FFT_A)))

    max_dims[0] = max(max_dims[0],FFT_A.shape[0])
    max_dims[1] = max(max_dims[1],FFT_A.shape[1])
    ratio_range[0] = min(ratio_range[0],np.percentile(FFT_ratios[-1],1))
    ratio_range[1] = max(ratio_range[1],np.percentile(FFT_ratios[-1],99))
ratio_range = np.max(np.abs(ratio_range))

for i in range(len(FFT_ratios)):
    if SHOW_RATIO:
        FFT_ratios[i] = np.stack([-1*FFT_ratios[i]/ratio_range*(FFT_ratios[i]<0),FFT_ratios[i]/ratio_range*(FFT_ratios[i]>0),np.zeros_like(FFT_ratios[i])],-1)
        resized = []
        for ch in range(FFT_ratios[i].shape[2]):
            resized.append(cv2.resize(FFT_ratios[i][:,:,ch],dsize=tuple(max_dims)))
        FFT_ratios[i] = np.stack(resized,-1)
    else:
        FFT_ratios[i] = cv2.resize(FFT_ratios[i] / ratio_range,dsize=tuple(max_dims))

FFT_ratios = np.mean(np.stack(FFT_ratios),0)
FFT_ratios *= ratio_range
ratio_range = np.percentile(FFT_ratios,99)
FFT_ratios /= ratio_range
plt.imshow(FFT_ratios)
if not SHOW_RATIO:
    plt.colorbar()
title = 'Average log(|FFT(%s)|%s)'%(FOLDER_A.split('/')[-2],'\n/|FFT(%s)|'%(FOLDER_B.split('/')[-2]) if SHOW_RATIO else '')
plt.title(title)
saving_path = '/home/tiras/ybahat/SRGAN/results' if 'tiras' in os.getcwd() else '/media/ybahat/data/projects/SRGAN/results'
saving_name = 'FFT_%s%s_max%.2f.png'%(FOLDER_A.split('/')[-2],'_2_%s'%(FOLDER_B.split('/')[-2]) if SHOW_RATIO else '',ratio_range)
plt.savefig(os.path.join(saving_path,saving_name))
print('Done')