import os
import numpy as np
from tqdm import tqdm
from data.util import modcrop,bgr2ycbcr,ycbcr2rgb
import torch
from JPEG_module import JPEG
import cv2

dataset_folder = '/media/ybahat/data/Datasets/BSD100_test/BSD100_test_HR'
images_list = os.listdir(dataset_folder)

# jpeg_compressor_8 = JPEG.JPEG(compress=True,downsample_and_quantize=False,chroma_mode=True,block_size=8)
jpeg_compressor = JPEG.JPEG(compress=True, downsample_and_quantize=True, downsample_only=True, chroma_mode=True, block_size=16)
jpeg_extractor = JPEG.JPEG(compress=False, chroma_mode=True, block_size=16)
jpeg_compressor.Set_Q_Table(torch.tensor(90))
jpeg_extractor.Set_Q_Table(torch.tensor(90))
rmse_NN,rmse_interp,rmse_NN_orig = 0,0,0
for im_name in tqdm(images_list):
    image = cv2.imread(os.path.join(dataset_folder,im_name))
    image = bgr2ycbcr(modcrop(image,16), only_y=False).astype(float)
    im_shape = list(image.shape[:2])
    subsampled_chroma = np.array(image)[::2,::2,1:]
    recovered_image_NN =  np.tile(np.expand_dims(np.expand_dims(subsampled_chroma,2),1),[1,2,1,2,1]).reshape(im_shape+[-1])
    recovered_image_NN = 255*ycbcr2rgb(np.concatenate([np.expand_dims(image[...,0],-1),recovered_image_NN],-1)/255)
    recovered_image_interp = cv2.resize(subsampled_chroma,tuple(im_shape[::-1]), interpolation=cv2.INTER_LINEAR)
    recovered_image_interp = 255*ycbcr2rgb(np.concatenate([np.expand_dims(image[...,0],-1),recovered_image_interp],-1)/255)
    image_DCT = jpeg_compressor(torch.from_numpy(np.expand_dims(image.transpose((2, 0, 1)), 0)).cuda().float())
    recovered_image_DCT = np.clip(255*ycbcr2rgb(jpeg_extractor(image_DCT).data[0].cpu().numpy().transpose((1,2,0))/255),0,255)#.astype(np.uint8)
    rmse_NN += np.sqrt((np.mean(recovered_image_DCT.astype(float)-recovered_image_NN.astype(float))**2))
    rmse_interp += np.sqrt((np.mean(recovered_image_DCT.astype(float) - recovered_image_interp.astype(float)) ** 2))
    rmse_NN_orig += np.sqrt((np.mean(255*ycbcr2rgb(image/255).astype(float) - recovered_image_NN.astype(float)) ** 2))


print('Average RMSE: %.6f/%.6f (NN/bilinear) gray levels'%(rmse_NN/len(images_list),rmse_interp/len(images_list)))
print('Average RMSE NN-original: %.6f'%(rmse_NN_orig/len(images_list)))