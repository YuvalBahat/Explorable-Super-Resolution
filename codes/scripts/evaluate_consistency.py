import CEM.CEMnet as CEMnet
from CEM.imresize_CEM import imresize
import data.util as util
import matplotlib.pyplot as plt
import numpy as np
import os

SCALE_FACTOR = 32
# LR_IM_PATH = '/home/ybahat/Dropbox/PhD/ExplorableSR/SharedTomerYuval/PULSE/face.png' # None
LR_IM_PATH = '/home/ybahat/Dropbox/PhD/ExplorableSR/SharedTomerYuval/PULSE/Depixelized_LR.png' # None
PIXELATED_IM_PATH = '/home/ybahat/Dropbox/PhD/ExplorableSR/SharedTomerYuval/PULSE/pixel-barack.png'
# SR_IM_PATH = '/home/ybahat/Dropbox/PhD/ExplorableSR/SharedTomerYuval/PULSE/face.png'
SR_IM_PATH = '/home/ybahat/Dropbox/PhD/ExplorableSR/SharedTomerYuval/PULSE/Obama'
sr_image_paths = [os.path.join(SR_IM_PATH,f) for f in os.listdir(SR_IM_PATH) if '.png' in f]

CEM_conf = CEMnet.Get_CEM_Conf(SCALE_FACTOR)
CEM_conf.lower_magnitude_bound = 0.1
CEM_net = CEMnet.CEMnet(CEM_conf, upscale_kernel=None)
CEM_net.WrapArchitecture_PyTorch(only_padders=True)

if LR_IM_PATH is None:
    # Loading pixelated image:
    LR_im = util.read_img(None, PIXELATED_IM_PATH)[:,:,[2,1,0]]

    # Converting to LR:
    MAJORITY_THRESHOLD = 0.2
    vert_pixel_borders = np.concatenate([np.zeros([1,1]).astype(np.int),1+np.argwhere(np.mean(np.mean(np.diff(LR_im,axis=0)!=0,2)>MAJORITY_THRESHOLD,1)>MAJORITY_THRESHOLD).reshape([-1,1])])
    horiz_pixel_borders = np.concatenate([np.zeros([1,1]).astype(np.int),1+np.argwhere(np.mean(np.mean(np.diff(LR_im,axis=1)!=0,2)>MAJORITY_THRESHOLD,0)>MAJORITY_THRESHOLD).reshape([1,-1])],1)
    LR_im = LR_im[vert_pixel_borders,horiz_pixel_borders,:]
    plt.imsave('Depixelized_LR.png',LR_im)
else:
    LR_im = util.read_img(None, LR_IM_PATH)[:, :, [2, 1, 0]]

# Adding 1 row at the bottom (edge-padding) to make the size 32x32:
LR_im = np.concatenate([LR_im,LR_im[-1:,:,:]],0)
# plt.imshow(LR_im)

# Loading PULSE SR output:
for SR_path in sr_image_paths:
    print('Processing image %s...'%(SR_path))
    SR_im = util.read_img(None, SR_path)[:,:,[2,1,0]]
    downsampled_SR = imresize(SR_im,1/SCALE_FACTOR)
    plt.imsave(SR_path[:-4]+'_DS.png',np.clip(downsampled_SR,0,1))
    consistent_im = CEM_net.Enforce_DT_on_Image_Pair(LR_im,SR_im)
    plt.imsave(SR_path[:-4]+'_consistent.png',np.clip(consistent_im,0,1))
    downsampled_consistent = imresize(consistent_im,1/SCALE_FACTOR)
    plt.imsave(SR_path[:-4] + '_consistent_DS.png', np.clip(downsampled_consistent, 0, 1))


# LR_im_padded = np.pad(LR_im,((CEM_net.invalidity_margins_LR,CEM_net.invalidity_margins_LR),(CEM_net.invalidity_margins_LR,CEM_net.invalidity_margins_LR),(0,0)),'edge')
# SR_im_padded = np.pad(SR_im,((CEM_net.invalidity_margins_HR,CEM_net.invalidity_margins_HR),(CEM_net.invalidity_margins_HR,CEM_net.invalidity_margins_HR),(0,0)),'edge')
# consistent_im_padded = CEM_net.Enforce_DT_on_Image_Pair(LR_im_padded,SR_im_padded)[CEM_net.invalidity_margins_HR:-CEM_net.invalidity_margins_HR,CEM_net.invalidity_margins_HR:-CEM_net.invalidity_margins_HR,:]
# plt.imsave('consistent_SR_padded.png',np.clip(consistent_im_padded,0,1))