import os
import sys
import time
import argparse
import numpy as np
from collections import OrderedDict

import options.options as option
import utils.util as util
from data.util import bgr2ycbcr
from data import create_dataset, create_dataloader
from models import create_model
from utils.logger import PrintLogger
from scipy.stats import norm
import imageio

# options
parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, required=True, help='Path to options JSON file.')
parser.add_argument('-single_GPU', action='store_true', help='Utilize only one GPU')
if parser.parse_args().single_GPU:
    util.Assign_GPU()
opt = option.parse(parser.parse_args().opt, is_train=False)
util.mkdirs((path for key, path in opt['path'].items() if not key == 'pretrain_model_G'))
opt = option.dict_to_nonedict(opt)

# print to file and std_out simultaneously
sys.stdout = PrintLogger(opt['path']['log'])
print('\n**********' + util.get_timestamp() + '**********')

# Create test dataset and dataloader
test_loaders = []
for phase, dataset_opt in sorted(opt['datasets'].items()):
    test_set = create_dataset(dataset_opt)
    test_loader = create_dataloader(test_set, dataset_opt)
    print('Number of test images in [{:s}]: {:d}'.format(dataset_opt['name'], len(test_set)))
    test_loaders.append(test_loader)

# Create model
model = create_model(opt)
SAVE_IMAGE_COLLAGE = True
SAVE_GIF_4_STOCHASTIC = True
# Parameters for GIF:
LATENT_RANGE = 0.9
NUM_SAMPLES = 31#Must be odd for a collage to be saved
assert SAVE_IMAGE_COLLAGE or not SAVE_GIF_4_STOCHASTIC,'Must use image collage for creating GIF'
assert np.round(NUM_SAMPLES/2)!=NUM_SAMPLES/2,'Pick an odd number of samples'
for test_loader in test_loaders:
    test_set_name = test_loader.dataset.opt['name']
    print('\nTesting [{:s}]...'.format(test_set_name))
    test_start_time = time.time()
    dataset_dir = os.path.join(opt['path']['results_root'], test_set_name)
    util.mkdir(dataset_dir)

    per_image_saved_patch = min([min(im['HR'].shape[1:]) for im in test_loader.dataset])-2
    num_val_images = len(test_loader.dataset)
    val_images_collage_rows = int(np.floor(np.sqrt(num_val_images)))
    while val_images_collage_rows>1:
        if np.round(num_val_images/val_images_collage_rows)==num_val_images/val_images_collage_rows:
            break
        val_images_collage_rows -= 1

    test_results = OrderedDict()
    test_results['psnr'] = []
    test_results['ssim'] = []
    test_results['psnr_y'] = []
    test_results['ssim_y'] = []
    SAVE_GIF_4_STOCHASTIC = SAVE_GIF_4_STOCHASTIC and opt['network_G']['latent_input']
    if SAVE_GIF_4_STOCHASTIC:
        optional_Zs = np.arange(start=-2,stop=0,step=0.001)
        optional_Zs = optional_Zs[int(np.argwhere(norm.cdf(optional_Zs)>=(1-LATENT_RANGE)/2)[0]):]
        optional_Zs = optional_Zs#+[0]+[-1*val for val in optional_Zs[::-1]]
        Z_latent = []
        for frame_num in range(int((NUM_SAMPLES-1)/2)):
            Z_latent.append(optional_Zs[int(frame_num*len(optional_Zs)/((NUM_SAMPLES-1)/2))])
        Z_latent = Z_latent+[0]+[-1*val for val in Z_latent[::-1]]
    else:
        Z_latent = [0]
    frames = []
    for cur_Z in sorted(Z_latent):
        if SAVE_IMAGE_COLLAGE:
            image_collage, GT_image_collage = [], []
        idx = 0
        for data in test_loader:
            if idx % val_images_collage_rows == 0:  image_collage.append([]);   GT_image_collage.append([])
            idx += 1
            need_HR = False if test_loader.dataset.opt['dataroot_HR'] is None else True
            data['Z'] = cur_Z
            model.feed_data(data, need_HR=need_HR)
            img_path = data['LR_path'][0]
            img_name = os.path.splitext(os.path.basename(img_path))[0]

            model.test()  # test
            visuals = model.get_current_visuals(need_HR=need_HR)

            sr_img = util.tensor2img(visuals['SR'],out_type=np.float32)  # float32

            # save images
            suffix = opt['suffix']
            if not SAVE_IMAGE_COLLAGE:
                if suffix:
                    save_img_path = os.path.join(dataset_dir, img_name + suffix + '.png')
                else:
                    save_img_path = os.path.join(dataset_dir, img_name + '.png')
                util.save_img(sr_img, save_img_path)

            # calculate PSNR and SSIM
            if need_HR:
                sr_img *= 255.
                if cur_Z==0:
                    gt_img = util.tensor2img(visuals['HR'],out_type=np.float32)  # float32
                    gt_img *= 255.
                    psnr = util.calculate_psnr(sr_img, gt_img)
                    ssim = util.calculate_ssim(sr_img, gt_img)
                    test_results['psnr'].append(psnr)
                    test_results['ssim'].append(ssim)
                if SAVE_IMAGE_COLLAGE:
                    margins2crop = ((np.array(sr_img.shape[:2]) - per_image_saved_patch) / 2).astype(np.int32)
                    image_collage[-1].append(
                        np.clip(sr_img[margins2crop[0]:-margins2crop[0], margins2crop[1]:-margins2crop[1], :], 0,
                                255).astype(np.uint8))
                    if cur_Z==0:
                        # Save GT HR images:
                        GT_image_collage[-1].append(
                            np.clip(gt_img[margins2crop[0]:-margins2crop[0], margins2crop[1]:-margins2crop[1], :], 0,
                                    255).astype(np.uint8))
            # else:
            #     print(img_name)
        if SAVE_IMAGE_COLLAGE:
            cur_collage = np.concatenate([np.concatenate(col, 0) for col in image_collage], 1)
            if cur_Z==0:
                if suffix:
                    save_img_path = os.path.join(dataset_dir+ suffix + '%s.png')
                else:
                    save_img_path = os.path.join(dataset_dir+ '.png')
                util.save_img(cur_collage, save_img_path%(''))
                # Save GT HR images:
                util.save_img(np.concatenate([np.concatenate(col, 0) for col in GT_image_collage], 1),
                              save_img_path%'_GT_HR')
        if SAVE_GIF_4_STOCHASTIC:
            frames.append(cur_collage)
        if need_HR and cur_Z==0:  # metrics
            # Average PSNR/SSIM results
            ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
            ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
            print('----Average PSNR/SSIM results for {}----\n\tPSNR: {:.6f} dB; SSIM: {:.6f}\n'\
                    .format(test_set_name, ave_psnr, ave_ssim))
            os.rename(dataset_dir,dataset_dir+'_PSNR{:.3f}'.format(ave_psnr))
            if test_results['psnr_y'] and test_results['ssim_y']:
                ave_psnr_y = sum(test_results['psnr_y']) / len(test_results['psnr_y'])
                ave_ssim_y = sum(test_results['ssim_y']) / len(test_results['ssim_y'])
                print('----Y channel, average PSNR/SSIM----\n\tPSNR_Y: {:.6f} dB; SSIM_Y: {:.6f}\n'\
                    .format(ave_psnr_y, ave_ssim_y))
    if SAVE_GIF_4_STOCHASTIC:
        frames = [frame[:,:,::-1] for frame in frames]#Channels are originally ordered as BGR for cv2
        os.mkdir(os.path.join(dataset_dir+ suffix + '_%d_frames'%(model.gradient_step_num)))
        for i,frame in enumerate(frames+frames[-2:0:-1]):
            imageio.imsave(os.path.join(dataset_dir+ suffix + '_%d_frames'%(model.gradient_step_num),'%d.png'%(i)),frame)
        # imageio.mimsave(os.path.join(dataset_dir+ suffix + '_%d.gif'%(model.gradient_step_num)),frames+frames[-2:0:-1],fps=2)