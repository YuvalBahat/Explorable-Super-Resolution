import os
import sys
import time
import argparse
import numpy as np
from collections import OrderedDict
import cv2
import matplotlib
matplotlib.use('Qt5Agg')
matplotlib.interactive(True)
import matplotlib.pyplot as plt
from tqdm import tqdm
import options.options as option
import utils.util as util
from data.util import bgr2ycbcr
from data import create_dataset, create_dataloader
from models import create_model
from utils.logger import PrintLogger,Logger
from scipy.stats import norm
import imageio
import torch
import subprocess

# Parameters:
SAVE_IMAGE_COLLAGE = False
TEST_LATENT_OUTPUT = 'stats'#'GIF','video',None,'stats'
# Parameters for GIF:
LATENT_DISTRIBUTION = 'rand_Uniform'#'Uniform'#'rand_Uniform','Gaussian','Input_Z_Im','Desired_Im','max_STD','min_STD','UnDesired_Im','Desired_Im_VGG','UnDesired_Im_VGG','UnitCircle','Desired_Im_hist
NUM_Z_ITERS = 250
NON_ARBITRARY_Z_INPUTS = ['Input_Z_Im','Desired_Im','max_STD','min_STD','UnDesired_Im','Desired_Im_VGG','UnDesired_Im_VGG','Desired_Im_hist'] #
LATENT_RANGE = 1
NUM_SAMPLES = 50#Must be odd for a collage to be saved
INPUT_Z_IM_PATH = os.path.join('/home/ybahat/Dropbox/PhD/DataTermEnforcingArch/Results/SRGAN/NoiseInput',LATENT_DISTRIBUTION)
if 'Desired_Im' in LATENT_DISTRIBUTION:
    INPUT_Z_IM_PATH = INPUT_Z_IM_PATH.replace(LATENT_DISTRIBUTION,'Desired_Im')
TEST_IMAGE = None#'comic'#None
LATENT_CHANNEL_NUM = 0#Overridden when UnitCircle
OTHER_CHANNELS_VAL = 0
# options
parser = argparse.ArgumentParser()
parser.add_argument('-opt', type=str, required=True, help='Path to options JSON file.')
parser.add_argument('-single_GPU', action='store_true', help='Utilize only one GPU')
if parser.parse_args().single_GPU:
    util.Assign_GPU()
opt = option.parse(parser.parse_args().opt, is_train=False)
util.mkdirs((path for key, path in opt['path'].items() if not key == 'pretrain_model_G'))
opt = option.dict_to_nonedict(opt)
if LATENT_DISTRIBUTION in NON_ARBITRARY_Z_INPUTS:
    LATENT_CHANNEL_NUM = None
else:
    TEST_IMAGE = None

# print to file and std_out simultaneously
sys.stdout = PrintLogger(opt['path']['log'])
print('\n**********' + util.get_timestamp() + '**********')

# Create test dataset and dataloader
test_loaders = []
for phase, dataset_opt in sorted(opt['datasets'].items()):
    test_set = create_dataset(dataset_opt,specific_image=TEST_IMAGE)
    test_loader = create_dataloader(test_set, dataset_opt)
    print('Number of test images in [{:s}]: {:d}'.format(dataset_opt['name'], len(test_set)))
    test_loaders.append(test_loader)

# Create model
if 'VGG' in LATENT_DISTRIBUTION:
    model = create_model(opt,init_Fnet=True)
else:
    model = create_model(opt)
# assert SAVE_IMAGE_COLLAGE or not TEST_LATENT_OUTPUT,'Must use image collage for creating GIF'
# TEST_LATENT_OUTPUT = TEST_LATENT_OUTPUT if opt['network_G']['latent_input'] else None
assert len(test_set)==1 or LATENT_DISTRIBUTION not in NON_ARBITRARY_Z_INPUTS or not TEST_LATENT_OUTPUT,'Use 1 image only for these Z input types'
assert np.round(NUM_SAMPLES/2)!=NUM_SAMPLES/2 or not SAVE_IMAGE_COLLAGE,'Pick an odd number of samples'
assert LATENT_DISTRIBUTION == 'rand_Uniform' or TEST_LATENT_OUTPUT!='stats','Why not using rand_uniform when collecting stats?'
for test_loader in test_loaders:
    test_set_name = test_loader.dataset.opt['name']
    print('\nTesting [{:s}]...'.format(test_set_name))
    test_start_time = time.time()
    dataset_dir = os.path.join(opt['path']['results_root'], test_set_name)
    util.mkdir(dataset_dir)
    num_val_images = len(test_loader.dataset)
    if SAVE_IMAGE_COLLAGE:
        per_image_saved_patch = min([min(im['HR'].shape[1:]) for im in test_loader.dataset])
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
    if TEST_LATENT_OUTPUT:
        if LATENT_DISTRIBUTION=='Gaussian':#When I used Gaussian latent input, I set the range to cover LATENT_RANGE of the probability:
            optional_Zs = np.arange(start=-2,stop=0,step=0.001)
            optional_Zs = optional_Zs[int(np.argwhere(norm.cdf(optional_Zs)>=(1-LATENT_RANGE)/2)[0]):]
            optional_Zs = optional_Zs#+[0]+[-1*val for val in optional_Zs[::-1]]
            Z_latent = []
            for frame_num in range(int((NUM_SAMPLES-1)/2)):
                Z_latent.append(optional_Zs[int(frame_num*len(optional_Zs)/((NUM_SAMPLES-1)/2))])
            Z_latent = Z_latent+[0]+[-1*val for val in Z_latent[::-1]]
        elif LATENT_DISTRIBUTION=='Uniform':
            Z_latent = list(np.linspace(start=-LATENT_RANGE,stop=0,num=np.ceil(NUM_SAMPLES/2)))[:-1]
            Z_latent = Z_latent+[0]+[-z for z in Z_latent[::-1]]
        elif LATENT_DISTRIBUTION == 'rand_Uniform':
            Z_latent = np.random.uniform(low=-LATENT_RANGE,high=LATENT_RANGE,size=[NUM_SAMPLES,1,model.num_latent_channels,1,1]) if opt['network_G']['DTE_arch'] else np.zeros(1)
        elif LATENT_DISTRIBUTION=='Input_Z_Im' or 'Desired_Im' in LATENT_DISTRIBUTION:
            Z_image_names = os.listdir(INPUT_Z_IM_PATH)
            if 'Desired_Im' in LATENT_DISTRIBUTION:
                logger = Logger(opt)
                Z_image_names = [im for im in Z_image_names if im.split('.')[0]==TEST_IMAGE]
            Z_latent = [imageio.imread(os.path.join(INPUT_Z_IM_PATH,im)) for im in Z_image_names]
        elif LATENT_DISTRIBUTION=='UnitCircle':
            LATENT_CHANNEL_NUM = 1#For folder name only
            thetas = np.linspace(0,2*np.pi*(NUM_SAMPLES-1)/NUM_SAMPLES,num=NUM_SAMPLES)
            Z_latent = [np.reshape([OTHER_CHANNELS_VAL]+list(util.pol2cart(1,theta)),newshape=[1,3,1,1]) for theta in thetas]
        else:
            logger = Logger(opt)
            Z_latent = [None]
            Z_image_names = [None]
    else:
        Z_latent = [0]
    frames = []
    if LATENT_DISTRIBUTION not in NON_ARBITRARY_Z_INPUTS+['UnitCircle','rand_Uniform']:
        Z_latent = sorted(Z_latent)
    image_idx = -1
    for data in tqdm(test_loader):
        image_idx += 1
        image_high_freq_versions = []
        for z_sample_num,cur_Z_raw in enumerate(Z_latent):
            if SAVE_IMAGE_COLLAGE:
                image_collage, GT_image_collage = [], []
            if LATENT_DISTRIBUTION in NON_ARBITRARY_Z_INPUTS:
                cur_Z_image = cur_Z_raw
            elif LATENT_DISTRIBUTION not in NON_ARBITRARY_Z_INPUTS+['UnitCircle','rand_Uniform'] and model.num_latent_channels > 1:
                cur_Z = np.reshape(np.stack((model.num_latent_channels * [OTHER_CHANNELS_VAL])[:LATENT_CHANNEL_NUM] +
                                            [cur_Z_raw] + (model.num_latent_channels * [OTHER_CHANNELS_VAL])[LATENT_CHANNEL_NUM + 1:],0),[1,-1,1,1])
                cur_channel_cur_Z = cur_Z if isinstance(cur_Z, int) else cur_Z[0, LATENT_CHANNEL_NUM].squeeze()
            elif LATENT_DISTRIBUTION=='UnitCircle':
                cur_channel_cur_Z = np.mod(np.arctan2(cur_Z_raw[0,2],cur_Z_raw[0,1]),2*np.pi)/2/np.pi*360
            elif LATENT_DISTRIBUTION == 'rand_Uniform':
                cur_Z = cur_Z_raw
                cur_channel_cur_Z = None # Not any more: 0 here causes PSNR calculations (and other stuff) to be performed.
            if SAVE_IMAGE_COLLAGE and image_idx % val_images_collage_rows == 0:  image_collage.append([]);   GT_image_collage.append([])
            need_HR = False if test_loader.dataset.opt['dataroot_HR'] is None else True
            img_path = data['LR_path'][0]
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            if TEST_LATENT_OUTPUT:
                if LATENT_DISTRIBUTION == 'Input_Z_Im':
                    cur_Z = util.Convert_Im_2_Zinput(Z_image=cur_Z_image,im_size=data['LR'].size()[2:],Z_range=LATENT_RANGE,single_channel=model.num_latent_channels==1)
                elif 'Desired_Im' in LATENT_DISTRIBUTION:
                    LR_Z = 1e-1
                    objective = ('max_' if 'UnDesired_Im' in LATENT_DISTRIBUTION else '')+('VGG' if 'VGG' in LATENT_DISTRIBUTION else ('hist' if 'hist' in LATENT_DISTRIBUTION else 'L1'))
                    Z_optimizer = util.Z_optimizer(objective=objective,LR_size=data['LR'].size()[2:],model=model,Z_range=LATENT_RANGE,initial_LR=LR_Z,loggers=logger,max_iters=NUM_Z_ITERS,data=data)
                    cur_Z = Z_optimizer.optimize()
                elif 'STD' in LATENT_DISTRIBUTION:
                    LR_Z = 1e-1
                    Z_optimizer = util.Z_optimizer(objective=LATENT_DISTRIBUTION,LR_size=data['LR'].size()[2:],model=model,Z_range=LATENT_RANGE,initial_LR=LR_Z,loggers=logger,max_iters=NUM_Z_ITERS,data=data)
                    cur_Z = Z_optimizer.optimize()
            data['Z'] = cur_Z
            model.feed_data(data, need_HR=need_HR)

            model.test()  # test
            visuals = model.get_current_visuals(need_HR=need_HR)

            sr_img = util.tensor2img(visuals['SR'],out_type=np.float32)  # float32

            # save images
            suffix = opt['suffix']
            if not SAVE_IMAGE_COLLAGE:
                if TEST_LATENT_OUTPUT=='stats':
                    save_img_path = os.path.join(dataset_dir, img_name + '_s%%0%dd'%(len(str(NUM_SAMPLES-1)))%(z_sample_num) + '.png')
                elif suffix:
                    save_img_path = os.path.join(dataset_dir, img_name + suffix + '.png')
                else:
                    save_img_path = os.path.join(dataset_dir, img_name + '.png')
                util.save_img((255*sr_img).astype(np.uint8), save_img_path)

            # calculate PSNR and SSIM
            if need_HR:
                if z_sample_num==0:
                    if opt['network_G']['DTE_arch']:
                        gt_img = util.tensor2img(visuals['HR'], out_type=np.float32)  # float32
                        img_projected_2_kernel_subspace = model.DTE_net.Project_2_Subspace(gt_img)
                        gt_orthogonal_component = gt_img-img_projected_2_kernel_subspace #model.DTE_net.Return_Orthogonal_Component(gt_img)
                        HR_STD = 255*np.std(gt_orthogonal_component,axis=(0,1)).mean()
                        gt_img *= 255.
                    else:
                        HR_STD = 0
                if TEST_LATENT_OUTPUT=='stats':
                    if opt['network_G']['DTE_arch']:
                        image_high_freq_versions.append(sr_img-img_projected_2_kernel_subspace)
                    if z_sample_num==(len(Z_latent)-1):
                        if opt['network_G']['DTE_arch']:
                            # normalized_pixel_STD = np.mean(np.std(np.stack(image_high_freq_versions),0))/np.std(gt_orthogonal_component,axis=(0,1)).mean()
                            pixel_STD = 255*np.mean(np.std(np.stack(image_high_freq_versions),0))
                        else:
                            # normalized_pixel_STD = 0
                            pixel_STD = 0
                        # Save GT image for reference:
                        util.save_img((255 * util.tensor2img(visuals['HR'], out_type=np.float32)).astype(np.uint8),
                                      os.path.join(dataset_dir, img_name + '_HR_STD%.3f_SR_STD%.3f.png'%(HR_STD,pixel_STD)))
                sr_img *= 255.
                if LATENT_DISTRIBUTION in NON_ARBITRARY_Z_INPUTS or cur_channel_cur_Z==0:
                    psnr = util.calculate_psnr(sr_img, gt_img)
                    ssim = util.calculate_ssim(sr_img, gt_img)
                    test_results['psnr'].append(psnr)
                    test_results['ssim'].append(ssim)
                if SAVE_IMAGE_COLLAGE:
                    if len(test_set)>1:
                        margins2crop = ((np.array(sr_img.shape[:2]) - per_image_saved_patch) / 2).astype(np.int32)
                    else:
                        margins2crop = [0,0]
                    image_collage[-1].append(np.clip(util.crop_center(sr_img,margins2crop), 0,255).astype(np.uint8))
                    if LATENT_DISTRIBUTION in NON_ARBITRARY_Z_INPUTS or cur_channel_cur_Z==0:
                        # Save GT HR images:
                        GT_image_collage[-1].append(
                            np.clip(util.crop_center(gt_img,margins2crop), 0,255).astype(np.uint8))
            # else:
            #     print(img_name)
        if SAVE_IMAGE_COLLAGE:
            cur_collage = np.concatenate([np.concatenate(col, 0) for col in image_collage], 1)
            if LATENT_DISTRIBUTION in NON_ARBITRARY_Z_INPUTS or cur_channel_cur_Z==0:
                if suffix:
                    save_img_path = os.path.join(dataset_dir+ suffix + '%s.png')
                else:
                    save_img_path = os.path.join(dataset_dir+ '.png')
                util.save_img(cur_collage, save_img_path%(''))
                # Save GT HR images:
                util.save_img(np.concatenate([np.concatenate(col, 0) for col in GT_image_collage], 1),save_img_path%'_GT_HR')
            if TEST_LATENT_OUTPUT:
                if LATENT_DISTRIBUTION not in NON_ARBITRARY_Z_INPUTS:
                    cur_collage = cv2.putText(cur_collage, '%.2f'%(cur_channel_cur_Z), (0, 50),cv2.FONT_HERSHEY_SCRIPT_COMPLEX, fontScale=2.0, color=(255, 255, 255))
                frames.append(cur_collage)
        if need_HR and ((LATENT_DISTRIBUTION not in NON_ARBITRARY_Z_INPUTS and cur_channel_cur_Z==0) or not TEST_LATENT_OUTPUT):  # metrics
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
    if TEST_LATENT_OUTPUT in ['GIF','video']:
        folder_name = os.path.join(dataset_dir+ suffix +'_%s'%(LATENT_DISTRIBUTION)+ '_%d%s'%(model.gradient_step_num,'_frames' if LATENT_DISTRIBUTION not in NON_ARBITRARY_Z_INPUTS else ''))
        if model.num_latent_channels>1 and LATENT_DISTRIBUTION not in NON_ARBITRARY_Z_INPUTS:
            folder_name += '_Ch%d'%(LATENT_CHANNEL_NUM)
        if TEST_LATENT_OUTPUT=='GIF':
            frames = [frame[:,:,::-1] for frame in frames]#Channels are originally ordered as BGR for cv2
        elif TEST_LATENT_OUTPUT == 'video':
            video = cv2.VideoWriter(folder_name+'.avi',0,25,frames[0].shape[:2])
        if not os.path.isdir(folder_name):        os.mkdir(folder_name)
        for i,frame in enumerate(frames+(frames[-2:0:-1] if LATENT_DISTRIBUTION not in NON_ARBITRARY_Z_INPUTS+['UnitCircle'] else [])):
            if TEST_LATENT_OUTPUT == 'GIF':
                if LATENT_DISTRIBUTION in NON_ARBITRARY_Z_INPUTS:
                    im_name = ''
                    if Z_image_names[i] is not None:
                        im_name += Z_image_names[i].split('.')[0]
                    im_name += '_PSNR%.3f'%(test_results['psnr'][i])
                else:
                    im_name = '%d'%(i)
                    # im_name = ('%d_%.2f'%(i,(Z_latent+Z_latent[-2:0:-1])[i])).replace('.','_')
                imageio.imsave(os.path.join(folder_name,'%s.png'%(im_name)),frame)
                if LATENT_DISTRIBUTION in NON_ARBITRARY_Z_INPUTS and LATENT_DISTRIBUTION!='Input_Z_Im':
                    Z_2_save = cur_Z.data.cpu().numpy().transpose((2,3,1,0)).squeeze()
                    imageio.imsave(os.path.join(folder_name,'%s_Z.png'%(im_name)),((Z_2_save+LATENT_RANGE)/2*255/LATENT_RANGE).astype(np.uint8))
            elif TEST_LATENT_OUTPUT=='video':
                video.write(frame)
        if TEST_LATENT_OUTPUT == 'video':
            cv2.destroyAllWindows()
            video.release()
        elif LATENT_DISTRIBUTION not in NON_ARBITRARY_Z_INPUTS:
            os.chdir(folder_name)
            subprocess.call(['ffmpeg', '-r','5','-i', '%d.png','-b:v','2000k', 'CH_%d.avi'%(LATENT_CHANNEL_NUM)])
