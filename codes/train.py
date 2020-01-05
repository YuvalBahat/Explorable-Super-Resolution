import os.path
import sys
import math
import argparse
import time
import random
from collections import OrderedDict,deque
import matplotlib
matplotlib.use('Agg')
import torch
import numpy as np
import options.options as option
from utils import util
from data import create_dataloader, create_dataset
from models import create_model
from utils.logger import Logger, PrintLogger
import tqdm
from datetime import datetime

IGNORED_KEYS_LIST = ['l_d_real','l_d_fake','D_real','D_fake','psnr_val','LR_decrease','Correctly_distinguished','l_g_range','D_loss_STD','l_g_pix']

def main():
    # options
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Path to option JSON file.')
    parser.add_argument('-single_GPU', action='store_true',help='Utilize only one GPU')
    if parser.parse_args().single_GPU:
        available_GPUs = util.Assign_GPU()
    else:
        available_GPUs = util.Assign_GPU(max_GPUs=None)
    opt = option.parse(parser.parse_args().opt, is_train=True,batch_size_multiplier=len(available_GPUs))

    if not opt['train']['resume']:
        util.mkdir_and_rename(opt['path']['experiments_root'])  # Modify experiment name if exists
        util.mkdirs((path for key, path in opt['path'].items() if not key == 'experiments_root' and \
            not key == 'pretrain_model_G' and not key == 'pretrain_model_D'))
    option.save(opt)
    opt = option.dict_to_nonedict(opt)  # Convert to NoneDict, which return None for missing key.

    # print to file and std_out simultaneously
    sys.stdout = PrintLogger(opt['path']['log'])

    # random seed
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    print("Random Seed: ", seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # create train and val dataloader
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            max_accumulation_steps = max([opt['train']['grad_accumulation_steps_G'], opt['train']['grad_accumulation_steps_D']])
            train_set = create_dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt['batch_size']))
            print('Number of train images: {:,d}, iters: {:,d}'.format(len(train_set), train_size))
            # current_step = 0
            # if opt['train']['resume']:
            #     model_name = [name for name in os.listdir(opt['path']['models']) if '_G.pth' in name]
            #     model_name = sorted([m for m in model_name if re.search('(\d)+(?=_G.pth)', m) is not None], key=lambda x: int(re.search('(\d)+(?=_G.pth)', x).group(0)))[-1]
            #     current_step = int(re.search('(\d)+(?=_G.pth)', model_name).group(0))*max_accumulation_steps
            total_iters = int(opt['train']['niter']*max_accumulation_steps)#-current_step
            total_epoches = int(math.ceil(total_iters / train_size))
            print('Total epoches needed: {:d} for iters {:,d}'.format(total_epoches, total_iters))
            train_loader = create_dataloader(train_set, dataset_opt)
        elif phase == 'val':
            val_dataset_opt = dataset_opt
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt)
            print('Number of val images in [{:s}]: {:d}'.format(dataset_opt['name'], len(val_set)))
        else:
            raise NotImplementedError('Phase [{:s}] is not recognized.'.format(phase))
    assert train_loader is not None
    # Create model
    model = create_model(opt,max_accumulation_steps)
    # create logger
    logger = Logger(opt)
    # Save validation set results as image collage:
    SAVE_IMAGE_COLLAGE = True
    per_image_saved_patch = min([min(im['HR'].shape[1:]) for im in val_loader.dataset])-2
    num_val_images = len(val_loader.dataset)
    val_images_collage_rows = int(np.floor(np.sqrt(num_val_images)))
    while val_images_collage_rows>1:
        if np.round(num_val_images/val_images_collage_rows)==num_val_images/val_images_collage_rows:
            break
        val_images_collage_rows -= 1
    start_time = time.time()
    min_accumulation_steps = min([opt['train']['grad_accumulation_steps_G'],opt['train']['grad_accumulation_steps_D']])
    save_GT_HR = True
    lr_too_low = False
    print('---------- Start training -------------')
    last_saving_time = time.time()
    recently_saved_models = deque(maxlen=4)
    for epoch in range(int(math.floor(model.step / train_size)),total_epoches):
        for i, train_data in enumerate(train_loader):
            gradient_step_num = model.step // max_accumulation_steps
            not_within_batch = model.step % max_accumulation_steps == (max_accumulation_steps - 1)
            saving_step = ((time.time()-last_saving_time)>60*opt['logger']['save_checkpoint_freq']) and not_within_batch
            if saving_step:
                last_saving_time = time.time()

            # save models
            # if lr_too_low or (gradient_step_num % opt['logger']['save_checkpoint_freq'] == 0 and not_within_batch):
            if lr_too_low or saving_step:
                recently_saved_models.append(model.save(gradient_step_num))
                model.save_log()
                if lr_too_low:
                    break
                if len(recently_saved_models)>3:
                    model_2_delete = recently_saved_models.popleft()
                    os.remove(model_2_delete)
                    if model.D_exists:
                        os.remove(model_2_delete.replace('_G.','_D.'))
                print('{}: Saving the model before iter {:d}.'.format(datetime.now().strftime('%H:%M:%S'),gradient_step_num))

            if model.step > total_iters:
                break

            # training
            model.feed_data(train_data)
            model.optimize_parameters()

            time_elapsed = time.time() - start_time
            if not_within_batch:    start_time = time.time()

            # log
            if gradient_step_num % opt['logger']['print_freq'] == 0 and not_within_batch:
                logs = model.get_current_log()
                print_rlt = OrderedDict()
                print_rlt['model'] = opt['model']
                print_rlt['epoch'] = epoch
                print_rlt['iters'] = gradient_step_num
                print_rlt['time'] = time_elapsed
                for k, v in logs.items():
                    print_rlt[k] = v
                print_rlt['lr'] = model.get_current_learning_rate()
                logger.print_format_results('train', print_rlt,keys_ignore_list=IGNORED_KEYS_LIST)

            # validation
            if not_within_batch and (gradient_step_num) % opt['train']['val_freq'] == 0: # and gradient_step_num>=opt['train']['D_init_iters']:
                print_rlt = OrderedDict()
                if model.generator_changed:
                    print('---------- validation -------------')
                    start_time = time.time()
                    if False and SAVE_IMAGE_COLLAGE and model.gradient_step_num%opt['train']['val_save_freq'] == 0: #Saving training images:
                        GT_image_collage = []
                        cur_train_results = model.get_current_visuals(entire_batch=True)
                        train_psnrs = [
                            util.calculate_psnr(util.tensor2img(cur_train_results['SR'][im_num], out_type=np.float32) * 255,
                                                util.tensor2img(cur_train_results['HR'][im_num], out_type=np.float32) * 255) for
                            im_num in range(len(cur_train_results['SR']))]
                        #Save latest training batch output:
                        save_img_path = os.path.join(os.path.join(opt['path']['val_images']),
                                                     '{:d}_Tr_PSNR{:.3f}.png'.format(gradient_step_num, np.mean(train_psnrs)))
                        util.save_img(np.clip(np.concatenate((np.concatenate([util.tensor2img(cur_train_results['HR'][im_num], out_type=np.float32) * 255 for im_num in
                                 range(len(cur_train_results['SR']))],0), np.concatenate([util.tensor2img(cur_train_results['SR'][im_num], out_type=np.float32) * 255 for im_num in
                                 range(len(cur_train_results['SR']))],0)), 1), 0, 255).astype(np.uint8), save_img_path)
                    Z_latent = [0]+([-1,1] if opt['network_G']['latent_input'] else [])
                    print_rlt['psnr'] = 0
                    for cur_Z in Z_latent:
                        model.perform_validation(data_loader=val_loader,cur_Z=cur_Z,print_rlt=print_rlt,save_GT_HR=save_GT_HR,
                                                 save_images=((model.gradient_step_num) % opt['train']['val_save_freq'] == 0) or save_GT_HR)
                        if save_GT_HR:  # Save GT Uncomp images
                            save_GT_HR = False
                    model.log_dict['psnr_val'].append((gradient_step_num,print_rlt['psnr']/len(Z_latent)))
                else:
                    print('Skipping validation because generator is unchanged')
                time_elapsed = time.time() - start_time
                # Save to log
                print_rlt['model'] = opt['model']
                print_rlt['epoch'] = epoch
                print_rlt['iters'] = gradient_step_num
                print_rlt['time'] = time_elapsed
                model.display_log_figure()
                model.generator_changed = False
                logger.print_format_results('val', print_rlt,keys_ignore_list=IGNORED_KEYS_LIST)
                print('-----------------------------------')

            # update learning rate
            if not_within_batch:
                lr_too_low = model.update_learning_rate(gradient_step_num)
            # current_step += 1
        if lr_too_low:
            print('Stopping training because LR is too low')
            break

    print('Saving the final model.')
    model.save(gradient_step_num)
    print('End of training.')


if __name__ == '__main__':
    # # OpenCV get stuck in transform when used in DataLoader
    # # https://github.com/pytorch/pytorch/issues/1838
    # # However, cause problem reading lmdb
    # import torch.multiprocessing as mp
    # mp.set_start_method('spawn', force=True)
    main()
