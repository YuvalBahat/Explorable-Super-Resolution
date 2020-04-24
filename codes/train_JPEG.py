import os.path
import sys
import math
import argparse
import time
import random
from collections import OrderedDict,deque
import matplotlib
matplotlib.use('Agg')
# matplotlib.use('Qt5Agg')
import torch
import numpy as np
import options.options as option
from utils import util
from data import create_dataloader, create_dataset
from models import create_model
from utils.logger import Logger, PrintLogger
import tqdm
from datetime import datetime
import cv2
import copy

# USE_Y_GENERATOR_4_CHROMA = True

def main():
    # options
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Path to option JSON file.')
    parser.add_argument('-single_GPU', action='store_true',help='Utilize only one GPU')
    parser.add_argument('-chroma', action='store_true',help='Training the chroma-channels generator')
    if parser.parse_args().single_GPU:
        available_GPUs = util.Assign_GPU(maxMemory=0.66)
    else:
        # available_GPUs = util.Assign_GPU(max_GPUs=None,maxMemory=0.8,maxLoad=0.8)
        available_GPUs = util.Assign_GPU(max_GPUs=None)
    opt = option.parse(parser.parse_args().opt, is_train=True,batch_size_multiplier=len(available_GPUs),name='JPEG'+('_chroma' if parser.parse_args().chroma else ''))

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
    DEBUG = False
    # Create model
    if DEBUG:
        from models.base_model import BaseModel
        model = BaseModel
        model.step = 0
    else:
        model = create_model(opt,max_accumulation_steps,chroma_mode=opt['name'][:len('JPEG/chroma')]=='JPEG/chroma')
    # if model.chroma_mode:
    #     if USE_Y_GENERATOR_4_CHROMA:
    #         opt_Y = option.parse(parser.parse_args().opt, is_train=False,batch_size_multiplier=len(available_GPUs),name='JPEG')
    #         opt_Y = option.dict_to_nonedict(opt_Y)  # Convert to NoneDict, which return None for missing key.
    #         opt_Y['Y_model'] = True
    #         # opt_Y['train']['gan_weight'] = 0 # Prevent loading or creating of D for the Y channel
    #         # opt_Y['train']['resume'] = 0 # Loading a pre-trained model whose exact path is specified, not like loading a partially trained model by looking for the highest step num in a certain folder.
    #         opt_Y['path']['pretrain_model_G'] = opt['path']['Y_channel_model_G']
    #         model_Y = create_model(opt_Y)
    #     else:
    #         raise Exception('Unsupported currently')

    # create logger
    logger = Logger(opt)
    # Save validation set results as image collage:
    SAVE_IMAGE_COLLAGE = True
    # per_image_saved_patch = min([min(im['Uncomp'].shape[1:]) for im in val_loader.dataset])-2
    # num_val_images = len(val_loader.dataset)
    # val_images_collage_rows = int(np.floor(np.sqrt(num_val_images)))
    # while val_images_collage_rows>1:
    #     if np.round(num_val_images/val_images_collage_rows)==num_val_images/val_images_collage_rows:
    #         break
    #     val_images_collage_rows -= 1
    start_time = time.time()
    # min_accumulation_steps = min([opt['train']['grad_accumulation_steps_G'],opt['train']['grad_accumulation_steps_D']])
    save_GT_Uncomp = True
    lr_too_low = False
    print('---------- Start training -------------')
    last_saving_time = time.time()
    recently_saved_models = deque(maxlen=4)
    for epoch in range(int(math.floor(model.step / train_size)),total_epoches):
        for i, train_data in enumerate(train_loader):
            model.gradient_step_num = model.step // max_accumulation_steps
            not_within_batch = model.step % max_accumulation_steps == (max_accumulation_steps - 1)
            saving_step = ((time.time()-last_saving_time)>60*opt['logger']['save_checkpoint_freq']) and not_within_batch
            if saving_step:
                last_saving_time = time.time()

            # save models
            if lr_too_low or saving_step:
                model.save_log()
                recently_saved_models.append(model.save(model.gradient_step_num))
                if len(recently_saved_models)>3:
                    model_2_delete = recently_saved_models.popleft()
                    os.remove(model_2_delete)
                    if model.D_exists:
                        os.remove(model_2_delete.replace('_G.','_D.'))
                print('{}: Saving the model before iter {:d}.'.format(datetime.now().strftime('%H:%M:%S'),model.gradient_step_num))
                if lr_too_low:
                    break

            if model.step > total_iters:
                break

            time_elapsed = time.time() - start_time
            if not_within_batch:    start_time = time.time()
            # log
            if model.gradient_step_num % opt['logger']['print_freq'] == 0 and not_within_batch:
                logs = model.get_current_log()
                print_rlt = OrderedDict()
                print_rlt['model'] = opt['model']
                print_rlt['epoch'] = epoch
                print_rlt['iters'] = model.gradient_step_num
                print_rlt['time'] = time_elapsed
                for k, v in logs.items():
                    print_rlt[k] = v
                print_rlt['lr'] = model.get_current_learning_rate()
                logger.print_format_results('train', print_rlt,keys_ignore_list=['avg_est_err'])
                model.display_log_figure()

            # validation
            if (not_within_batch or i==0) and (model.gradient_step_num) % opt['train']['val_freq'] == 0: # and model.gradient_step_num>=opt['train']['D_init_iters']:
                print_rlt = OrderedDict()
                if model.generator_changed:
                    print('---------- validation -------------')
                    start_time = time.time()
                    if False and SAVE_IMAGE_COLLAGE and model.gradient_step_num%opt['train']['val_save_freq'] == 0: #Saving training images:
                        # GT_image_collage,quantized_image_collage = [],[]
                        cur_train_results = model.get_current_visuals(entire_batch=True)
                        train_psnrs = [util.calculate_psnr(util.tensor2img(cur_train_results['Decomp'][im_num], out_type=np.uint8,min_max=[0,255]),
                            util.tensor2img(cur_train_results['Uncomp'][im_num], out_type=np.uint8,min_max=[0,255])) for im_num in range(len(cur_train_results['Decomp']))]
                        #Save latest training batch output:
                        save_img_path = os.path.join(os.path.join(opt['path']['val_images']),
                                                     '{:d}_Tr_PSNR{:.3f}.png'.format(model.gradient_step_num, np.mean(train_psnrs)))
                        util.save_img(np.clip(np.concatenate((np.concatenate([util.tensor2img(cur_train_results['Uncomp'][im_num], out_type=np.uint8,min_max=[0,255]) for im_num in
                                 range(len(cur_train_results['Decomp']))],0), np.concatenate(
                                [util.tensor2img(cur_train_results['Decomp'][im_num], out_type=np.uint8,min_max=[0,255]) for im_num in range(len(cur_train_results['Decomp']))],
                                0)), 1), 0, 255).astype(np.uint8), save_img_path)
                    Z_latent = [0]+([-0.5,0.5] if opt['network_G']['latent_input'] else [])
                    print_rlt['psnr'] = 0
                    for cur_Z in Z_latent:
                        model.perform_validation(data_loader=val_loader,cur_Z=cur_Z,print_rlt=print_rlt,GT_and_quantized=save_GT_Uncomp,
                                                 save_images=((model.gradient_step_num) % opt['train']['val_save_freq'] == 0) or save_GT_Uncomp)
                    if save_GT_Uncomp:  # Save GT Uncomp images
                        save_GT_Uncomp = False
                    print_rlt['psnr'] /= len(Z_latent)
                    model.log_dict['psnr_val'].append((model.gradient_step_num,print_rlt['psnr']))
                else:
                    print('Skipping validation because generator is unchanged')
                time_elapsed = time.time() - start_time
                # Save to log
                print_rlt['model'] = opt['model']
                print_rlt['epoch'] = epoch
                print_rlt['iters'] = model.gradient_step_num
                print_rlt['time'] = time_elapsed
                # model.display_log_figure()
                # model.generator_changed = False
                logger.print_format_results('val', print_rlt,keys_ignore_list=['avg_est_err'])
                print('-----------------------------------')

            # training
            # if model.chroma_mode:
            #     data = copy.deepcopy(train_data)
            #     data['Uncomp'] = train_data['Uncomp'][:,0,...].unsqueeze(1)
            #     model_Y.feed_data(data)
            #     model_Y.test()
            #     model.y_channel_input = model_Y.Output_Batch(within_0_1=False)
            #     train_data['Uncomp'][:,0,...] = model.y_channel_input.squeeze(1)
            model.feed_data(train_data)
            model.optimize_parameters()


            # update learning rate
            if not_within_batch:
                lr_too_low = model.update_learning_rate(model.gradient_step_num)
            # current_step += 1
        if lr_too_low:
            print('Stopping training because LR is too low')
            break

    print('Saving the final model.')
    model.save(model.gradient_step_num)
    model.save_log()
    print('End of training.')


if __name__ == '__main__':
    # # OpenCV get stuck in transform when used in DataLoader
    # # https://github.com/pytorch/pytorch/issues/1838
    # # However, cause problem reading lmdb
    # import torch.multiprocessing as mp
    # mp.set_start_method('spawn', force=True)
    main()
