import os
import os.path
import sys
from multiprocessing import Pool
import numpy as np
import cv2
from socket import gethostname
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.progress_bar import ProgressBar


def main():
    """A multi-thread tool to crop sub imags."""
    dataset_root_path = '/home/ybahat/Datasets' if gethostname()=='ybahat-System-Product-Name' else '/home/tiras/datasets' if 'tiras' in os.getcwd() else '/media/ybahat/data/Datasets'
    input_folder = os.path.join(dataset_root_path,'DIV2K_train/DIV2K_train_HR')
    save_folder = os.path.join(dataset_root_path,'DIV2K_train/DIV2K_train_sub_HR')
    n_thread = 20
    crop_sz = 256#480
    step = 30#240
    thres_sz = 48
    compression_level = 3  # 3 is the default value in cv2
    multi_scale = False
    # CV_IMWRITE_PNG_COMPRESSION from 0 to 9. A higher value means a smaller size and longer
    # compression time. If read raw images during training, use 0 for faster IO speed.

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        print('mkdir [{:s}] ...'.format(save_folder))
    else:
        print('Folder [{:s}] already exists. Exit...'.format(save_folder))
        sys.exit(1)

    img_list = []
    for root, _, file_list in sorted(os.walk(input_folder)):
        path = [os.path.join(root, x) for x in file_list]  # assume only images in the input_folder
        img_list.extend(path)

    def update(arg):
        pbar.update(arg)

    pbar = ProgressBar(len(img_list))

    pool = Pool(n_thread)
    for path in img_list:
        pool.apply_async(worker,
            args=(path, save_folder, crop_sz, step, thres_sz, compression_level,multi_scale),
            callback=update)
    pool.close()
    pool.join()
    print('All subprocesses done.')


def worker(path, save_folder, crop_sz, step, thres_sz, compression_level,multi_scale=False):
    img_name = os.path.basename(path)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    n_channels = len(img.shape)
    if n_channels == 2:
        h, w = img.shape
    elif n_channels == 3:
        h, w, c = img.shape
    else:
        raise ValueError('Wrong image shape - {}'.format(n_channels))

    num_scales = int(np.floor(np.log2(min(h,w)/crop_sz))+1) if multi_scale else 1
    desired_steps_per_scale = (min(h, w) // (2 ** (num_scales - 1)) - crop_sz) // step
    step *= 2**(num_scales-1)

    for scale_num in range(num_scales):
        cur_scale_img = img if scale_num==0 else cv2.resize(img,(w//2,h//2))
        h,w = cur_scale_img.shape[:2]
        crop_sz = min(h,w)-(desired_steps_per_scale-1)*step
        h_space = np.arange(0, h - crop_sz + 1, step)
        if h - (h_space[-1] + crop_sz) > thres_sz:
            h_space = np.append(h_space, h - crop_sz)
        w_space = np.arange(0, w - crop_sz + 1, step)
        if w - (w_space[-1] + crop_sz) > thres_sz:
            w_space = np.append(w_space, w - crop_sz)

        index = 0
        for x in h_space:
            for y in w_space:
                index += 1
                if n_channels == 2:
                    crop_img = cur_scale_img[x:x + crop_sz, y:y + crop_sz]
                else:
                    crop_img = cur_scale_img[x:x + crop_sz, y:y + crop_sz, :]
                crop_img = np.ascontiguousarray(crop_img)
                # var = np.var(crop_img / 255)
                # if var > 0.008:
                #     print(img_name, index_str, var)
                cv2.imwrite(
                    os.path.join(save_folder, img_name.replace('.png', '_scale{:1d}_s{:03d}.png'.format(scale_num,index))),
                    crop_img, [cv2.IMWRITE_PNG_COMPRESSION, compression_level])
        step = step//2
    return 'Processing {:s} ...'.format(img_name)


if __name__ == '__main__':
    main()
