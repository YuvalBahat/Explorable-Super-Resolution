import os
from collections import OrderedDict
from datetime import datetime
import json
from socket import gethostname
import numpy as np
try:
    import GPUtil
except:
    pass
import time


running_on_Technion = gethostname() in ['Yuval-Technion','tiras']
def Assign_GPU():
    excluded_IDs = [2]
    GPU_2_use = GPUtil.getAvailable(order='memory', excludeID=excluded_IDs)
    if len(GPU_2_use) == 0:
        print('No available GPUs. waiting...')
        while len(GPU_2_use) == 0:
            time.sleep(10)
            GPU_2_use = GPUtil.getAvailable(order='memory', excludeID=excluded_IDs)
    print('Using GPU #%d' % (GPU_2_use[0]))
    return GPU_2_use


def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')

# def Return_Field(opts,field):
#     if len(field)>1:
#         opts = Return_Field(opts,field[:-1])
#     return opts[field[-1]]

def parse(opt_path, is_train=True,batch_size_multiplier=None,name=None):
    # remove comments starting with '//'
    json_str = ''
    with open(opt_path, 'r') as f:
        for line in f:
            line = line.split('//')[0] + '\n'
            json_str += line
    opt = json.loads(json_str, object_pairs_hook=OrderedDict)

    opt['timestamp'] = get_timestamp()
    opt['is_train'] = is_train
    scale = opt['scale']
    dataset_root_path =  '/home/tiras/datasets' if 'tiras' in os.getcwd() else '/media/ybahat/data/Datasets' if running_on_Technion else '/home/ybahat/data/Databases'
    if 'root' not in opt['path']:
        opt['path']['root'] = '/media/ybahat/data/projects/SRGAN' if running_on_Technion else '/home/ybahat/PycharmProjects/SRGAN'
    # if running_on_Technion:
    #     opt['datasets']['train']['n_workers'] = 0
    # datasets
    non_degraded_images_fieldname = 'dataroot_Uncomp' if name=='JPEG' else 'dataroot_HR'
    for phase, dataset in opt['datasets'].items():
        phase = phase.split('_')[0]
        dataset['phase'] = phase
        dataset['scale'] = scale
        is_lmdb = False
        if non_degraded_images_fieldname in dataset and dataset[non_degraded_images_fieldname] is not None:
            dataset[non_degraded_images_fieldname] = os.path.expanduser(os.path.join(dataset_root_path,dataset[non_degraded_images_fieldname]))
            if dataset[non_degraded_images_fieldname].endswith('lmdb'):
                is_lmdb = True
        if 'dataroot_HR_bg' in dataset and dataset['dataroot_HR_bg'] is not None:
            dataset['dataroot_HR_bg'] = os.path.expanduser(os.path.join(dataset_root_path,dataset['dataroot_HR_bg']))
        if 'dataroot_LR' in dataset and dataset['dataroot_LR'] is not None:
            dataset['dataroot_LR'] = os.path.expanduser(os.path.join(dataset_root_path,dataset['dataroot_LR']))
            if dataset['dataroot_LR'].endswith('lmdb'):
                is_lmdb = True
        dataset['data_type'] = 'lmdb' if is_lmdb else 'img'
        if 'train' in opt.keys() and any([field in opt['train'] for field in ['pixel_domain','feature_domain']]):
            assert opt['model']=='srragan','Unsupported'
        if phase == 'train' and 'subset_file' in dataset and dataset['subset_file'] is not None:
            dataset['subset_file'] = os.path.expanduser(dataset['subset_file'])

    # path
    for key, path in opt['path'].items():
        if path and key in opt['path']:
            opt['path'][key] = os.path.expanduser(path)
    if 'tiras' in os.getcwd():
        opt['path']['root'] = opt['path']['root'].replace('/media/ybahat/data/projects/', '/home/tiras/ybahat/')
    if name is not None:
        opt['name'] = os.path.join(name,opt['name'])
    experiments_root = os.path.join(opt['path']['root'], 'experiments', opt['name'])
    opt['path']['experiments_root'] = experiments_root
    opt['path']['models'] = os.path.join(experiments_root, 'models')
    if opt['network_G']['latent_input']=='None':
        opt['network_G']['latent_channels'] = 0
    if is_train:
        opt['path']['log'] = experiments_root
        opt['path']['val_images'] = os.path.join(experiments_root, 'val_images')
        if batch_size_multiplier is not None:
            opt['datasets']['train']['batch_size'] *= batch_size_multiplier
            opt['datasets']['train']['n_workers'] *= batch_size_multiplier
        while np.mod(opt['datasets']['train']['batch_size_4_grads_G'],opt['datasets']['train']['batch_size'])!=0 or \
                np.mod(opt['datasets']['train']['batch_size_4_grads_D'], opt['datasets']['train']['batch_size']) != 0:
            opt['datasets']['train']['batch_size'] -= 1
        assert opt['datasets']['train']['batch_size']>0,'Batch size must be greater than 0'
        # assert opt['datasets']['train']['batch_size_4_grads_G']%opt['datasets']['train']['batch_size']==0,'Must have integer batches in a gradient step.'
        # assert opt['datasets']['train']['batch_size_4_grads_D']%opt['datasets']['train']['batch_size']==0,'Must have integer batches in a gradient step.'
        assert opt['datasets']['train']['batch_size_4_grads_D']>=opt['datasets']['train']['batch_size_4_grads_G'],'Currently not supporting G_batch>D_batch'
        opt['train']['grad_accumulation_steps_G'] = opt['datasets']['train']['batch_size_4_grads_G']//opt['datasets']['train']['batch_size']
        opt['train']['grad_accumulation_steps_D'] = opt['datasets']['train']['batch_size_4_grads_D']//opt['datasets']['train']['batch_size']
        # assert opt['network_G']['sigmoid_range_limit']==0 or opt['train']['range_weight'] ==0,'Reconsider using range penalty when using tanh range limiting of high frequencies'
        if opt['network_D']['which_model_D']=='PatchGAN':
            assert opt['train']['gan_type'] in ['lsgan','wgan-gp']
        else:
            assert (opt['train']['gan_type']!='lsgan'),'lsgan GAN type should be used with Patch discriminator. For regular D, use vanilla type.'
    else:  # test
        results_root = os.path.join(opt['path']['root'], 'results', opt['name'])
        opt['path']['results_root'] = results_root
        opt['path']['log'] = results_root

    # network
    opt['network_G']['scale'] = scale

    return opt

def save(opt):
    dump_dir = opt['path']['experiments_root'] if opt['is_train'] else opt['path']['results_root']
    dump_path = os.path.join(dump_dir, 'options.json')
    with open(dump_path, 'w') as dump_file:
        json.dump(opt, dump_file, indent=2)


class NoneDict(dict):
    def __missing__(self, key):
        return None


# convert to NoneDict, which return None for missing key.
def dict_to_nonedict(opt):
    if isinstance(opt, dict):
        new_opt = dict()
        for key, sub_opt in opt.items():
            new_opt[key] = dict_to_nonedict(sub_opt)
        return NoneDict(**new_opt)
    elif isinstance(opt, list):
        return [dict_to_nonedict(sub_opt) for sub_opt in opt]
    else:
        return opt
