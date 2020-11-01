import os
from collections import OrderedDict
from datetime import datetime
import json
# from jsonpath_rw import parse as nested_key_parse
from socket import gethostname
import numpy as np
from deepdiff import DeepDiff
try:
    import GPUtil
except:
    pass

def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')

def print_config_change(dict,key):
    print('%s:' % (key))
    print('\tFrom: %s' % (dict[key]['old_value']))
    print('\tTo: %s' % (dict[key]['new_value']))

def parse(opt_path, is_train=True,batch_size_multiplier=None,name=None):
    OVERRIDING_KEYS = [['train','resume'],['datasets','train','n_workers'],['train','val_running_avg_steps']]
    opt = parse_conf(opt_path=opt_path,is_train=is_train,batch_size_multiplier=batch_size_multiplier,name=name)
    if is_train and opt['train']['resume']:
        saved_opt = parse_conf(opt_path=os.path.join(opt['path']['experiments_root'],'options.json'),is_train=is_train,batch_size_multiplier=batch_size_multiplier,name=name)
        for key in OVERRIDING_KEYS:
            cur_opt,cur_saved_opt = opt,saved_opt
            for sub_key in key[:-1]:
                cur_opt,cur_saved_opt = cur_opt[sub_key],cur_saved_opt[sub_key]
            cur_saved_opt[key[-1]] = cur_opt[key[-1]]
        saved_opt['train']['resume'] = opt['train']['resume']
        opt_diff = DeepDiff(opt,saved_opt)
        if len(opt_diff.keys())>0:
            print('Using some saved configuration values that are different from the current ones. This means changing:')
            if 'values_changed' in opt_diff.keys():
                [print_config_change(opt_diff['values_changed'],key) for key in opt_diff['values_changed']]
            if 'type_changes' in opt_diff.keys():
                [print_config_change(opt_diff['type_changes'],key) for key in opt_diff['type_changes']]
            if any([key not in ['values_changed','type_changes'] for key in opt_diff]):
                print('More configuration keys added or removed...')
        return saved_opt
    return opt

def parse_conf(opt_path, is_train=True,batch_size_multiplier=None,name=None):
    # remove comments starting with '//'
    json_str = ''
    with open(opt_path, 'r') as f:
        for line in f:
            line = line.split('//')[0] + '\n'
            json_str += line
    opt = json.loads(json_str, object_pairs_hook=OrderedDict)
    JPEG_run = name is not None and 'JPEG' in name
    if JPEG_run:
        if name=='JPEG_chroma':
            opt['input_downsampling'] = 2#Curenntly assuming downsampling with factor 2 of the chroma channels
            bare_name = opt['name'].split('/')[-1]
            if  bare_name[:len('chroma_')] != 'chroma_':
                opt['name'] = os.path.join('/'.join(opt['name'].split('/')[:-1]),'chroma_'+bare_name)
            for dataset in opt['datasets'].keys():
                if opt['datasets'][dataset]['mode'][-len('_chroma'):]!='_chroma':
                    opt['datasets'][dataset]['mode'] += '_chroma'
                opt['datasets'][dataset]['input_downsampling'] = opt['input_downsampling']
        else:
            opt['input_downsampling'] = 1
        if opt['name'][:len('JPEG/')]!='JPEG/': #Accomodating the case where the name was already modified, in which case I shouldn't add another 'JPEG/' prefix:
            opt['name'] = os.path.join('JPEG', opt['name'])
        opt['scale'] = 8*opt['input_downsampling']
    scale = opt['scale']
    opt['timestamp'] = get_timestamp()
    opt['is_train'] = is_train
    if 'datasets' in opt.keys():
        dataset_root_path =  Locally_Adapt_Path(opt['path']['datasets'] if 'datasets' in opt['path'].keys() else opt['path']['root'])
        # if 'root' not in opt['path']:
        #     opt['path']['root'] = '/media/ybahat/data/projects/SRGAN' if running_on_Technion else '/home/ybahat/PycharmProjects/SRGAN'
        non_degraded_images_fieldname = 'dataroot_Uncomp' if JPEG_run else 'dataroot_HR'
        for phase, dataset in opt['datasets'].items():
            phase = phase.split('_')[0]
            dataset['phase'] = phase
            dataset['scale'] = scale
            is_lmdb = False
            if non_degraded_images_fieldname in dataset and dataset[non_degraded_images_fieldname] is not None:
                dataset[non_degraded_images_fieldname] = os.path.expanduser(os.path.join(dataset_root_path,dataset[non_degraded_images_fieldname]))
                if dataset[non_degraded_images_fieldname].endswith('lmdb'):
                    is_lmdb = True
            if 'dataroot_LR' in dataset and dataset['dataroot_LR'] is not None:
                dataset['dataroot_LR'] = os.path.expanduser(os.path.join(dataset_root_path,dataset['dataroot_LR']))
                if dataset['dataroot_LR'].endswith('lmdb'):
                    is_lmdb = True
            dataset['data_type'] = 'lmdb' if is_lmdb else 'img'
            if 'train' in opt.keys() and any([field in opt['train'] for field in ['pixel_domain','feature_domain']]):
                assert opt['model'] in ['srragan','srgan'],'Unsupported'
            if phase == 'train' and 'subset_file' in dataset and dataset['subset_file'] is not None:
                dataset['subset_file'] = os.path.expanduser(dataset['subset_file'])

    # path
    for key, path in opt['path'].items():
        if path and key in opt['path']:
            opt['path'][key] = os.path.expanduser(path)
    if 'tiras' in os.getcwd():
        opt['path']['root'] = opt['path']['root'].replace('/media/ybahat/data/projects/', '/home/tiras/ybahat/')
    if not JPEG_run and name is not None:
        opt['name'] = os.path.join(name)
    experiments_root = os.path.join(opt['path']['root'], 'experiments', opt['name'])
    opt['path']['experiments_root'] = experiments_root
    opt['path']['models'] = os.path.join(experiments_root, 'models')
    if 'latent_input' not in opt['network_G'].keys():
        opt['network_G']['latent_input'] = 'None'
    if opt['network_G']['latent_input']=='None':
        opt['network_G']['latent_channels'] = 0
    opt['path']['log'] = experiments_root
    if is_train:
        opt['path']['val_images'] = os.path.join(experiments_root, 'val_images')
        # Legacy support:
        if 'batch_size_per_GPU' not in opt['datasets']['train'].keys():
            opt['datasets']['train']['batch_size_per_GPU'] = 1*opt['datasets']['train']['batch_size']
        if 'D_update_measure' not in opt['train'].keys():
            opt['train']['D_update_measure'] = 'post_train_D_diff'

        opt['datasets']['train']['batch_size'] = 1*opt['datasets']['train']['batch_size_per_GPU']
        if batch_size_multiplier is not None:
            opt['datasets']['train']['batch_size'] *= batch_size_multiplier
            opt['datasets']['train']['n_workers'] *= batch_size_multiplier
        if 'batch_size_4_grads_G' not in opt['datasets']['train'].keys():
            opt['datasets']['train']['batch_size_4_grads_G'] = 1*opt['datasets']['train']['batch_size']
            opt['datasets']['train']['batch_size_4_grads_D'] = 1*opt['datasets']['train']['batch_size']
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
        if 'network_D' in opt.keys():
            if opt['network_D']['which_model_D']=='PatchGAN':
                assert opt['train']['gan_type'] in ['lsgan','wgan-gp','wgan-sn','wgan-sngp']
            else:
                assert (opt['train']['gan_type']!='lsgan'),'lsgan GAN type should be used with Patch discriminator. For regular D, use vanilla type.'
    else:  # test
        results_root = os.path.join(opt['path']['root'], 'results', opt['name'])
        opt['path']['results_root'] = results_root
        # opt['path']['log'] = results_root

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

def Locally_Adapt_Path(org_path):
    path = org_path+''
    if 'tiras' in os.getcwd():
        path = '/home/tiras/datasets'
    elif gethostname()=='sipl-yuval.ef.technion.ac.il':
        path = '/media/ybahat/data/Datasets'
    return path