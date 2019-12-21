import os.path
import random
import numpy as np
import cv2
import torch
import torch.utils.data as data
import data.util as util
from DTE.imresize_DTE import imresize


class JpegDataset(data.Dataset):
    '''
    Read LR and HR image pairs.
    If only HR image is provided, generate LR image on-the-fly.
    The pair is ensured by 'sorted' function, so please check the name convention.
    '''

    def __init__(self, opt):
        super(JpegDataset, self).__init__()
        self.opt = opt
        self.paths_Uncomp = None
        self.Uncomp_env = None
        self.quality_factors = opt['jpeg_quality_factor']
        if not isinstance(self.quality_factors,list):
            self.quality_factors = [self.quality_factors]
        self.QF_probs = opt['QF_probs']
        if self.QF_probs is None:
            self.QF_probs = np.ones([len(self.quality_factors)])
        else:
            assert len(self.QF_probs)==len(self.quality_factors)
        self.QF_probs /= self.QF_probs.sum()

        # read image list from subset list txt
        if opt['subset_file'] is not None and opt['phase'] == 'train':
            with open(opt['subset_file']) as f:
                self.paths_Uncomp = sorted([os.path.join(opt['dataroot_Uncomp'], line.rstrip('\n')) \
                        for line in f])
            if opt['dataroot_LR'] is not None:
                raise NotImplementedError('Now subset only supports generating LR on-the-fly.')
        else:  # read image list from lmdb or image files
            self.Uncomp_env, self.paths_Uncomp = util.get_image_paths(opt['data_type'], opt['dataroot_Uncomp'])
        assert self.paths_Uncomp, 'Error: Uncomp path is empty.'

        if self.opt['phase'] == 'train':
            assert not self.opt['patch_size']%8,'Training for JPEG compression artifacts removal - Training images should have an integer number of 8x8 blocks.'
        else:
            # self.per_index_QF = np.round(np.linspace(start=self.quality_factors[0][0],stop=self.quality_factors[0][1]-1,num=len(self))).astype(int)
            if len(self.quality_factors)>=len(self):
                sampled_QFs = np.round(np.linspace(start=0,stop=len(self.quality_factors),num=len(self))).astype(int)
                per_range_len = [1 if (QF in sampled_QFs) else 0 for QF in self.quality_factors]
            else:
                num_exact_values = sum([not isinstance(QF,list) for QF in self.quality_factors])
                per_range_len = [((len(self)-num_exact_values)//(len(self.quality_factors)-num_exact_values)) if isinstance(QF,list) else 1 for QF in self.quality_factors]
                if any([isinstance(QF,list) for QF in self.quality_factors]):
                    per_range_len[np.argwhere([isinstance(QF,list) for QF in self.quality_factors])[0][0]] += len(self)-sum(per_range_len)
                else:
                    per_range_len[0] += len(self) - sum(per_range_len)
            self.per_index_QF = []
            for i,QF_range_len in enumerate(per_range_len):
                if isinstance(self.quality_factors[i], list):
                    self.per_index_QF += list(np.round(np.linspace(start=self.quality_factors[i][0],stop=self.quality_factors[i][1]-1,num=QF_range_len)).astype(int))
                else:
                    self.per_index_QF += [self.quality_factors[i]]*QF_range_len

        self.random_scale_list = [1]

    def __getitem__(self, index):
        scale = self.opt['scale']
        Uncomp_size = self.opt['patch_size']

        # get Uncomp image
        Uncomp_path = self.paths_Uncomp[index]
        img_Uncomp = 255*util.read_img(self.Uncomp_env, Uncomp_path)
        # modcrop in the validation / test phase
        if self.opt['phase'] != 'train':
            img_Uncomp = util.modcrop(img_Uncomp, 8)
        # change color space if necessary
        if self.opt['color']:
            img_Uncomp = util.channel_convert(img_Uncomp.shape[2], self.opt['color'], [img_Uncomp])[0]

        # randomly scale during training
        if self.opt['phase'] == 'train':
            QF = self.quality_factors[np.random.choice(len(self.quality_factors), p=self.QF_probs)]
            if isinstance(QF, list):
                QF = np.random.randint(low=QF[0],high=QF[1])
                # raise Exception('QF range is unsupported yet')

            random_scale = random.choice(self.random_scale_list)
            H_s, W_s, _ = img_Uncomp.shape

            def _mod(n, random_scale, scale, thres):
                rlt = int(n * random_scale)
                rlt = (rlt // scale) * scale
                return thres if rlt < thres else rlt

            H_s = _mod(H_s, random_scale, scale, Uncomp_size)
            W_s = _mod(W_s, random_scale, scale, Uncomp_size)
            img_Uncomp = cv2.resize(np.copy(img_Uncomp), (W_s, H_s), interpolation=cv2.INTER_LINEAR)
            # force to 3 channels
            if img_Uncomp.ndim == 2:
                img_Uncomp = np.expand_dims(img_Uncomp,-1)
                # img_Uncomp = cv2.cvtColor(img_Uncomp, cv2.COLOR_GRAY2BGR)
        else:
            QF = self.per_index_QF[index]

        H, W, _ = img_Uncomp.shape
        # # using DTE imresize:
        # img_LR = imresize(img_HR,scale_factor=[1/float(scale)],kernel=self.kernel)

        if self.opt['phase'] == 'train':
            # if the image size is too small
            H, W, _ = img_Uncomp.shape

            # randomly crop
            rnd_h_Uncomp = random.randint(0, max(0, H - Uncomp_size))
            rnd_w_Uncomp = random.randint(0, max(0, W - Uncomp_size))
            img_Uncomp = img_Uncomp[rnd_h_Uncomp:rnd_h_Uncomp + Uncomp_size, rnd_w_Uncomp:rnd_w_Uncomp + Uncomp_size, :]

            # augmentation - flip, rotate
            img_Uncomp = util.augment([img_Uncomp], self.opt['use_flip'], self.opt['use_rot'])
            img_Uncomp = img_Uncomp[0]

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_Uncomp.shape[2] == 3:
            img_Uncomp = img_Uncomp[:, :, [2, 1, 0]]
        img_Uncomp = torch.from_numpy(np.ascontiguousarray(np.transpose(img_Uncomp, (2, 0, 1)))).float()

        return {'Uncomp': img_Uncomp,  'Uncomp_path': Uncomp_path,'QF':QF}

    def __len__(self):
        return len(self.paths_Uncomp)
