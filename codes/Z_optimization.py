import torch
import numpy as np
from models.modules.loss import GANLoss,FilterLoss
from skimage.color import rgb2hsv,hsv2rgb
from scipy.signal import convolve2d
import time
from scipy.ndimage.morphology import binary_opening
from sklearn.feature_extraction.image import extract_patches_2d


class Optimizable_Temperature(torch.nn.Module):
    def __init__(self,initial_temperature=None):
        super(Optimizable_Temperature,self).__init__()
        self.log_temperature = torch.nn.Parameter(data=torch.zeros([1]).type(torch.cuda.DoubleTensor))
        if initial_temperature is not None:
            self.log_temperature.data = torch.log(torch.tensor(initial_temperature).type(torch.cuda.DoubleTensor))

    def forward(self):
        return torch.exp(self.log_temperature)

class SoftHistogramLoss(torch.nn.Module):
    def __init__(self,bins,min,max,desired_hist_image_mask=None,desired_hist_image=None,gray_scale=True,input_im_HR_mask=None,patch_size=1,automatic_temperature=False,
            image_Z=None,temperature=0.05,dictionary_not_histogram=False,no_patch_DC=False,no_patch_STD=False):
        self.temperature = temperature#0.05**2#0.006**6
        self.exp_power = 2#6
        self.SQRT_EPSILON = 1e-7
        super(SoftHistogramLoss,self).__init__()
        # min correspond to the CENTER of the first bin, and max to the CENTER of the last bin
        self.device = torch.device('cuda')
        self.bin_width = (max-min)/(bins-1)
        self.max = max
        self.no_patch_DC = no_patch_DC
        self.no_patch_STD = no_patch_STD
        assert no_patch_DC or not no_patch_STD,'Not supporting removing of only patch STD without DC'
        self.temperature_optimizer = automatic_temperature
        if automatic_temperature:
            self.optimizable_temperature = Optimizable_Temperature(self.temperature)
            self.image_Z = image_Z
        else:
            self.temperature = torch.tensor(self.temperature).type(torch.cuda.DoubleTensor)
            # self.temperature = self.temperature*self.bin_width*85 # 0.006 was calculated for pixels range [0,1] with 85 bins. So I adjust it acording to actual range and n_bins, manifested in self.bin_width
        self.bin_centers = torch.linspace(min,max,bins)
        self.gray_scale = gray_scale
        self.patch_size = patch_size
        self.num_dims = 3
        self.KDE = not gray_scale or patch_size>1 # Using Kernel Density Estimation rather than histogram
        if gray_scale:
            self.num_dims = self.num_dims//3
            self.bins = 1. * self.bin_centers.view([1] + list(self.bin_centers.size())).type(torch.cuda.DoubleTensor)
            if desired_hist_image is not None:
                desired_hist_image = [hist_im.mean(1, keepdim=True).view([-1,1]) for hist_im in desired_hist_image]
        if patch_size>1:
            assert gray_scale and (desired_hist_image is not None),'Not supporting color images or patch histograms for model training loss for now'
            self.num_dims = patch_size**2
            DESIRED_HIST_PATCHES_OVERLAP = (self.num_dims-patch_size)/self.num_dims # Patches overlap should correspond to entire patch but one row/column.
            desired_im_patch_extraction_mat = [ReturnPatchExtractionMat(hist_im_mask,patch_size=patch_size,device=self.device,
                patches_overlap=DESIRED_HIST_PATCHES_OVERLAP) for hist_im_mask in desired_hist_image_mask]
            desired_hist_image = [torch.sparse.mm(desired_im_patch_extraction_mat[i],desired_hist_image[i]).view([self.num_dims,-1,1]) for i in range(len(desired_hist_image))]
            # desired_hist_image = [self.Desired_Im_2_Bins(hist_im,prune_only=True) for hist_im in desired_hist_image]
            desired_hist_image = torch.cat(desired_hist_image,1)
            if self.no_patch_DC:
                desired_hist_image = desired_hist_image-torch.mean(desired_hist_image,dim=0,keepdim=True)
                if self.no_patch_STD:
                    self.mean_patches_STD = torch.max(torch.std(desired_hist_image, dim=0, keepdim=True),other=torch.tensor(1/255).to(self.device))
                    desired_hist_image = (desired_hist_image/self.mean_patches_STD)
                    self.mean_patches_STD = 1*self.mean_patches_STD.mean().item()
                    desired_hist_image = desired_hist_image*self.mean_patches_STD#I do that to preserve the original (pre-STD normalization) dynamic range, to avoid changing the kernel support size.
            self.desired_hist_image_mask = None
        else:
            if len(desired_hist_image)>1:   print('Not supproting multiple hist image versions for non-patch histogram/dictionary. Removing extra image versions.')
            desired_hist_image,desired_hist_image_mask = desired_hist_image[0],desired_hist_image_mask[0]
            self.desired_hist_image_mask = torch.from_numpy(desired_hist_image_mask).to(self.device).view([-1]).type(torch.ByteTensor) if desired_hist_image_mask is not None else None
            if desired_hist_image is not None:
                desired_hist_image = 1 * desired_hist_image.view([self.num_dims, -1, 1])
        if self.KDE:
            if self.desired_hist_image_mask is not None:
                desired_hist_image = desired_hist_image[:,self.desired_hist_image_mask,:]
            # The bins are now simply the multi-dimensional pixels/patches. So now I remove redundant bins, by checking if there is duplicacy:
            # if patch_size==1:#Otherwise I already did this step before for each image version, and I avoid repeating this pruning for the entire patches collection for memory limitation reasons.
            self.bins = self.Desired_Im_2_Bins(desired_hist_image)
        if not dictionary_not_histogram:
            self.loss = torch.nn.KLDivLoss()
        if patch_size>1:
            self.patch_extraction_mat = ReturnPatchExtractionMat(input_im_HR_mask.data.cpu().numpy(),patch_size=patch_size,device=self.device,patches_overlap=0.5)#.to(self.device)
            self.image_mask = None
        else:
            self.image_mask = input_im_HR_mask.view([-1]).type(torch.ByteTensor) if input_im_HR_mask is not None else None
        self.dictionary_not_histogram = dictionary_not_histogram
        if not dictionary_not_histogram:
            if not automatic_temperature and desired_hist_image is not None:
                with torch.no_grad():
                    self.desired_hists_list = [self.ComputeSoftHistogram(desired_hist_image,image_mask=self.desired_hist_image_mask,return_log_hist=False,
                                                                  reshape_image=False,compute_hist_normalizer=True).detach()]
            else:
                self.desired_hist_image = desired_hist_image

    def Feed_Desired_Hist_Im(self,desired_hist_image):
        self.desired_hists_list = []
        for desired_im in desired_hist_image:
            if self.gray_scale:
                desired_im = desired_im.mean(0, keepdim=True).view([1,-1, 1])
            with torch.no_grad():
                self.desired_hists_list.append(self.ComputeSoftHistogram(desired_im,image_mask=self.desired_hist_image_mask,return_log_hist=False,
                                                                  reshape_image=False,compute_hist_normalizer=True).detach())

    def Desired_Im_2_Bins(self,desired_im):
        image_2_big = True
        num_sub_images = 1
        while image_2_big:
            try:
                bins = []
                sub_image_sizes = [desired_im.size(1)//num_sub_images]*(num_sub_images-1)
                sub_image_sizes += ([desired_im.size(1)-sum(sub_image_sizes)] if desired_im.size(1)-sum(sub_image_sizes)>0 else [])
                sub_images = torch.split(desired_im,sub_image_sizes,dim=1)
                for im in sub_images:
                    repeated_elements_mat = (im.view([self.num_dims, -1, 1]) - im.view([im.size(0)] + [1, -1])).abs()
                    repeated_elements_mat = (repeated_elements_mat < self.bin_width / 2).all(0)
                    repeated_elements_mat = torch.mul(repeated_elements_mat,(1 - torch.diag(torch.ones([repeated_elements_mat.size(0)]))).type(
                                                          repeated_elements_mat.dtype).to(repeated_elements_mat.device))
                    repeated_elements_mat = torch.triu(repeated_elements_mat).any(1) ^ 1
                    bins.append(im[:, repeated_elements_mat])
                del repeated_elements_mat
                image_2_big = False
            except:
                num_sub_images += 1
                print('Hist bin pruning failed, retrying with %d sub-images' % (num_sub_images))
        # if prune_only:
        #     return bins
        bins = [b.view([desired_im.size(0), 1, -1]).type(torch.cuda.DoubleTensor) for b in bins]
        return torch.cat(bins,-1)

    def TemperatureSearch(self,desired_image,initial_image,desired_KL_div):
        log_temperature_range = [0.1,1]
        STEP_SIZE = 10
        KL_DIV_TOLERANCE = 0.1
        cur_KL_div = []
        desired_temp_within_range = False
        with torch.no_grad():
            while True:
                next_temperature = np.exp(np.mean(log_temperature_range))
                if np.isinf(next_temperature) or next_temperature==0:
                    print('KL div. is %.3e even for temperature of %.3e, aborting temperature search with that.'%(cur_KL_div[-1],self.temperature))
                    break
                self.temperature = 1*next_temperature
                desired_im_hist = self.ComputeSoftHistogram(desired_image,image_mask=self.desired_hist_image_mask,return_log_hist=False,reshape_image=False,compute_hist_normalizer=True)
                initial_image_hist = self.ComputeSoftHistogram(initial_image,image_mask=self.image_mask,return_log_hist=True,reshape_image=False,compute_hist_normalizer=False)
                cur_KL_div.append(self.loss(initial_image_hist,desired_im_hist).item())
                KL_div_too_big = cur_KL_div[-1] > desired_KL_div
                if np.abs(np.log(max([0,cur_KL_div[-1]])/desired_KL_div))<=np.log(1+KL_DIV_TOLERANCE):
                    print('Automatically set histogram temperature to %.3e'%(self.temperature))
                    break
                elif not desired_temp_within_range:
                    if len(cur_KL_div)==1:
                        initial_KL_div_too_big = KL_div_too_big
                    else:
                        desired_temp_within_range = initial_KL_div_too_big^KL_div_too_big
                    if not desired_temp_within_range:
                        if KL_div_too_big:
                            log_temperature_range[1] += STEP_SIZE
                        else:
                            log_temperature_range[0] -= STEP_SIZE
                if desired_temp_within_range:
                    if KL_div_too_big:
                        log_temperature_range[0] = 1*np.log(self.temperature)
                    else:
                        log_temperature_range[1] = 1*np.log(self.temperature)

    def ComputeSoftHistogram(self,image,image_mask,return_log_hist,reshape_image,compute_hist_normalizer,temperature=None):
        CANONICAL_KDE_4_DICTIONARY = True
        if temperature is None:
            temperature = 1*self.temperature
        if not reshape_image:
            image = image.type(torch.cuda.DoubleTensor)
        else:
            if self.patch_size > 1:
                image = torch.sparse.mm(self.patch_extraction_mat, image.view([-1, 1])).view([self.num_dims, -1])
                if self.no_patch_DC:
                    image = image-torch.mean(image,dim=0,keepdim=True)
                    if self.no_patch_STD:
                        image = image / torch.max(torch.std(image, dim=0, keepdim=True), other=torch.tensor(1 / 255).to(self.device))*self.mean_patches_STD
            else:
                image = image.contiguous().view([self.num_dims,-1])
                if image_mask is not None:
                    image = image[:, image_mask]
            image = image.unsqueeze(-1).type(torch.cuda.DoubleTensor)
        hist = (image-self.bins).abs()
        hist = torch.min(hist,(image-self.bins-self.max).abs())
        hist = torch.min(hist,(image-self.bins+self.max).abs())
        if not self.dictionary_not_histogram or CANONICAL_KDE_4_DICTIONARY:
            hist = -((hist+self.SQRT_EPSILON)**self.exp_power)/temperature
        hist = hist.mean(0)
        if self.dictionary_not_histogram and not CANONICAL_KDE_4_DICTIONARY:
            # return torch.exp(self.bin_width/(hist+self.bin_width/2))
            return hist.min(dim=1)[0].view([1,-1])
            # return hist.min(dim=1)[0].view([1, -1])
        if self.dictionary_not_histogram and CANONICAL_KDE_4_DICTIONARY:
            return -1*torch.log(torch.exp(hist).mean(1)).view([1, -1])
        hist = torch.exp(hist).mean(0)
        if compute_hist_normalizer or not self.KDE:
            self.normalizer = hist.sum()/image.size(1)
        hist = (hist/self.normalizer/image.size(1)).type(torch.cuda.FloatTensor)
        if self.KDE: # Adding another "bin" to account for all other missing bins
            hist = torch.cat([hist,(1-torch.min(torch.tensor(1).type(hist.dtype).to(hist.device),hist.sum())).view([1])])
        if return_log_hist:
            return torch.log(hist+torch.finfo(hist.dtype).eps).view([1,-1])
        else:
            return hist.view([1,-1])

    def forward(self,cur_images):
        cur_images_hists,KLdiv_grad_sizes = [],[]
        for i,cur_image in enumerate(cur_images):
            if self.gray_scale:
                cur_image = cur_image.mean(0, keepdim=True)
            if self.temperature_optimizer:
                self.temperature = self.optimizable_temperature()
                self.desired_hists_list.append(self.ComputeSoftHistogram(self.desired_hist_image, image_mask=self.desired_hist_image_mask,return_log_hist=False,
                                                              reshape_image=False, compute_hist_normalizer=True))
            else:
                temperature = self.temperature*(1 if (len(cur_images)==1 or True) else 5**(i-1)) #I used to multiply temperature for multi-scale histogram - I'm not sure why I did that, and I cancel it now since I use multiple images for the random initializations of the z optimization.
            cur_images_hists.append(self.ComputeSoftHistogram(cur_image, self.image_mask, return_log_hist=True,reshape_image=True, compute_hist_normalizer=False,temperature=temperature))
            if self.temperature_optimizer:
                KLdiv_grad_sizes.append(-1*(torch.autograd.grad(outputs=self.loss(cur_images_hists[-1],self.desired_hists_list[-1]),inputs=self.image_Z,create_graph=True)[0]).norm(p=2))
        if self.temperature_optimizer:
            return self.loss(torch.cat(cur_images_hists,0),torch.cat(self.desired_hists_list,0)),torch.stack(KLdiv_grad_sizes).mean()
        elif self.dictionary_not_histogram:
            return torch.cat(cur_images_hists,0).mean(1).type(torch.cuda.FloatTensor)
        else:
            return self.loss(torch.cat(cur_images_hists,0),torch.cat(self.desired_hists_list,0)).type(torch.cuda.FloatTensor)

def ReturnPatchExtractionMat(mask,patch_size,device,patches_overlap=1,return_non_covered=False):
    RANDOM_PATCHES_SELECTION = False #If true, patches are dropped in a random order, satisfying the maximal overlap constraint, rather than moving columns first ,than rows. This typically discards of much more patches.
    mask = binary_opening(mask, np.ones([patch_size, patch_size]).astype(np.bool))
    patches_indexes = extract_patches_2d(np.multiply(mask, 1 + np.arange(mask.size).reshape(mask.shape)),
                                         (patch_size, patch_size)).reshape([-1, patch_size**2])
    patches_indexes = patches_indexes[np.all(patches_indexes > 0, 1), :] - 1
    if patches_overlap<1:
        # I discard patches by discarding those containing too many pixels that are already covered by a previous patch. Patches are ordered right to left, top to bottom.
        # For exampe, if the stride corresponds to one row/column, it would be one row. There might be simpler ways to achieve this...
        unique_indexes = list(set(list(patches_indexes.reshape([-1]))))
        min_index = min(unique_indexes)
        index_taken_indicator = np.zeros([max(unique_indexes) - min(unique_indexes)]).astype(np.bool)
        valid_patches = np.ones([patches_indexes.shape[0]]).astype(np.bool)
        randomized_patches_indexes = np.random.permutation(patches_indexes.shape[0])
        oredered_patches_indexes = randomized_patches_indexes if RANDOM_PATCHES_SELECTION else np.arange(patches_indexes.shape[0])
        for patch_num in oredered_patches_indexes:
        # for patch_num, patch in enumerate(patches_indexes):
            if (patches_overlap==0 and np.any(index_taken_indicator[patches_indexes[patch_num,:] - min_index - 1]))\
                    or np.mean(index_taken_indicator[patches_indexes[patch_num,:] - min_index - 1])>patches_overlap:
                valid_patches[patch_num] = False
                continue
            index_taken_indicator[patches_indexes[patch_num,:] - min_index - 1] = True
        patches_indexes = patches_indexes[valid_patches]
        print('%.3f of desired pixels are covered by assigned patches'%(index_taken_indicator[unique_indexes-min_index-1].mean()))
        if return_non_covered:
            non_covered_indexes = np.array(unique_indexes)
            non_covered_indexes = non_covered_indexes[np.logical_not(index_taken_indicator[non_covered_indexes - min_index - 1])]
            non_covered_pixels_extraction_mat = Patch_Indexes_2_Sparse_Mat(non_covered_indexes,mask.size,device)
    patch_extraction_mat = Patch_Indexes_2_Sparse_Mat(patches_indexes,mask.size,device)
    if return_non_covered:
        if not patches_overlap<1:
            non_covered_pixels_extraction_mat = None#torch.sparse.FloatTensor(torch.Size([0, mask.size]))
        return patch_extraction_mat,non_covered_pixels_extraction_mat
    else:
        return patch_extraction_mat

def Patch_Indexes_2_Sparse_Mat(patches_indexes,mask_size,device):
    corresponding_mat_rows = np.arange(patches_indexes.size).reshape([-1])
    return torch.sparse.FloatTensor(
        torch.LongTensor([corresponding_mat_rows, patches_indexes.transpose().reshape([-1])]),
        torch.FloatTensor(np.ones([corresponding_mat_rows.size])), torch.Size([patches_indexes.size, mask_size])).to(device)

class Optimizable_Z(torch.nn.Module):
    def __init__(self,Z_shape,Z_range=None,initial_pre_tanh_Z=None,Z_mask=None,random_perturbations=False):
        super(Optimizable_Z, self).__init__()
        # self.device = torch.device('cuda')
        self.Z = torch.nn.Parameter(data=torch.zeros(Z_shape).type(torch.cuda.FloatTensor))
        if Z_mask is not None and not np.all(Z_mask):
            self.mask = torch.from_numpy(Z_mask).type(torch.cuda.FloatTensor).to(self.Z.data.device)
            self.initial_pre_tanh_Z = 1*initial_pre_tanh_Z.type(torch.cuda.FloatTensor).to(self.Z.data.device)
        else:
            self.mask = None
        if initial_pre_tanh_Z is not None:
            assert initial_pre_tanh_Z.size()[1:]==self.Z.data.size()[1:] and (initial_pre_tanh_Z.size(0) in [1,self.Z.data.size(0)]),'Initilizer size does not match desired Z size'
            if random_perturbations:
                initial_pre_tanh_Z += torch.normal(mean=torch.zeros_like(initial_pre_tanh_Z), std=0.001 * torch.ones_like(initial_pre_tanh_Z))
            self.Z.data[:initial_pre_tanh_Z.size(0),...] = initial_pre_tanh_Z.to(self.Z.data.device)
        self.Z_range = Z_range
        if Z_range is not None:
            self.tanh = torch.nn.Tanh()


    def forward(self):
        if self.Z_range is not None:
            self.Z.data = torch.min(torch.max(self.Z,torch.tensor(-torch.finfo(self.Z.dtype).max).type(self.Z.dtype).to(self.Z.device)),torch.tensor(torch.finfo(self.Z.dtype).max).type(self.Z.dtype).to(self.Z.device))
        if self.mask is not None:
            self.Z.data = self.mask * self.Z.data + (1 - self.mask) * self.initial_pre_tanh_Z
        if self.Z_range is not None:
            return self.Z_range*self.tanh(self.Z)
        else:
            return self.Z

    def PreTanhZ(self):
        if self.mask is not None:
            return self.mask * self.Z.data + (1 - self.mask) * self.initial_pre_tanh_Z
        else:
            return self.Z.data

    def Randomize_Z(self,what_2_shuffle):
        assert what_2_shuffle in ['all','allButFirst']
        if what_2_shuffle=='all':
            torch.nn.init.xavier_uniform_(self.Z.data,gain=100)
        else:
            torch.nn.init.xavier_uniform_(self.Z.data[1:], gain=100)
            # self.Z.data[1] = 1 * self.Z.data[3]

    def Return_Detached_Z(self):
        return self.forward().detach()

def ArcTanH(input_tensor):
    return 0.5*torch.log((1+input_tensor+torch.finfo(input_tensor.dtype).eps)/(1-input_tensor+torch.finfo(input_tensor.dtype).eps))

def TV_Loss(image):
    # return torch.pow((image[:,:,:,:-1]-image[:,:,:,1:]).abs(),0.1).mean(dim=(1,2,3))+torch.pow((image[:,:,:-1,:]-image[:,:,1:,:]).abs(),0.1).mean(dim=(1,2,3))
    return (image[:,:,:,:-1]-image[:,:,:,1:]).abs().mean(dim=(1,2,3))+(image[:,:,:-1,:]-image[:,:,1:,:]).abs().mean(dim=(1,2,3))

class Z_optimizer():
    MIN_LR = 1e-5
    PATCH_SIZE_4_STD = 7
    def __init__(self,objective,Z_size,model,Z_range,max_iters,data=None,loggers=None,image_mask=None,Z_mask=None,initial_Z=None,initial_LR=None,existing_optimizer=None,
                 batch_size=1,HR_unpadder=None,auto_set_hist_temperature=False,random_Z_inits=False,jpeg_extractor=None):
        self.data_keys = {'reconstructed':'SR','GT':'HR'} if jpeg_extractor is None else {'reconstructed':'Decomp','GT':'Uncomp'}
        if (initial_Z is not None or 'cur_Z' in model.__dict__.keys()):
            if initial_Z is None:
                initial_Z = 1*model.GetLatent()
            initial_pre_tanh_Z = initial_Z/Z_range
            initial_pre_tanh_Z = torch.clamp(initial_pre_tanh_Z,min=-1+torch.finfo(initial_pre_tanh_Z.dtype).eps,max=1.-torch.finfo(initial_pre_tanh_Z.dtype).eps)
            initial_pre_tanh_Z = ArcTanH(initial_pre_tanh_Z)

        else:
            initial_pre_tanh_Z = None
        self.Z_model = Optimizable_Z(Z_shape=[batch_size,model.num_latent_channels] + list(Z_size), Z_range=Z_range,initial_pre_tanh_Z=initial_pre_tanh_Z,Z_mask=Z_mask,
            random_perturbations=(random_Z_inits and 'random' not in objective) or ('random' in objective and 'limited' in objective))
        assert (initial_LR is not None) or (existing_optimizer is not None),'Should either supply optimizer from previous iterations or initial LR for new optimizer'
        self.objective = objective
        self.data = data
        self.device = torch.device('cuda')
        self.jpeg_extractor = jpeg_extractor
        self.model = model
        self.model_training = HR_unpadder is not None
        if image_mask is None:
            if 'fake_H' in model.__dict__.keys():
                self.image_mask = torch.ones(list(model.fake_H.size()[2:])).type(model.fake_H.dtype).to(self.device)
            else:
                self.image_mask = None
            self.Z_mask = None#torch.ones(Z_size).type(model.fake_H.dtype).to(self.device)
        else:
            assert Z_mask is not None,'Should either supply both masks or niether'
            self.image_mask = torch.from_numpy(image_mask).type(model.fake_H.dtype).to(self.device)
            self.Z_mask = torch.from_numpy(Z_mask).type(model.fake_H.dtype).to(self.device)
            self.initial_Z = 1.*model.GetLatent()
            self.image_mask.requires_grad = False
            self.Z_mask.requires_grad = False
        if 'local' in objective:#Used in relative STD change and periodicity objective cases:
            desired_overlap = 1 if 'STD' in objective else 0.5
            self.patch_extraction_map,self.non_covered_indexes_extraction_mat = ReturnPatchExtractionMat(mask=image_mask,
                patch_size=self.PATCH_SIZE_4_STD,device=model.fake_H.device,patches_overlap=desired_overlap,return_non_covered=True)
            # self.patch_extraction_map, self.non_covered_indexes_extraction_mat =\
            #     self.patch_extraction_map.to(model.fake_H.device),self.non_covered_indexes_extraction_mat.to(model.fake_H.device)
        if not self.model_training:
            self.initial_STD = self.Masked_STD(first_image_only=True)
            print('Initial STD: %.3e' % (self.initial_STD.mean().item()))
        if existing_optimizer is None:
            if any([phrase in objective for phrase in ['l1','scribble']]) and 'random' not in objective:
                if data is not None and self.data_keys['GT'] in data.keys():
                    self.GT_HR = data[self.data_keys['GT']]
                if self.image_mask is None:
                    self.loss = torch.nn.L1Loss().to(torch.device('cuda'))
                else:
                    loss_mask = self.image_mask
                    SMOOTHING_MARGIN = 1
                    if 'scribble' in objective:
                        scribble_mask_tensor = torch.from_numpy(data['scribble_mask']).type(loss_mask.dtype).to(loss_mask.device)
                        scribble_multiplier = np.ones_like(data['scribble_mask']).astype(np.float32)
                        scribble_multiplier += data['brightness_factor']*(data['scribble_mask']==2)-data['brightness_factor']*(data['scribble_mask']==3)
                        if SMOOTHING_MARGIN>0:
                            scribble_multiplier = convolve2d(np.pad(scribble_multiplier,((SMOOTHING_MARGIN,SMOOTHING_MARGIN),(SMOOTHING_MARGIN,SMOOTHING_MARGIN)),mode='edge'),
                                                             np.ones([SMOOTHING_MARGIN*2+1,SMOOTHING_MARGIN*2+1])/((SMOOTHING_MARGIN*2+1)**2),mode='valid')
                        L1_loss_mask = loss_mask*((scribble_mask_tensor>0)*(scribble_mask_tensor<4)).float()
                        TV_loss_masks = [loss_mask*(scribble_mask_tensor==id).float().unsqueeze(0).unsqueeze(0) for id in torch.unique(scribble_mask_tensor*loss_mask) if id>3]
                        cur_HSV = rgb2hsv(np.clip(255*self.model.fake_H[0].data.cpu().numpy().transpose((1,2,0)).copy(),0,255))
                        cur_HSV[:,:,2] = cur_HSV[:,:,2]* scribble_multiplier
                        desired_RGB = hsv2rgb(cur_HSV)
                        desired_RGB = np.expand_dims(desired_RGB.transpose((2,0,1)),0)/255
                        desired_RGB_mask = (scribble_mask_tensor==2)+(scribble_mask_tensor==3)
                        self.GT_HR = self.GT_HR*(1-desired_RGB_mask).float()+desired_RGB_mask.float()*torch.from_numpy(desired_RGB).type(loss_mask.dtype).to(loss_mask.device)
                    def Scribble_Loss(produced_im,GT_im):
                        loss_per_im = []
                        for im_num in range(produced_im.size(0)):
                            loss_per_im.append(torch.nn.functional.l1_loss(input=produced_im[im_num].unsqueeze(0) * L1_loss_mask.to(self.device),
                                                            target=GT_im * L1_loss_mask.to(self.device)).to(torch.device('cuda')))
                            # if torch.any(TV_loss_mask.type(torch.uint8)):
                            if len(TV_loss_masks)>0:
                                loss_per_im[-1] = loss_per_im[-1] + Scribble_TV_Loss(produced_im[im_num].unsqueeze(0))
                        return torch.stack(loss_per_im,0)

                    def Scribble_TV_Loss(produced_im):
                        loss = 0
                        for TV_loss_mask in TV_loss_masks:
                            for y_shift in [-1,0,1]: # Taking differences to 8 neighbors, but calculating only 4 differences for each point (3 y shifts * 2 x shifts minus 2 discarded), to avoid duplicate differences
                                for x_shift in [-1,0]:
                                    if y_shift in [0,1] and x_shift==0:
                                        continue
                                    point = np.array([y_shift,x_shift])
                                    cur_mask = self.Return_Translated_SubImage(TV_loss_mask,point) * self.Return_Translated_SubImage(TV_loss_mask, -point)
                                    loss = loss + (cur_mask * (self.Return_Translated_SubImage(produced_im,point) - self.Return_Translated_SubImage(produced_im, -point)).abs()).mean(dim=(1, 2, 3))
                        return loss

                    self.loss = Scribble_Loss
                # scheduler_threshold = 1e-2
            elif 'Mag' in objective:
                self.desired_patches = torch.sparse.mm(self.patch_extraction_map, self.model.fake_H.mean(dim=1).view([-1, 1])).view([self.PATCH_SIZE_4_STD ** 2, -1])
                desired_STD = torch.max(torch.std(self.desired_patches,dim=0,keepdim=True),torch.tensor(1/255).to(self.device))
                self.desired_patches = (self.desired_patches-torch.mean(self.desired_patches,dim=0,keepdim=True))/desired_STD*\
                    (desired_STD+data['STD_increment']*(1 if 'increase' in objective else -1))+torch.mean(self.desired_patches,dim=0,keepdim=True)
            elif 'desired_SVD' in objective:
                self.loss = FilterLoss(latent_channels='SVDinNormedOut_structure_tensor',constant_Z=data['desired_Z'],
                                       reference_images={'min':data['reference_image_min'],'max':data['reference_image_max']},masks={'LR':self.Z_mask,'HR':self.image_mask})
            elif 'STD' in objective and not any([phrase in objective for phrase in ['periodicity','TV','dict','hist']]):
                assert self.objective.replace('local_','') in ['max_STD', 'min_STD','STD_increase','STD_decrease']
                if any([phrase in objective for phrase in ['increase','decrease']]):
                    STD_CHANGE_FACTOR = 1.05
                    STD_CHANGE_INCREMENT = data['STD_increment']
                    self.desired_STD = self.initial_STD
                    if STD_CHANGE_INCREMENT is None:#Using multiplicative desired STD factor:
                        self.desired_STD *= STD_CHANGE_FACTOR if 'increase' in objective else 1/STD_CHANGE_FACTOR
                    else:#Using an additive increment:
                        self.desired_STD += STD_CHANGE_INCREMENT if 'increase' in objective else -STD_CHANGE_INCREMENT
            elif 'periodicity' in objective:
                self.STD_PRESERVING_WEIGHT = 20#0.2 if 'Plus' in objective else 20
                self.PLUS_MEANS_STD_INCREASE = True
                if 'nonInt' in objective:
                    image_size = list(self.model.fake_H.size()[2:])
                    self.periodicity_points,self.half_period_points = [],[]
                    if 'Plus' in objective and self.PLUS_MEANS_STD_INCREASE:
                        self.desired_STD = self.initial_STD + data['STD_increment']
                    for point in data['periodicity_points']:
                        point = np.array(point)
                        self.periodicity_points.append([])
                        self.half_period_points.append([])
                        for half_period_round in range(1+('Plus' in objective and not self.PLUS_MEANS_STD_INCREASE)):
                            for minus_point in range(2):
                                cur_point = 1*point
                                if half_period_round:
                                    cur_point *= 0.5
                                if minus_point:
                                    cur_point *= -1
                                y_range, x_range = [IndexingHelper(cur_point[0]),IndexingHelper(cur_point[0], negative=True)], [IndexingHelper(cur_point[1]),IndexingHelper(cur_point[1],negative=True)]
                                ranges = []
                                for axis,cur_range in enumerate([x_range,y_range]):
                                    cur_range = [cur_range[0] if cur_range[0] is not None else 0,image_size[axis]+cur_range[1] if cur_range[1] is not None else image_size[axis]]
                                    cur_range = np.linspace(start=cur_range[0],stop=cur_range[1],
                                                            num=image_size[axis]-np.ceil(np.abs(np.array([0,image_size[axis]])-cur_range)).astype(np.int16).max())/image_size[axis]*2-1
                                    ranges.append(cur_range)
                                grid = np.meshgrid(*ranges)
                                if half_period_round:
                                    self.half_period_points[-1].append(torch.from_numpy(np.stack(grid, -1)).view([1] + list(grid[0].shape) + [2]).type(
                                        self.model.fake_H.dtype).to(self.model.fake_H.device))
                                else:
                                    self.periodicity_points[-1].append(torch.from_numpy(np.stack(grid,-1)).view([1]+list(grid[0].shape)+[2]).type(
                                        self.model.fake_H.dtype).to(self.model.fake_H.device))
                else:
                    self.periodicity_points = [np.array(point) for point in data['periodicity_points']]
            elif 'VGG' in objective and 'random' not in objective:
                self.GT_HR_VGG = model.netF(self.GT_HR).detach().to(self.device)
                self.loss = torch.nn.L1Loss().to(torch.device('cuda'))
            elif 'TV' in objective:
                self.STD_PRESERVING_WEIGHT = 100
            elif any([phrase in objective for phrase in ['hist','dict']]):
                self.automatic_temperature = auto_set_hist_temperature
                self.STD_PRESERVING_WEIGHT = 1e4
                if self.automatic_temperature:
                    assert 'hist' in objective,'Unsupported  for dictionary'
                    self.data['Z'] = self.Z_model()
                    pre_tanh_Z = self.Z_model.Z
                    pre_tanh_Z.requires_grad = True
                    model.feed_data(self.data, need_GT=False)
                    d_KLdiv_2_d_temperature = SoftHistogramLoss(bins=256,min=0,max=1,desired_hist_image=self.data[self.data_keys['GT']].detach(),desired_hist_image_mask=data['Desired_Im_Mask'],
                                                                input_im_HR_mask=self.image_mask,gray_scale=True,patch_size=3 if 'patch' in objective else 1,automatic_temperature=True,image_Z=pre_tanh_Z)
                    temperature_optimizer = torch.optim.Adam(d_KLdiv_2_d_temperature.optimizable_temperature.parameters(), lr=0.5)
                    temperature_optimizer.zero_grad()
                    initial_image = model.netG(model.var_L).to(self.device)
                    temperatures,gradient_sizes,KL_divs = [],[],[]
                    NUM_ITERS = 50
                    for tempertaure_seeking_iter in range(NUM_ITERS):
                        cur_KL_div,temperature_gradients_size = d_KLdiv_2_d_temperature(initial_image)
                        temperature_gradients_size.backward(retain_graph=(tempertaure_seeking_iter<(NUM_ITERS-1)))
                        temperature_optimizer.step()
                        KL_divs.append(cur_KL_div.item())
                        temperatures.append(d_KLdiv_2_d_temperature.temperature.item())
                        gradient_sizes.append(temperature_gradients_size.item())
                    optimal_temperature = temperatures[np.argmin(gradient_sizes)]
                else:
                    optimal_temperature = 5e-4 if 'hist' in objective else 1e-3
                self.loss = SoftHistogramLoss(bins=256,min=0,max=1,desired_hist_image=self.data[self.data_keys['GT']] if self.data is not None else None,
                    desired_hist_image_mask=data['Desired_Im_Mask'] if self.data is not None else None,input_im_HR_mask=self.image_mask,
                    gray_scale=True,patch_size=6 if 'patch' in objective else 1,temperature=optimal_temperature,dictionary_not_histogram='dict' in objective,
                    no_patch_DC='noDC' in objective,no_patch_STD='no_localSTD' in objective)
            elif 'Adversarial' in objective:
                self.netD = model.netD
                self.loss = GANLoss('wgan-gp', 1.0, 0.0).to(self.device)
            elif 'limited' in objective:
                self.initial_image = 1*model.fake_H.detach()
                self.rmse_weight = data['rmse_weight']
            self.optimizer = torch.optim.Adam(self.Z_model.parameters(), lr=initial_LR)
        else:
            self.optimizer = existing_optimizer
        self.LR = initial_LR
        self.scheduler = None#torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer,verbose=True,threshold=1e-2,min_lr=self.MIN_LR,cooldown=10)
        self.loggers = loggers
        self.cur_iter = 0
        self.max_iters = max_iters
        self.random_Z_inits = 'all' if (random_Z_inits or self.model_training)\
            else 'allButFirst' if (initial_pre_tanh_Z is not None and initial_pre_tanh_Z.size(0)<batch_size)\
            else False
        self.HR_unpadder = HR_unpadder

    def Masked_STD(self,first_image_only=False):
        if 'local' in self.objective:
            values_2_return = []
            for im_num in range(1 if first_image_only else self.model.fake_H.size(0)):
                values_2_return.append(torch.sparse.mm(self.patch_extraction_map,self.model.fake_H[im_num].mean(dim=0).view([-1, 1])).view([self.PATCH_SIZE_4_STD ** 2, -1]).std(dim=0))
                if self.non_covered_indexes_extraction_mat is not None:
                    values_2_return[-1] = torch.cat([values_2_return[-1],torch.sparse.mm(self.non_covered_indexes_extraction_mat,self.model.fake_H[im_num].mean(dim=0).view(
                    [-1, 1])).std(dim=0)], 0)
            return torch.stack(values_2_return, 1)
        else:
            return torch.std(self.model.fake_H * self.image_mask, dim=(1, 2, 3)).view(1,-1)

    def feed_data(self,data):
        self.data = data
        self.cur_iter = 0
        if 'l1' in self.objective:
            self.GT_HR = data[self.data_keys['GT']].to(self.device)
        elif 'hist' in self.objective:
            self.loss.Feed_Desired_Hist_Im(data[self.data_keys['GT']].to(self.device))

    def Manage_Model_Grad_Requirements(self,disable):
        if disable:
            self.original_requires_grad_status = []
            for p in self.model.netG.parameters():
                self.original_requires_grad_status.append(p.requires_grad)
                p.requires_grad = False
        else:
            for i, p in enumerate(self.model.netG.parameters()):
                p.requires_grad = self.original_requires_grad_status[i]

    def optimize(self):
        if 'Adversarial' in self.objective:
            self.model.netG.train(True) # Preventing image padding in the DTE code, to have the output fitD's input size
        self.Manage_Model_Grad_Requirements(disable=True)
        self.loss_values = []
        if self.random_Z_inits and self.cur_iter==0:
            self.Z_model.Randomize_Z(what_2_shuffle=self.random_Z_inits)
        z_iter = self.cur_iter
        while True:
            if self.max_iters>0:
                if z_iter==(self.cur_iter+self.max_iters):
                    break
            elif len(self.loss_values)>=-self.max_iters:# Would stop when loss siezes to decrease, or after 5*(-)max_iters
                if z_iter==(self.cur_iter-5*self.max_iters):
                    break
                if (self.loss_values[self.max_iters] - self.loss_values[-1]) / np.abs(self.loss_values[self.max_iters]) < 1e-2 * self.LR:
                    break
            self.optimizer.zero_grad()
            self.data['Z'] = self.Z_model()
            self.model.feed_data(self.data, need_GT=False)
            self.model.fake_H = self.model.netG(self.model.model_input)
            if self.jpeg_extractor is not None:
                self.model.fake_H = self.jpeg_extractor(self.model.fake_H)
            if self.model_training:
                self.model.fake_H = self.HR_unpadder(self.model.fake_H)
            if 'random' in self.objective:
                if 'l1' in self.objective:
                    data_in_loss_domain = self.model.fake_H
                elif 'VGG' in self.objective:
                    data_in_loss_domain = self.model.netF(self.model.fake_H)
                Z_loss = torch.min((data_in_loss_domain.unsqueeze(0) - data_in_loss_domain.unsqueeze(1)).abs() + torch.eye(
                        data_in_loss_domain.size(0)).unsqueeze(2).unsqueeze(3).unsqueeze(4).to(data_in_loss_domain.device), dim=0)[0]
                if 'limited' in self.objective:
                    rmse = (data_in_loss_domain - self.initial_image).abs()
                    if z_iter==0:
                        rmse_weight = 1*self.rmse_weight#*Z_loss.mean().item()/rmse.mean().item()
                    Z_loss = Z_loss-rmse_weight*rmse
                if self.Z_mask is not None:
                    Z_loss = Z_loss*self.Z_mask
                Z_loss = -1*Z_loss.mean(dim=(1,2,3))
            elif any([phrase in self.objective for phrase in ['l1','scribble']]):
                Z_loss = self.loss(self.model.fake_H.to(self.device), self.GT_HR.to(self.device))
            elif 'desired_SVD' in self.objective:
                Z_loss = self.loss({'SR':self.model.fake_H.to(self.device)}).mean()
            elif any([phrase in self.objective for phrase in ['hist','dict']]):
                Z_loss = self.loss(self.model.fake_H.to(self.device))
                if 'localSTD' in self.objective:
                    Z_loss = Z_loss+(self.STD_PRESERVING_WEIGHT*(self.Masked_STD(first_image_only=False)-self.initial_STD)**2).mean(0).to(self.device)
            elif 'Adversarial' in self.objective:
                Z_loss = self.loss(self.netD(self.model.DTE_net.HR_unpadder(self.model.fake_H).to(self.device)),True)
            elif 'STD' in self.objective and not any([phrase in self.objective for phrase in ['periodicity','TV']]):
                Z_loss = self.Masked_STD(first_image_only=False)
                if any([phrase in self.objective for phrase in ['increase', 'decrease']]):
                    Z_loss = (Z_loss-self.desired_STD)**2
                Z_loss = Z_loss.mean(0)
            elif 'Mag' in self.objective:
                values_2_return = []
                for im_num in range(self.model.fake_H.size(0)):
                    values_2_return.append(((torch.sparse.mm(self.patch_extraction_map,self.model.fake_H[im_num].mean(dim=0).view([-1, 1])).view(
                        [self.PATCH_SIZE_4_STD ** 2, -1]) - self.desired_patches) ** 2).mean())
                Z_loss = torch.stack(values_2_return,0)
            elif 'periodicity' in self.objective:
                Z_loss = self.PeriodicityLoss().to(self.device)
                if 'Plus' in self.objective and self.PLUS_MEANS_STD_INCREASE:
                    Z_loss = Z_loss+self.STD_PRESERVING_WEIGHT*((self.Masked_STD(first_image_only=False)-self.desired_STD)**2).mean()
            elif 'TV' in self.objective:
                Z_loss = (self.STD_PRESERVING_WEIGHT*(self.Masked_STD(first_image_only=False)-self.initial_STD)**2).mean(0)+TV_Loss(self.model.fake_H * self.image_mask).to(self.device)
            elif 'VGG' in self.objective:
                Z_loss = self.loss(self.model.netF(self.model.fake_H).to(self.device),self.GT_HR_VGG)
            if 'max' in self.objective:
                Z_loss = -1*Z_loss
            # Z_loss.backward(retain_graph=(self.HR_unpadder is not None))
            cur_LR = self.optimizer.param_groups[0]['lr']
            if self.loggers is not None:
                for logger_num,logger in enumerate(self.loggers):
                    cur_value = Z_loss[logger_num].item() if Z_loss.dim()>0 else Z_loss.item()
                    logger.print_format_results('val', {'epoch': 0, 'iters': z_iter, 'time': time.time(), 'model': '','lr': cur_LR, 'Z_loss': cur_value}, dont_print=True)
            if not self.model_training:
                self.latest_Z_loss_values = [val.item() for val in Z_loss]
            Z_loss = Z_loss.mean()
            Z_loss.backward()
            self.loss_values.append(Z_loss.item())
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step(Z_loss)
                if cur_LR<=1.2*self.MIN_LR:
                    break
            z_iter += 1
        if 'Adversarial' in self.objective:
            self.model.netG.train(False) # Preventing image padding in the DTE code, to have the output fitD's input size
        if 'random' in self.objective and 'limited' in self.objective:
            self.loss_values[0] = self.loss_values[1] #Replacing the first loss values which is close to 0 in this case, to prevent discarding optimization because loss increased compared to it.
        # if 'STD' in self.objective or 'periodicity' in self.objective:
        if not self.model_training:
            print('Final STDs: ',['%.3e'%(val.item()) for val in self.Masked_STD(first_image_only=False).mean(0)])
        self.cur_iter = z_iter+1
        Z_2_return = self.Z_model.Return_Detached_Z()
        self.Manage_Model_Grad_Requirements(disable=False)
        if self.model_training:# Results of all optimization iterations were cropped, so I do another one without cropping and with Gradients computation (for model training)
            self.data['Z'] = Z_2_return
            self.model.feed_data(self.data, need_GT=False)
            self.model.fake_H = self.model.netG(self.model.model_input)
        return Z_2_return

    def Return_Translated_SubImage(self,image, translation):
        y_range, x_range = [IndexingHelper(translation[0]), IndexingHelper(translation[0], negative=True)], [IndexingHelper(translation[1]), IndexingHelper(translation[1], negative=True)]
        return image[:, :, y_range[0]:y_range[1], x_range[0]:x_range[1]]

    def Return_Interpolated_SubImage(self,image, grid):
        return torch.nn.functional.grid_sample(image, grid.repeat([image.size(0),1,1,1]))

    def PeriodicityLoss(self):
        loss = 0 if 'Plus' in self.objective and self.PLUS_MEANS_STD_INCREASE else (self.STD_PRESERVING_WEIGHT*(self.Masked_STD(first_image_only=False)-self.initial_STD)**2).mean()
        image = self.model.fake_H
        mask = self.image_mask.unsqueeze(0).unsqueeze(0)
        for point_num,point in enumerate(self.periodicity_points):
            if 'nonInt' in self.objective:
                cur_mask = self.Return_Interpolated_SubImage(mask,point[0])*self.Return_Interpolated_SubImage(mask,point[1])
                loss = loss + (cur_mask * (self.Return_Interpolated_SubImage(image, point[0]) - self.Return_Interpolated_SubImage(image,point[1])).abs()).mean(dim=(1, 2, 3))
                if 'Plus' in self.objective and not self.PLUS_MEANS_STD_INCREASE:
                    cur_half_cycle_mask = self.Return_Interpolated_SubImage(mask,self.half_period_points[point_num][0])*self.Return_Interpolated_SubImage(mask,self.half_period_points[point_num][1])
                    loss = loss - (cur_half_cycle_mask * (self.Return_Interpolated_SubImage(image, self.half_period_points[point_num][0]) -\
                        self.Return_Interpolated_SubImage(image,self.half_period_points[point_num][1])).abs()).mean(dim=(1, 2, 3))
            else:
                cur_mask = self.Return_Translated_SubImage(mask,point)*self.Return_Translated_SubImage(mask,-point)
                loss = loss+(cur_mask*(self.Return_Translated_SubImage(image,point)-self.Return_Translated_SubImage(image,-point)).abs()).mean(dim=(1, 2, 3))
        return loss

    def ReturnStatus(self):
        return self.Z_model.PreTanhZ(),self.optimizer

def IndexingHelper(index,negative=False):
    if negative:
        return index if index < 0 else None
    else:
        return index if index > 0 else None
