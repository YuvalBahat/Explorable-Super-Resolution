import os
import torch
import torch.nn as nn
import CEM.CEMnet as CEMnet
import numpy as np
import collections
import matplotlib.pyplot as plt
import re
import copy
# from torch.nn import Upsample
# from utils.util import compute_RF_numerical

class BaseModel():
    def __init__(self, opt):
        self.opt = opt
        self.save_dir = opt['path']['models']  # save models
        self.device = torch.device('cuda' if opt['gpu_ids'] is not None else 'cpu')
        self.is_train = opt['is_train']
        self.schedulers = []
        self.optimizers = []
        self.weights_averaging_counter = 0

    def feed_data(self, data):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        pass

    def get_current_losses(self):
        pass

    def print_network(self):
        pass

    def save(self, label):
        pass

    def load(self):
        pass

    def update_learning_rate(self,cur_step=None):
        for scheduler in self.schedulers:
            scheduler.step(cur_step)

    def get_current_learning_rate(self):
        return self.optimizers[0].param_groups[0]['lr']

    def Convert_2_LR(self,input,size):
        return nn.functional.interpolate(input,size=size,mode='bilinear',align_corners=True)
        # return Upsample(size=size,mode='bilinear')

    # helper printing function that can be used by subclasses
    def get_network_description(self, network):
        if isinstance(network, nn.DataParallel):
            network = network.module
        output = {}
        output['s'] = str(network)
        output['n'] = sum(map(lambda x: x.numel(), network.parameters()))
        # receptive_field = self.numeric_effective_field(network)
        # networks_device = next(network.parameters()).device
        # receptive_field = compute_RF_numerical(network.cpu(),np.ones([1,3,256,256]))
        # network.to(networks_device)
        if any([token in str(network.__class__) for token in ['Discriminator','DnCNN','Sequential']]):
            kernel_size,strides = self.return_kernel_sizes_and_strides(network)
            output['receptive_field'] = self.calc_receptive_field(kernel_size,strides)
        return output
        #     return s,n,receptive_field
        # else:
        #     return s, n
    def numeric_effective_field(self,model):
        INOUT_SIZE = 501
        mid_index = INOUT_SIZE//2+1
        zeros_input = torch.zeros(1,next(model.parameters()).shape[1],INOUT_SIZE,INOUT_SIZE).type(next(model.parameters()).dtype).to(next(model.parameters()).device)
        delta_input = torch.zeros_like(zeros_input)
        delta_input[0,:,mid_index,mid_index] = 1
        model.eval()
        diffs_image = (model(delta_input)-model(zeros_input)).abs().squeeze(0).data.cpu().numpy()
        model.train()
        for i_ind in [0,-1]:
            for j_ind in [0,-1]:
                if np.max(diffs_image[:,i_ind,j_ind])>0:
                    return None
        receptive_field = np.maximum(np.argwhere(np.sum(diffs_image,axis=(0,1)))[-1]-mid_index,mid_index-np.argwhere(np.sum(diffs_image,axis=(0,1)))[0])
        receptive_field = 1+2*np.maximum(receptive_field,np.maximum(np.argwhere(np.sum(diffs_image,axis=(0,2)))[-1]-mid_index,mid_index-np.argwhere(np.sum(diffs_image,axis=(0,1)))[0]))
        return receptive_field


    def return_kernel_sizes_and_strides(self,network):
        kernel_sizes,strides = [],[]
        if 'num_modules' not in self.__dir__():
            self.num_modules = network.num_modules if 'num_modules' in network.__dir__() else None
        children = [child for child in network.children()]
        if len(children)>0:
            last_child = self.num_modules if (isinstance(network,nn.ModuleList) and self.num_modules is not None) else len(children)
            for child in children[:last_child]:
                temp = self.return_kernel_sizes_and_strides(child)
                kernel_sizes += temp[0]
                strides += temp[1]
        if 'Conv' in str(network.__class__):
            kernel_sizes.append(network.kernel_size[0])
            strides.append(network.stride[0])
        return (kernel_sizes,strides)

    def calc_receptive_field(self,kernel_sizes, strides):
        assert len(kernel_sizes) == len(strides), 'Parameter lists must have same length'
        if strides[-1] > 1:
            print('Stride %d in top layer is not taken into account in receptive field size' % (strides[-1]))
        field_size = kernel_sizes[0]
        for i in range(1, len(kernel_sizes)):
            field_size += (kernel_sizes[i] - 1) * int(np.prod([stride for stride in strides[:i]]))
        return field_size

    # helper saving function that can be used by subclasses
    def save_network(self, save_dir, network, network_label, iter_label,optimizer):
        save_filename = '{}_{}.pth'.format(iter_label, network_label)
        save_path = os.path.join(save_dir, save_filename)
        if isinstance(network, nn.DataParallel):
            network = network.module
        model_state_dict = network.state_dict()
        optimizer_state_dict = optimizer.state_dict()
        # for SD in [model_state_dict,optimizer_state_dict]:
        for key, param in model_state_dict.items():
            model_state_dict[key] = param.cpu()
        torch.save({'model_state_dict':model_state_dict,'optimizer_state_dict':optimizer_state_dict}, save_path)
        return save_path

    # helper loading function that can be used by subclasses
    def load_network(self, load_path, network, strict=False,optimizer=None):
        if isinstance(network, nn.DataParallel):
            network = network.module
        loaded_state_dict = torch.load(load_path)
        if 'optimizer_state_dict' in loaded_state_dict.keys():
            if optimizer is not None:# Not performed in the test case:
                optimizer.load_state_dict(loaded_state_dict['optimizer_state_dict'])
            loaded_state_dict = loaded_state_dict['model_state_dict']
        if self.opt['network_G']['CEM_arch']:
            loaded_state_dict = CEMnet.Adjust_State_Dict_Keys(loaded_state_dict,network.state_dict())
        # network.load_state_dict(loaded_state_dict, strict=(strict and not self.opt['network_G']['CEM_arch']))
        loaded_state_dict = self.process_loaded_state_dict(loaded_state_dict=loaded_state_dict,current_state_dict=network.state_dict())
        network.load_state_dict(loaded_state_dict, strict=strict)

    def Set_Require_Grad_Status(self,network,status):
        return
        for p in network.parameters():
            p.requires_grad = status

    # def average_across_model_snapshots(self,apply):
    #     #     Used before performing evaluation during training. When apply=True, saving the current generator weights, and loading and averaging across latest svaed weights. When apply=False, restoring current weights.
    #     if apply:
    #         self.temp_stored_G_weights = copy.deepcopy(self.netG.state_dict())
    #         averaged_dicts = []
    #         for snapshot in os.listdir(self.save_dir):
    #             if '_G.pth' in snapshot and int(re.search('(\d)+(?=_G)',snapshot).group(0))>self.gradient_step_num-self.opt['train']['val_save_freq']:
    #                 averaged_dicts.append(os.path.join(self.save_dir,snapshot))
    #         print('Evaluating by averaging weights over %d latest snapshots'%(len(averaged_dicts)+1))
    #         averaged_state_dict = self.netG.state_dict()
    #         for i,snapshot in enumerate(averaged_dicts):
    #             loaded_snapshot = torch.load(snapshot)
    #             for key in averaged_state_dict:
    #                 averaged_state_dict[key] = (i+1)/(i+2)*averaged_state_dict[key]+1/(i+2)*loaded_snapshot['model_state_dict'][key.replace('module.','')].cuda()
    #         self.netG.load_state_dict(averaged_state_dict)
    #     else:
    #         self.netG.load_state_dict(self.temp_stored_G_weights)

    def update_running_avg(self):
        translated_step_num = [self.gradient_step_num+i for i in range(min([self.opt['train']['val_running_avg_steps'],self.gradient_step_num+1]))]
        steps_before_eval = [(v%self.opt['train']['val_freq'])==0 for v in translated_step_num]
        if any(steps_before_eval):
            cur_state_dic = self.netG.state_dict()
            # if steps_before_eval[-1]: #If this is the first step of calculating running average:
            if self.weights_averaging_counter==0:  # If this is the first step of calculating running average:
                self.weights_averaging_counter = 1
                self.running_avg_weights = copy.deepcopy(cur_state_dic)
            else:
                self.weights_averaging_counter += 1
                for key in cur_state_dic:
                    self.running_avg_weights[key] = (self.weights_averaging_counter-1)/self.weights_averaging_counter*self.running_avg_weights[key]+\
                        1/self.weights_averaging_counter*cur_state_dic[key]
        else:
            self.weights_averaging_counter = 0

    def toggle_running_avg_weight(self,on):
        if on:
            self.temp_saved_weights = copy.deepcopy(self.netG.state_dict())
            self.netG.load_state_dict(self.running_avg_weights)
        else:
            self.netG.load_state_dict(self.temp_saved_weights)

    def process_loaded_state_dict(self,loaded_state_dict,current_state_dict):
        SPECTRAL_NORMALIZATIONFIX_PATCH = False
        modified_state_dict = collections.OrderedDict()
        LATENT_WEIGHTS_RELATIVE_STD = 0.
        # num_latent_channels = self.opt['network_G']['latent_channels']
        current_keys = [k for k in current_state_dict.keys()]
        if not SPECTRAL_NORMALIZATIONFIX_PATCH:
            assert len(current_keys)==len([key for key in loaded_state_dict.keys()]),'Loaded model and current one should have the same number of parameters'
        modified_key_names_counter,zero_extended_weights_counter = 0,0
        for i,key in enumerate(loaded_state_dict.keys()):
            current_key = current_keys[i]
            loaded_size = loaded_state_dict[key].size()
            current_size = current_state_dict[current_key].size()
            if SPECTRAL_NORMALIZATIONFIX_PATCH and len(loaded_size)!=len(current_size):
                continue
            if key!=current_key:
                assert loaded_size[:1]+loaded_size[2:]==current_size[:1]+current_size[2:],'Unmatching parameter sizes after changing parameter key name'
                modified_key_names_counter += 1
            if 'latent_input' in self.__dict__ and self.latent_input is not None and self.num_latent_channels>0 and \
                'weight' in key and loaded_state_dict[key].dim()>1 and \
                current_state_dict[current_key].size()[1] in list(loaded_state_dict[key].size()[1] + np.arange(self.num_latent_channels)+1):
                # current_state_dict[current_key].size()[1] in list(loaded_state_dict[key].size()[1]+self.num_latent_channels*np.array([1,self.opt['scale']**2])):
                # In case we initialize a newly trained model that has latent input, with pre-trained model that doesn't have, add weights corresponding to
                # the added input layers (added as first layers), whose STD is LATENT_WEIGHTS_RELATIVE_STD*(STD of existing weights in this kernel):
                additional_channels = current_state_dict[current_key].size()[1]-loaded_state_dict[key].size()[1]
                loaded_weights_STD = loaded_state_dict[key].std()
                modified_state_dict[current_key] = torch.cat([LATENT_WEIGHTS_RELATIVE_STD*loaded_weights_STD/current_state_dict[current_key][:,:additional_channels,:,:].std()*\
                    current_state_dict[current_key][:,:additional_channels,:,:].view([current_state_dict[current_key].size()[0],additional_channels]+list(current_state_dict[current_key].size()[2:])).cuda(),\
                                                              loaded_state_dict[key].cuda()],1)
                # self.channels_idx_4_grad_amplification[i] = [c for c in range(additional_channels)]
                zero_extended_weights_counter += 1
            elif 'CEM_net' in self.__dict__ and self.CEM_arch and any([CEM_op in key for CEM_op in self.CEM_net.OP_names]):
                continue # Not loading CEM module weights
            else:
                modified_state_dict[current_key] = loaded_state_dict[key]
        if modified_key_names_counter>0:
            print('Warning: Modified %d key names due to the change to using ModuleLists' % (modified_key_names_counter))
        if zero_extended_weights_counter>0:
            print('Warning: %d model weights were augmented with zeros to accommodate for larger inputs' % (zero_extended_weights_counter))
        return modified_state_dict

    def plot_curves(self, steps, loss, smoothing='yes'):
        assert smoothing in ['yes','no','extra']
        # SMOOTH_CURVES = True
        # if smoothing=='yes':
        #     smoothing = 'no'
        if smoothing!='no':
            steps_induced_upper_bound = np.ceil(1000/np.percentile(np.diff(steps),99)) if len(steps)>1 else 1
            smoothing_win = np.minimum(np.maximum(len(loss)/20,np.sqrt(len(loss))),steps_induced_upper_bound).astype(np.int32)
            if smoothing=='extra':
                smoothing_win = np.minimum(len(loss)//3,smoothing_win*100)
            loss = np.convolve(loss,np.ones([smoothing_win])/smoothing_win,'valid')
            if steps is not None:
                steps = np.convolve(steps, np.ones([smoothing_win]) / smoothing_win,'valid')
        if steps is not None:
            plt.plot(steps,loss)
        else:
            plt.plot(loss)
        return np.min(loss),np.max(loss)

    def display_log_figure(self):
        # keys_2_display = ['l_g_pix', 'l_g_fea', 'l_g_range', 'l_g_gan', 'l_d_real', 'l_d_fake', 'D_real', 'D_fake','D_logits_diff','psnr_val']
        keys_2_display = ['l_g_gan','D_logits_diff', 'psnr_val','l_g_pix_log_rel','l_g_fea','l_g_range','l_d_real','D_loss_STD','l_g_latent','l_e',
                          'l_g_latent_0','l_g_latent_1','l_g_latent_2','l_g_optimalZ','l_g_pix','l_g_highpass','l_g_shift_invariant','D_update_ratio',
                          'D_G_prob_ratio','Correctly_distinguished','Z_effect','post_train_D_diff','G_step_D_gain','clamped_portion','niqe_val']
        NON_SMOOTHED_LOGS = ['post_train_D_diff','D_update_ratio','D_logits_diff']
        PER_KEY_FIGURE = True
        legend_strings = []
        plt.figure(2)
        plt.clf()
        min_global_val, max_global_val = np.finfo(np.float32).max,np.finfo(np.float32).min
        for key in keys_2_display:
            if key in self.log_dict.keys() and len(self.log_dict[key])>0:
                if PER_KEY_FIGURE:
                    plt.figure(1)
                    plt.clf()
                if isinstance(self.log_dict[key][0],tuple) or len(self.log_dict[key][0])==2:
                    cur_curve = [np.array([val[0] for val in self.log_dict[key]]),np.array([val[1] for val in self.log_dict[key]])]
                    min_val,max_val = self.plot_curves(cur_curve[0],cur_curve[1],smoothing='no' if key in NON_SMOOTHED_LOGS else 'yes')
                    if 'LR_decrease' in self.log_dict.keys():
                        for decrease in self.log_dict['LR_decrease']:
                            plt.plot([decrease[0],decrease[0]],[min_val,max_val],'k')
                    if isinstance(self.log_dict[key][0][1],torch.Tensor):
                        series_avg = np.mean([val[1].data.cpu().numpy() for val in self.log_dict[key]])
                    else:
                        series_avg = np.mean([val[1] for val in self.log_dict[key]])
                else:
                    raise Exception('Should always have step numbers')
                    self.plot_curves(self.log_dict[key])
                    # plt.plot(self.log_dict[key])
                    if isinstance(self.log_dict[key][0][1],torch.Tensor):
                        series_avg = np.mean([val[1].data.cpu().numpy() for val in self.log_dict[key]])
                    else:
                        series_avg = np.mean(self.log_dict[key])
                cur_legend_string = [key + ' (%.2e)' % (series_avg)]
                if len(cur_curve[0])>100:
                    self.plot_curves(cur_curve[0], cur_curve[1], smoothing='extra')
                    cur_legend_string.append(key+'_smoothed')
                if PER_KEY_FIGURE:
                    plt.xlabel('Steps')
                    baseline_logs = ['quantized_'+key,'GT_'+key]
                    for bl_log in baseline_logs:
                        if bl_log in self.log_dict.keys() and len(self.log_dict[bl_log])>0:
                            plt.plot([cur_curve[0][0],cur_curve[0][-1]],2*[self.log_dict[bl_log][0][1]])
                            cur_legend_string.append('%s (%.2e)' % (bl_log,self.log_dict[bl_log][0][1]))
                    plt.legend(cur_legend_string, loc='best')
                    plt.savefig(os.path.join(self.log_path, 'logs_%s.pdf' % (key)))
                    plt.figure(2)
                    if key=='psnr_val':
                        cur_legend_string[0] = 'MSE_val' + ' (%s:%.2e)' % (key,series_avg)
                        cur_curve[1] = 255*np.exp(-cur_curve[1]/20)
                    if np.std(cur_curve[1])>0:
                        cur_curve[1] = (cur_curve[1]-np.mean(cur_curve[1]))/np.std(cur_curve[1])
                    else:
                        cur_curve[1] = (cur_curve[1] - np.mean(cur_curve[1]))
                    min_val,max_val = self.plot_curves(cur_curve[0],cur_curve[1],smoothing='no' if key in NON_SMOOTHED_LOGS else 'yes')
                    min_global_val,max_global_val = np.minimum(min_global_val,min_val),np.maximum(max_global_val,max_val)
                legend_strings.append(cur_legend_string[0])
        plt.legend(legend_strings,loc='best')
        plt.xlabel('Steps')
        if 'LR_decrease' in self.log_dict.keys():
            for decrease in self.log_dict['LR_decrease']:
                plt.plot([decrease[0], decrease[0]], [min_global_val,max_global_val], 'k')
        plt.savefig(os.path.join(self.log_path,'logs.pdf'))

