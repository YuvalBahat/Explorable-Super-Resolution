import os
import torch
import torch.nn as nn
import DTE.DTEnet as DTEnet
import numpy as np

class BaseModel():
    def __init__(self, opt):
        self.opt = opt
        self.save_dir = opt['path']['models']  # save models
        self.device = torch.device('cuda' if opt['gpu_ids'] is not None else 'cpu')
        self.is_train = opt['is_train']
        self.schedulers = []
        self.optimizers = []

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

    # helper printing function that can be used by subclasses
    def get_network_description(self, network):
        if isinstance(network, nn.DataParallel):
            network = network.module
        s = str(network)
        n = sum(map(lambda x: x.numel(), network.parameters()))
        if 'Discriminator' in str(network.__class__):
            kernel_size,strides = self.return_kernel_sizes_and_strides(network)
            receptive_field = self.calc_receptive_field(kernel_size,strides)
            return s,n,receptive_field
        else:
            return s, n

    def return_kernel_sizes_and_strides(self,network):
        kernel_sizes,strides = [],[]
        children = [child for child in network.children()]
        if len(children)>0:
            for child in children:
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
    def save_network(self, save_dir, network, network_label, iter_label):
        save_filename = '{}_{}.pth'.format(iter_label, network_label)
        save_path = os.path.join(save_dir, save_filename)
        if isinstance(network, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, save_path)

    # helper loading function that can be used by subclasses
    def load_network(self, load_path, network, strict=True):
        if isinstance(network, nn.DataParallel):
            network = network.module
        loaded_state_dict = torch.load(load_path)
        if self.opt['network_G']['DTE_arch']:
            loaded_state_dict = DTEnet.Adjust_State_Dict_Keys(loaded_state_dict,network.state_dict())
        # network.load_state_dict(loaded_state_dict, strict=(strict and not self.opt['network_G']['DTE_arch']))
        network.load_state_dict(loaded_state_dict, strict=strict)

