from scipy.signal import convolve2d as conv2
import numpy as np
tf_loaded,pytorch_loaded = False,False
try:
    import tensorflow as tf
    tf_loaded = True
except:
    pass
try:
    import torch
    import torch.nn as nn
    pytorch_loaded = True
except:
    pass
import copy
from imresize_DTE import imresize,calc_strides

class DTEnet:
    NFFT_add = 36

    def __init__(self,conf,ds_kernel=None):
        self.conf = conf
        self.ds_factor = np.array(conf.scale_factor,dtype=np.int32)
        assert np.round(self.ds_factor)==self.ds_factor,'Currently only supporting integer scale factors'
        assert ds_kernel is None,'To support given kernels, change the Return_Invalid_Margin_Size_in_LR function and make sure everything else works'
        if ds_kernel is None:
            # ds_kernel = np.rot90(imresize(None,[self.ds_factor,self.ds_factor],return_upscale_kernel=True),2).astype(np.float32)/(self.ds_factor**2)
            ds_kernel = Return_Default_kernel(self.ds_factor)
        self.ds_kernel = ds_kernel
        self.ds_kernel_invalidity_half_size_LR = self.Return_Invalid_Margin_Size_in_LR('ds_kernel')
        # self.ds_kernel_invalidity_half_size = np.argwhere(Return_Filter_Energy_Distribution(self.ds_kernel)[::-1]>=self.conf.filter_pertubation_limit)[0][0]
        self.compute_inv_hTh()
        self.invalidity_margins_LR = 2* self.ds_kernel_invalidity_half_size_LR + self.inv_hTh_invalidity_half_size
        self.invalidity_margins_HR = self.ds_factor * self.invalidity_margins_LR

    def Return_Invalid_Margin_Size_in_LR(self,filter):
        TEST_IM_SIZE = 100
        assert filter in ['ds_kernel','inv_hTh']
        if filter=='ds_kernel':
            output_im = imresize(np.ones([self.ds_factor*TEST_IM_SIZE, self.ds_factor*TEST_IM_SIZE]),[1/self.ds_factor],use_zero_padding=True)
        elif filter=='inv_hTh':
            output_im = conv2(np.ones([TEST_IM_SIZE,TEST_IM_SIZE]), self.inv_hTh,mode='same')
        output_im /= output_im[int(TEST_IM_SIZE/2),int(TEST_IM_SIZE/2)]
        invalidity_mask = np.exp(-np.abs(np.log(output_im)))<self.conf.filter_pertubation_limit
        margin_sizes = [len(np.argwhere(invalidity_mask[:int(TEST_IM_SIZE/2),int(TEST_IM_SIZE/2)])),
                        len(np.argwhere(invalidity_mask[int(TEST_IM_SIZE / 2),:int(TEST_IM_SIZE / 2)]))]
        assert margin_sizes[0]==margin_sizes[1],'Different margins for each axis. Currently not supporting non-isotropic filters'
        return margin_sizes[0]

    def Pad_LR_Batch(self,batch,num_recursion=1):
        # margin_size = 2*self.ds_kernel_invalidity_half_size_LR+self.inv_hTh_invalidity_half_size
        for i in range(num_recursion):
            batch = 1.0*np.pad(batch, pad_width=((0, 0), (self.invalidity_margins_LR, self.invalidity_margins_LR),(self.invalidity_margins_LR, self.invalidity_margins_LR), (0, 0)), mode='edge')
        return batch
    def Unpad_HR_Batch(self,batch,num_recursion=1):
        # margin_size = self.ds_factor*(2*self.ds_kernel_invalidity_half_size_LR+self.inv_hTh_invalidity_half_size)
        margins_2_remove = (self.ds_factor**(num_recursion))*self.invalidity_margins_LR*num_recursion
        return batch[:,margins_2_remove:-margins_2_remove,margins_2_remove:-margins_2_remove, :]

    def DT_Satisfying_Upscale(self,LR_image):
        margin_size = 2*self.inv_hTh_invalidity_half_size+self.ds_kernel_invalidity_half_size_LR
        LR_image = Pad_Image(LR_image,margin_size)
        HR_image = imresize(np.stack([conv2(LR_image[:,:,channel_num],self.inv_hTh,mode='same') for channel_num in range(LR_image.shape[-1])],-1),scale_factor=[self.ds_factor])
        return Unpad_Image(HR_image,self.ds_factor*margin_size)

    def WrapArchitecture_PyTorch(self,generated_image=None):
        assert pytorch_loaded,'Failed to load PyTorch - Necessary for this function of DTE'
        model =  DTE_PyTorch(self,generated_image)
        self.wraped_model = generated_image

        return model


    def WrapArchitecture(self,model,unpadded_input_t,generated_image_t=None):
        assert tf_loaded,'Failed to load TensorFlow - Necessary for this function of DTE'
        PAD_GENRATED_TOO = True
        self.model = model
        with model.as_default():
            self.compute_conv_with_inv_hTh_OP()
            self.create_scaling_OPs()
            # Padding image for inference time or leaving as-is for training (taking x2 the calculated margins for extra-safety):
            LR_padding_size = 2*self.invalidity_margins_LR
            HR_padding_size = 2*self.invalidity_margins_HR
            self.pre_pad_input = tf.placeholder(dtype=tf.bool,name='pre_pad_input')
            LR_padding_OP = Create_Tensor_Pad_OP(LR_padding_size)
            HR_padding_OP = Create_Tensor_Pad_OP(HR_padding_size)
            def Return_Padded_Input():
                return LR_padding_OP(unpadded_input_t)
                # return tf.pad(unpadded_input_t,paddings=tf.constant([[0,0],[LR_padding_size,LR_padding_size],[LR_padding_size,LR_padding_size],[0,0]]),
                #               mode='SYMMETRIC')
            def Return_Unpadded_Input():
                return unpadded_input_t
            self.LR_input_t = tf.cond(self.pre_pad_input,true_fn=Return_Padded_Input,false_fn=Return_Unpadded_Input)
            # self.LR_input_t = unpadded_input_t

            self.projected_upscaled_input = self.Upscale_OP(self.Conv_LR_with_Inv_hTh_OP((self.LR_input_t)))
            if generated_image_t is None:
                print('Creating an HR image generator network...')
                self.create_im_generator()
            else:
                print('Using the given generated image as HR image...')
                self.generated_im = generated_image_t
            if PAD_GENRATED_TOO:
                def Return_Padded_Generated_Im():
                    return HR_padding_OP(self.generated_im)
                def Return_Unpadded_Generated_Im():
                    return self.generated_im
                self.generated_im = tf.cond(self.pre_pad_input,true_fn=Return_Padded_Generated_Im,false_fn=Return_Unpadded_Generated_Im)

            # return tf.slice(self.generated_im, begin=[0,0,0,0],
            #          size=tf.stack([-1, self.ds_factor * tf.shape(unpadded_input_t)[1],
            #                         self.ds_factor * tf.shape(unpadded_input_t)[2], -1]))
            # return tf.slice(self.projected_upscaled_input+self.generated_im/1000, begin=[0,0,0,0],
            #          size=tf.stack([-1, self.ds_factor * tf.shape(unpadded_input_t)[1],
            #                         self.ds_factor * tf.shape(unpadded_input_t)[2], -1]))

            self.projected_generated_im = self.Upscale_OP(self.Conv_LR_with_Inv_hTh_OP(self.DownscaleOP(self.generated_im)))
            self.projected_2_ortho_generated_im = self.generated_im-self.projected_generated_im
            if PAD_GENRATED_TOO:
                output = tf.add(self.projected_upscaled_input,self.projected_2_ortho_generated_im,name='DTE_add_subspaces')
            else:
                output = self.projected_upscaled_input
            # Remove image padding for inference time or leaving as-is for training:
            def Return_Output():
                return output
            def Return_Cropped_Output():
                # margins_2_crop = tf.cast((tf.shape(output)-self.ds_factor*tf.shape(unpadded_input_t))/2,dtype=tf.int32)
                margins_2_crop = tf.cast((tf.shape(output)-self.ds_factor*tf.shape(unpadded_input_t))/2,dtype=tf.int32)
                return tf.slice(output,begin=tf.stack([0,margins_2_crop[1],margins_2_crop[2],0]),
                                size=tf.stack([-1,self.ds_factor*tf.shape(unpadded_input_t)[1],self.ds_factor*tf.shape(unpadded_input_t)[2],-1]))
            self.output_t = tf.cond(self.pre_pad_input,true_fn=Return_Cropped_Output,false_fn=Return_Output)
            if PAD_GENRATED_TOO:
                return self.output_t
            else:
                return tf.add(self.output_t,self.projected_2_ortho_generated_im,name='DTE_add_subspaces')

    def Enforce_DT_on_Image_Pair(self,LR_input,HR_input):
        margin_size = 2*self.ds_kernel_invalidity_half_size_LR*self.ds_factor
        # HR_projected_2_h_subspace = Unpad_Image(self.DT_Satisfying_Upscale(imresize(Pad_Image(HR_input,margin_size),scale_factor=[1/self.ds_factor])),margin_size)
        HR_projected_2_h_subspace = self.DT_Satisfying_Upscale(imresize(HR_input,scale_factor=[1/self.ds_factor]))
        return  HR_input-HR_projected_2_h_subspace+self.DT_Satisfying_Upscale(LR_input)
    def Supplement_Pseudo_DTE(self,input_t):
        return self.Learnable_Upscale_OP(self.Conv_LR_with_Learnable_OP(self.Learnable_DownscaleOP(input_t)))

    def create_im_generator(self):
        with self.model.as_default():
            with tf.variable_scope('DTE_generator'):
                upscaling_filter_shape = self.conf.filter_shape[0][:-2]+[3,self.ds_factor**2]
                self.conf.filter_shape[1] = copy.deepcopy(self.conf.filter_shape[1])
                self.conf.filter_shape[1][2] = 3
                self.upscaling_filter = tf.get_variable(shape=upscaling_filter_shape,name='upscaling_filter',
                    initializer=tf.random_normal_initializer(stddev=np.sqrt(self.conf.init_variance / np.prod(upscaling_filter_shape[0:3]))))
                self.G_filters_t = [tf.get_variable(shape=self.conf.filter_shape[ind+1], name='filter_%d' % ind,
                    initializer=tf.random_normal_initializer(stddev=np.sqrt(self.conf.init_variance / np.prod(self.conf.filter_shape[ind+1][0:3]))))
                                  for ind in range(self.conf.depth-1)]
            input_shape = tf.shape(self.LR_input_t)
            self.G_layers_t = [tf.reshape(tf.transpose(tf.reshape(tf.nn.depthwise_conv2d(self.LR_input_t,self.upscaling_filter,[1,1,1,1],padding='SAME'),
                shape=tf.stack([input_shape[0],input_shape[1],input_shape[2],input_shape[3],self.ds_factor,self.ds_factor])),perm=[0,1,4,2,5,3]),
                shape=tf.stack([input_shape[0],self.ds_factor*input_shape[1],self.ds_factor*input_shape[2],input_shape[3]]))]
            self.G_layers_t = [tf.nn.leaky_relu(self.G_layers_t[0],name='layer_0')] + [None] * (self.conf.depth-1)
            for l in range(self.conf.depth - 2):
                self.G_layers_t[l + 1] = tf.nn.leaky_relu(tf.nn.conv2d(self.G_layers_t[l], self.G_filters_t[l],[1, 1, 1, 1], "SAME", name='layer_%d' % (l + 1)))
            self.G_layers_t[l+2] = tf.nn.conv2d(self.G_layers_t[l+1], self.G_filters_t[l+1],[1, 1, 1, 1], "SAME", name='layer_%d' % (l + 2))
            self.generated_im = self.G_layers_t[-1]

    def compute_inv_hTh(self):
        hTh = conv2(self.ds_kernel,np.rot90(self.ds_kernel,2))*self.ds_factor**2
        hTh = Aliased_Down_Sampling(hTh,self.ds_factor)
        pad_pre = pad_post = np.array(self.NFFT_add/2,dtype=np.int32)
        hTh_fft = np.fft.fft2(np.pad(hTh,((pad_pre,pad_post),(pad_pre,pad_post)),mode='constant',constant_values=0))
        self.inv_hTh = np.real(np.fft.ifft2(1/hTh_fft))
        max_row = np.argmax(self.inv_hTh)//self.inv_hTh.shape[0]
        max_col = np.mod(np.argmax(self.inv_hTh),self.inv_hTh.shape[0])
        if not np.all(np.equal(np.ceil(np.array(self.inv_hTh.shape)/2),np.array([max_row,max_col])-1)):
            half_filter_size = np.min([self.inv_hTh.shape[0]-max_row-1,self.inv_hTh.shape[0]-max_col-1,max_row,max_col])
            self.inv_hTh = self.inv_hTh[max_row-half_filter_size:max_row+half_filter_size+1,max_col-half_filter_size:max_col+half_filter_size+1]
        filter_energy = Return_Filter_Energy_Distribution(self.inv_hTh)
        # self.inv_hTh_invalidity_half_size = np.argwhere(filter_energy[::-1]>=self.conf.filter_pertubation_limit)[0][0]
        self.inv_hTh_invalidity_half_size = self.Return_Invalid_Margin_Size_in_LR('inv_hTh')
        if self.conf.desired_inv_hTh_energy_portion<1:
            desired_half_filter_size = np.argwhere(filter_energy>=self.conf.desired_inv_hTh_energy_portion)[-1][0]
            self.inv_hTh = self.inv_hTh[desired_half_filter_size:-desired_half_filter_size,desired_half_filter_size:-desired_half_filter_size]

    def compute_conv_with_inv_hTh_OP(self):
        self.inv_hTh_t = tf.constant(np.tile(np.expand_dims(np.expand_dims(self.inv_hTh,axis=2),axis=3),reps=[1,1,3,1]))
        self.Conv_LR_with_Inv_hTh_OP = lambda x:tf.nn.depthwise_conv2d(input=x,filter=self.inv_hTh_t,strides=[1,1,1,1],padding='SAME')
        if self.conf.pseudo_DTE_supplement:
            with self.model.as_default():
                with tf.variable_scope('Generator'):
                    self.Conv_LR_with_Learnable_OP = lambda x: tf.nn.depthwise_conv2d(input=x,
                        filter=tf.get_variable(shape=self.inv_hTh_t.get_shape(),name='pseudo_inv_hTh_filter',
                        initializer=tf.random_normal_initializer(stddev=np.sqrt(self.conf.init_variance / np.prod(self.inv_hTh_t.get_shape().as_list()[0:3])))),
                        strides=[1, 1, 1, 1],padding='SAME')

    def create_scaling_OPs(self):
        downscale_antialiasing = tf.constant(np.tile(np.expand_dims(np.expand_dims(np.rot90(self.ds_kernel,2),axis=2),axis=3),reps=[1,1,3,1]))
        upscale_antialiasing = tf.constant(np.tile(np.expand_dims(np.expand_dims(self.ds_kernel*self.ds_factor**2,axis=2),axis=3),reps=[1,1,3,1]))
        pre_stride, post_stride = calc_strides(None, self.ds_factor)
        self.Aliased_Upscale_OP = lambda x:tf.reshape(tf.pad(tf.expand_dims(tf.expand_dims(x,axis=3),axis=2),
            paddings=tf.constant([[0,0],[0,0],[pre_stride[0],post_stride[0]],[0,0],[pre_stride[1],post_stride[1]],[0,0]])),
            shape=tf.stack([tf.shape(x)[0],self.ds_factor*tf.shape(x)[1],self.ds_factor*tf.shape(x)[2],tf.shape(x)[3]]))
        self.Upscale_OP = lambda x:tf.nn.depthwise_conv2d(input=self.Aliased_Upscale_OP(x),filter=upscale_antialiasing,strides=[1,1,1,1],padding='SAME')
        Cropped_2_Integer_Factors = lambda x:tf.slice(x,begin=[0,0,0,0],
            size=tf.stack([-1,tf.cast(tf.floor(tf.shape(x)[1]/self.ds_factor)*self.ds_factor,dtype=tf.int32),tf.cast(tf.floor(tf.shape(x)[2]/self.ds_factor)*self.ds_factor,dtype=tf.int32),-1]))
        Reshaped_input = lambda x:tf.reshape(Cropped_2_Integer_Factors(x),
            shape=tf.stack([tf.shape(x)[0],tf.cast(tf.shape(x)[1]/self.ds_factor,dtype=tf.int32),self.ds_factor,tf.cast(tf.shape(x)[2]/self.ds_factor,dtype=tf.int32),self.ds_factor,tf.shape(x)[3]]))
        self.Aliased_Downscale_OP = lambda x:tf.squeeze(tf.slice(Reshaped_input(x),begin=[0,0,pre_stride[0],0,pre_stride[1],0],size=[-1,-1,1,-1,1,-1]),axis=[2,4])
        self.DownscaleOP = lambda x:self.Aliased_Downscale_OP(tf.nn.depthwise_conv2d(input=x,filter=downscale_antialiasing,strides=[1,1,1,1],padding='SAME'))
        if self.conf.pseudo_DTE_supplement:
            with self.model.as_default():
                with tf.variable_scope('Generator'):
                    self.Learnable_Upscale_OP = lambda x:tf.nn.depthwise_conv2d(input=self.Aliased_Upscale_OP(x),
                        filter=tf.get_variable(shape=upscale_antialiasing.get_shape(),name='pseudo_upscale_filter',
                        initializer=tf.random_normal_initializer(stddev=np.sqrt(self.conf.init_variance / np.prod(upscale_antialiasing.get_shape().as_list()[0:3])))),strides=[1,1,1,1],padding='SAME')
                    self.Learnable_DownscaleOP = lambda x:self.Aliased_Downscale_OP(tf.nn.depthwise_conv2d(input=x,
                        filter=tf.get_variable(shape=downscale_antialiasing.get_shape(),name='pseudo_downscale_filter',
                        initializer=tf.random_normal_initializer(stddev=np.sqrt(self.conf.init_variance / np.prod(downscale_antialiasing.get_shape().as_list()[0:3])))),strides=[1,1,1,1],padding='SAME'))


class DTE_PyTorch(nn.Module):
    def __init__(self, DTEnet, generated_image):
        super(DTE_PyTorch, self).__init__()
        self.ds_factor = DTEnet.ds_factor
        self.generated_image_model = generated_image
        inv_hTh_t = torch.from_numpy(np.tile(np.expand_dims(np.expand_dims(DTEnet.inv_hTh,0),0),reps=[3,1,1,1])).type(torch.cuda.FloatTensor)
        inv_hTh_padding = np.floor(np.array(DTEnet.inv_hTh.shape)/2).astype(np.int32)
        Replication_Padder = nn.ReplicationPad2d((inv_hTh_padding[1],inv_hTh_padding[1],inv_hTh_padding[0],inv_hTh_padding[0]))
        self.Conv_LR_with_Inv_hTh_OP = lambda x: nn.functional.conv2d(Replication_Padder(x), weight= inv_hTh_t, groups=3)
        downscale_antialiasing = torch.from_numpy(np.tile(np.expand_dims(np.expand_dims(np.rot90(DTEnet.ds_kernel,2),axis=0),axis=0),reps=[3,1,1,1])).type(torch.cuda.FloatTensor)
        upscale_antialiasing = torch.from_numpy(np.tile(np.expand_dims(np.expand_dims(DTEnet.ds_kernel*DTEnet.ds_factor**2,axis=0),axis=0),reps=[3,1,1,1])).type(torch.cuda.FloatTensor)
        pre_stride, post_stride = calc_strides(None, DTEnet.ds_factor)
        Upscale_Padder = lambda x: nn.functional.pad(x,(pre_stride[1],post_stride[1],0,0,pre_stride[0],post_stride[0]))
        Aliased_Upscale_OP = lambda x:Upscale_Padder(x.unsqueeze(4).unsqueeze(3)).view([x.size()[0],x.size()[1],DTEnet.ds_factor*x.size()[2],DTEnet.ds_factor*x.size()[3]])
        antialiasing_padding = np.floor(np.array(DTEnet.ds_kernel.shape)/2).astype(np.int32)
        antialiasing_Padder = nn.ReplicationPad2d((antialiasing_padding[1],antialiasing_padding[1],antialiasing_padding[0],antialiasing_padding[0]))
        self.Upscale_OP = lambda x:nn.functional.conv2d(input=antialiasing_Padder(Aliased_Upscale_OP(x)),weight=upscale_antialiasing,groups=3)
        Reshaped_input = lambda x:x.view([x.size()[0],x.size()[1],int(x.size()[2]/self.ds_factor),self.ds_factor,int(x.size()[3]/self.ds_factor),self.ds_factor])
        Aliased_Downscale_OP = lambda x:Reshaped_input(x)[:,:,:,pre_stride[0],:,pre_stride[1]]
        self.DownscaleOP = lambda x:Aliased_Downscale_OP(nn.functional.conv2d(input=antialiasing_Padder(x),weight=downscale_antialiasing,groups=3))
    def forward(self, x):
        generated_image = self.generated_image_model(x)
        assert np.all(np.mod(generated_image.size()[2:],self.ds_factor)==0)
        projected_upscaled_input = self.Upscale_OP(self.Conv_LR_with_Inv_hTh_OP(x))
        projected_generated_im = self.Upscale_OP(self.Conv_LR_with_Inv_hTh_OP(self.DownscaleOP(generated_image)))
        projected_2_ortho_generated_im = generated_image - projected_generated_im
        return projected_upscaled_input+projected_2_ortho_generated_im

def Aliased_Down_Sampling(array,factor):
    pre_stride,post_stride = calc_strides(array,1/factor,align_center=True)
    if array.ndim>2:
        array = array[pre_stride[0]::factor,pre_stride[1]::factor,:]
    else:
        array = array[pre_stride[0]::factor, pre_stride[1]::factor]
    return array

def Return_Downscale_OP(ds_factor,ds_kernel=None):
    if ds_kernel is None:
        ds_kernel = Return_Default_kernel(ds_factor)
    downscale_antialiasing = tf.constant(np.tile(np.expand_dims(np.expand_dims(np.rot90(ds_kernel, 2), axis=2), axis=3), reps=[1, 1, 3, 1]))
    pre_stride, post_stride = calc_strides(None, ds_factor)
    # Cropped_2_Integer_Factors = lambda x: tf.slice(x, begin=[0, 0, 0, 0],size=tf.stack([-1, tf.cast(
    #     tf.floor(tf.shape(x)[1] / ds_factor) * ds_factor,dtype=tf.int32), tf.cast(tf.floor(tf.shape(x)[2] / ds_factor) * ds_factor,dtype=tf.int32), -1]))
    ds_factor_float_t = tf.constant(ds_factor,dtype=tf.float32)
    input_shape_assertion = lambda x: tf.Assert(condition=tf.logical_and(tf.equal(tf.round(tf.cast(tf.shape(x)[1],tf.float32) / ds_factor_float_t),tf.cast(tf.shape(x)[1],tf.float32) / ds_factor_float_t),
        tf.equal(tf.round(tf.cast(tf.shape(x)[2],tf.float32) / ds_factor_float_t),tf.cast(tf.shape(x)[2],tf.float32) / ds_factor_float_t)),data=[tf.shape(x)])
    # Reshaped_input = lambda x: tf.reshape(Cropped_2_Integer_Factors(x),shape=tf.stack(
    #     [tf.shape(x)[0], tf.cast(tf.shape(x)[1] / ds_factor, dtype=tf.int32),ds_factor, tf.cast(tf.shape(x)[2] / ds_factor, dtype=tf.int32),ds_factor, tf.shape(x)[3]]))
    ds_factor_int_t = tf.constant(int(ds_factor),dtype=tf.int32)
    Reshaped_input = lambda x: tf.reshape(x,shape=tf.stack(
        [tf.shape(x)[0], tf.cast(tf.shape(x)[1] / ds_factor_int_t, dtype=tf.int32),ds_factor_int_t, tf.cast(tf.shape(x)[2] / ds_factor_int_t, dtype=tf.int32),ds_factor_int_t, tf.shape(x)[3]]))
    Aliased_Downscale_OP = lambda x: tf.squeeze(tf.slice(Reshaped_input(x), begin=[0, 0, pre_stride[0], 0, pre_stride[1], 0], size=[-1, -1, 1, -1, 1, -1]),axis=[2, 4])
    def Downscale_OP(input):
        shape_assertion = input_shape_assertion(input)
        with tf.control_dependencies([shape_assertion]):
            return Aliased_Downscale_OP(tf.nn.depthwise_conv2d(input=input, filter=downscale_antialiasing, strides=[1, 1, 1, 1], padding='SAME'))
    return Downscale_OP

def Aliased_Down_Up_Sampling(array,factor):
    half_stride_size = np.floor(factor/2).astype(np.int32)
    input_shape = list(array.shape)
    if array.ndim>2:
        array = array[half_stride_size:-half_stride_size:factor,half_stride_size:-half_stride_size:factor,:]
    else:
        array = array[half_stride_size:-half_stride_size:factor, half_stride_size:-half_stride_size:factor]
    array = np.expand_dims(np.expand_dims(array,2),1)
    array = np.pad(array,((0,0),(half_stride_size,half_stride_size),(0,0),(half_stride_size,half_stride_size)),mode='constant')
    return np.reshape(array,newshape=input_shape)


def Return_Default_kernel(ds_factor):
    return np.rot90(imresize(None, [ds_factor, ds_factor], return_upscale_kernel=True), 2).astype(np.float32) / (ds_factor ** 2)
#     DELTA_SIZE = 11
#     delta_im = np.zeros([DELTA_SIZE,DELTA_SIZE])
#     delta_im[np.ceil(DELTA_SIZE/2).astype(np.int32)-1,np.ceil(DELTA_SIZE/2).astype(np.int32)-1] = 1
#     kernel = imresize(delta_im,scale_factor=[ds_factor,ds_factor],align_center=True)/(ds_factor**2)
#     kernel_support = np.nonzero(kernel[ds_factor*np.ceil(DELTA_SIZE/2).astype(np.int32)-1,:])[0]
#     return kernel[kernel_support[0]:kernel_support[-1]+1,kernel_support[0]:kernel_support[-1]+1]

def Return_Filter_Energy_Distribution(filter):
    sqrt_energy =  [np.sqrt(np.sum(filter**2))]+[np.sqrt(np.sum(filter[frame_num:-frame_num,frame_num:-frame_num]**2)) for frame_num in range(1,int(np.ceil(filter.shape[0]/2)))]
    return sqrt_energy/sqrt_energy[0]

def Pad_Image(image,margin_size):
    return np.pad(image,pad_width=((margin_size,margin_size),(margin_size,margin_size),(0,0)),mode='edge')

def Unpad_Image(image,margin_size):
    return image[margin_size:-margin_size,margin_size:-margin_size,:]

def Create_Tensor_Pad_OP(padding_size):
    pad_size_list = [1]
    while sum(pad_size_list)<padding_size:
        pad_size_list.append(np.minimum(1+sum(pad_size_list),padding_size-sum(pad_size_list)))
    def TF_Pad_OP(x):
        for cur_pad_size in pad_size_list:
            x = tf.pad(x, paddings=tf.constant([[0, 0], [cur_pad_size, cur_pad_size], [cur_pad_size, cur_pad_size], [0, 0]]),mode='SYMMETRIC')
        return x
    return TF_Pad_OP

def Get_DTE_Conf(sf):
    class conf:
        scale_factor = sf
        avoid_skip_connections = False
        generate_HR_image = False
        pseudo_DTE_supplement = False
        desired_inv_hTh_energy_portion = 1 - 1e-10
        filter_pertubation_limit = 0.999
    return conf
