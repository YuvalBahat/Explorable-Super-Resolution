import cv2
import numpy as np
from scipy.signal import convolve2d as conv2

def imresize(im, scale_factor=None, output_shape=None, kernel=None,align_center=False, return_upscale_kernel=False,use_zero_padding=False,antialiasing=True, kernel_shift_flag=False):
    assert kernel is None or kernel=='cubic' or isinstance(kernel,np.ndarray)
    imresize.kernels = getattr(imresize,'kernels',{})
    # imresize.sf = getattr(imresize,'sf',0)
    if scale_factor is None:
        scale_factor = [output_shape[0]/im.shape[0]]
    sf_4_kernel = np.maximum(scale_factor[0], 1 / scale_factor[0]).astype(np.int32)
    if isinstance(kernel,np.ndarray):
        assert str(sf_4_kernel) not in imresize.kernels.keys() or np.all(np.equal(kernel,imresize.kernels[str(sf_4_kernel)])),'If using non-default kernel, make sure I always use it.'
        imresize.kernels[str(sf_4_kernel)] = kernel
    elif str(sf_4_kernel) not in imresize.kernels.keys():
        DELTA_SIZE = 11
        # imresize.sf = sf_4_kernel
        delta_im = np.zeros([DELTA_SIZE, DELTA_SIZE])
        delta_im[np.ceil(DELTA_SIZE / 2).astype(np.int32) - 1, np.ceil(DELTA_SIZE / 2).astype(np.int32) - 1] = 1
        upscale_kernel = cv2.resize(delta_im,dsize=(sf_4_kernel*DELTA_SIZE,sf_4_kernel*DELTA_SIZE),interpolation=cv2.INTER_CUBIC)
        kernel_support = np.nonzero(upscale_kernel[sf_4_kernel * np.ceil(DELTA_SIZE / 2).astype(np.int32) - 1, :])[0]
        kernel_support = np.array([kernel_support[0],kernel_support[-1]])
        imresize.kernels[str(sf_4_kernel)] = upscale_kernel[kernel_support[0]:kernel_support[1] + 1, kernel_support[0]:kernel_support[1] + 1]
    # else:
    #     assert np.all(scale_factor==imresize.sf) or np.all(scale_factor==1/imresize.sf)
    assert len(scale_factor)==1 or scale_factor[0]==scale_factor[1]
    scale_factor = scale_factor[0]
    pre_stride,post_stride = calc_strides(im,scale_factor,align_center)
    kernel_post_padding = np.maximum(0,pre_stride-post_stride)
    kernel_pre_padding = np.maximum(0,post_stride-pre_stride)
    antialiasing_kernel = np.pad(imresize.kernels[str(sf_4_kernel)],
                                 ((kernel_pre_padding[0],kernel_post_padding[0]),(kernel_pre_padding[1],kernel_post_padding[1])),mode='constant')
    if scale_factor < 1:
        antialiasing_kernel = np.rot90(antialiasing_kernel * scale_factor ** 2, 2)
    if return_upscale_kernel:
        return antialiasing_kernel
    assert output_shape is None or np.all(scale_factor*np.array(im.shape[:2])==output_shape[:2])
    padding_size = np.floor(np.array(antialiasing_kernel.shape)/2).astype(np.int32)
    desired_size = scale_factor*np.array(im.shape[:2])
    assert np.all(desired_size==np.round(desired_size))
    desired_size = desired_size.astype(np.int32)
    if im.ndim<3:
        im = np.expand_dims(im,-1)
    output = []
    for channel_num in range(im.shape[2]):
        if scale_factor>1:#Upscale
            output.append(np.reshape(np.pad(np.expand_dims(np.expand_dims(im[:,:,channel_num],2),1),((0,0),(pre_stride[0],post_stride[0]),(0,0),(pre_stride[1],post_stride[1])),
                mode='constant'),newshape=desired_size))
            if use_zero_padding:
                output[-1] = conv2(output[-1],antialiasing_kernel,mode='same')
            else:
                output[-1] = conv2(np.pad(output[-1],pad_width=((padding_size[0],padding_size[0]),(padding_size[1],padding_size[1])),mode='edge'),antialiasing_kernel,mode='valid')
        else:
            if use_zero_padding:
                output.append(conv2(im[:,:,channel_num],antialiasing_kernel,mode='same'))
            else:
                output.append(conv2(np.pad(im[:,:,channel_num],pad_width=((padding_size[0],padding_size[0]),(padding_size[1],padding_size[1])),mode='edge'),
                                    antialiasing_kernel,mode='valid'))
            output[-1] = output[-1][pre_stride[0]::int(1 / scale_factor),pre_stride[1]::int(1 / scale_factor)]
    return np.squeeze(np.stack(output,-1))
    # return cv2.resize(im,dsize=tuple(desired_size.astype(np.int32)[::-1]),interpolation=cv2.INTER_CUBIC)
def calc_strides(array,factor,align_center = False):
    if align_center:
        half_image_size = np.ceil(np.array(array.shape[:2])/2*(factor if factor>1 else 1))
        pre_stride = np.mod(half_image_size,np.maximum(factor,1/factor))
        pre_stride[np.equal(pre_stride,0)] = np.maximum(factor,1/factor)
        pre_stride = (pre_stride-1).astype(np.int32)
        post_stride = np.maximum(factor,1/factor).astype(np.int32)-pre_stride-1
    else:
        post_stride = (np.floor(np.maximum(factor,1/factor)/2)*np.ones([2])).astype(np.int32)
        pre_stride = (np.maximum(factor,1/factor)-post_stride-1).astype(np.int32)
    return pre_stride,post_stride
