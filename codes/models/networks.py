import functools
import torch
import torch.nn as nn
from torch.nn import init

import models.modules.architecture as arch
import models.modules.sft_arch as sft_arch

####################
# initialize
####################


def weights_init_normal(m, std=0.02):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, std)  # BN also uses norm
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m, scale=1):
    if 'filter_layer' in m.__dict__ and m.__getattribute__('filter_layer'):
        return
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    # elif any([norm_type in classname for norm_type in ['BatchNorm2d','LayerNorm']]):
    elif any([norm_type in classname for norm_type in ['BatchNorm2d']]):
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='kaiming', scale=1, std=0.02):
    # scale for 'kaiming', std for 'normal'.
    print('initialization method [{:s}]'.format(init_type))
    if init_type == 'normal':
        weights_init_normal_ = functools.partial(weights_init_normal, std=std)
        net.apply(weights_init_normal_)
    elif init_type == 'kaiming':
        weights_init_kaiming_ = functools.partial(weights_init_kaiming, scale=scale)
        net.apply(weights_init_kaiming_)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [{:s}] not implemented'.format(init_type))


####################
# define network
####################


# Generator
def define_G(opt,CEM=None,num_latent_channels=None,**kwargs):
    gpu_ids = opt['gpu_ids']
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']
    opt_net['latent_input'] = opt_net['latent_input'] if opt_net['latent_input']!="None" else None

    if which_model == 'sr_resnet':  # SRResNet
        netG = arch.SRResNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], nf=opt_net['nf'], \
            nb=opt_net['nb'], upscale=opt_net['scale'], norm_type=opt_net['norm_type'], \
            act_type='relu', mode=opt_net['mode'], upsample_mode='pixelshuffle',range_correction=opt_net['range_correction'])

    elif which_model == 'RRDB_net':  # RRDB
        netG = arch.RRDBNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], nf=opt_net['nf'],
            nb=opt_net['nb'], gc=opt_net['gc'], upscale=opt_net['scale'], norm_type=opt_net['norm_type'],
            act_type='leakyrelu', mode=opt_net['mode'], upsample_mode='upconv',
            latent_input=(opt_net['latent_input']+'_'+opt_net['latent_input_domain']) if opt_net['latent_input'] is not None else None,num_latent_channels=num_latent_channels)
    elif which_model == 'DnCNN':
        DCT_G = opt_net['DCT_G']
        chroma_mode = kwargs['chroma_mode'] if 'chroma_mode' in kwargs.keys() else False
        in_nc = (opt['scale']**2+2*64 if chroma_mode else 64) if DCT_G else (3 if chroma_mode else 1)
        if 'no_high_freq_chroma_reconstruction' not in kwargs:
            kwargs['no_high_freq_chroma_reconstruction'] = True
        # if kwargs['no_high_freq_chroma_reconstruction'] and DCT_G:
        #     print('Warning: Using high frequency chroma reconstruction, since avoiding it is not yet supported for DCT generators.')
        out_nc = ((2*64 if kwargs['no_high_freq_chroma_reconstruction'] else 2*256) if chroma_mode else 64) if DCT_G else (3 if chroma_mode else 1)
        # out_nc = (2*(64 if kwargs['no_high_freq_chroma_reconstruction'] else opt['scale']**2) if chroma_mode else 64) if DCT_G else (3 if chroma_mode else 1)
        netG = arch.DnCNN(n_channels=opt_net['nf'],depth=opt_net['nb'],in_nc=in_nc,out_nc=out_nc,norm_type=opt_net['norm_type'],
                          latent_input=opt_net['latent_input'] if opt_net['latent_input'] is not None else None,
                          num_latent_channels=num_latent_channels,chroma_generator=chroma_mode,DCT_G=DCT_G,norm_input=opt_net['normalize_input'],
                          coordinates_input=opt['scale'] if opt_net['coordinates_input'] else None,avoid_padding=not bool(opt_net['padding']),
                          residual=opt_net['residual'])
    elif which_model == 'MSRResNet':  # SRResNet
        netG = arch.MSRResNet(in_nc=opt_net['in_nc'], out_nc=opt_net['out_nc'], nf=opt_net['nf'], \
                             nb=opt_net['nb'], upscale=opt_net['scale'])
    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))
    if opt_net['CEM_arch']:
        netG = CEM.WrapArchitecture_PyTorch(netG,opt['datasets']['train']['patch_size'] if opt['is_train'] else None)
    if opt['is_train'] and which_model != 'MSRResNet':# and which_model != 'DnCNN':
        init_weights(netG, init_type='kaiming', scale=0.1)
    if gpu_ids:
        assert torch.cuda.is_available()
        netG = nn.DataParallel(netG)
    return netG

# Discriminator
def define_D(opt,CEM=None,**kwargs):
    gpu_ids = opt['gpu_ids']
    opt_net = opt['network_D']
    which_model = opt_net['which_model_D']
    input_patch_size = opt['datasets']['train']['patch_size']
    # in_nc = opt_net['in_nc']*(2 if opt['network_D']['decomposed_input'] else 1)
    in_nc = opt_net['in_nc']
    assert not ((opt_net['pre_clipping'] or opt_net['decomposed_input']) and which_model!='PatchGAN'),'Unsupported yet'
    if CEM is not None:
        input_patch_size -= 2*CEM.invalidity_margins_HR
    if which_model == 'discriminator_vgg_128':
        kwargs = {}
        if 'num_2_strides' in opt_net:
            kwargs['num_2_strides'] = opt_net['num_2_strides']
        netD = arch.Discriminator_VGG_128(in_nc=in_nc, base_nf=opt_net['nf'], nb=opt_net['n_layers'],
            norm_type=opt_net['norm_type'], mode=opt_net['mode'], act_type=opt_net['act_type'],input_patch_size=input_patch_size,**kwargs)
    elif which_model == 'discriminator_vgg_128_nonModified':
        netD = arch.Discriminator_VGG_128_nonModified(in_nc=in_nc, nf=opt_net['nf'])
    elif which_model == 'dis_acd':  # sft-gan, Auxiliary Classifier Discriminator
        netD = sft_arch.ACD_VGG_BN_96()
    elif which_model=='PatchGAN':
        norm_layer = nn.BatchNorm2d
        if 'gp' in opt['train']['gan_type']:
            norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
        netD = arch.PatchGAN_Discriminator(input_nc=in_nc, opt_net=opt_net,ndf=opt_net['nf'], n_layers=opt_net['n_layers'], norm_layer=norm_layer)
    elif which_model == 'discriminator_vgg_96':
        netD = arch.Discriminator_VGG_96(in_nc=in_nc, base_nf=opt_net['nf'], \
            norm_type=opt_net['norm_type'], mode=opt_net['mode'], act_type=opt_net['act_type'])
    elif which_model == 'discriminator_vgg_192':
        netD = arch.Discriminator_VGG_192(in_nc=in_nc, base_nf=opt_net['nf'], \
            norm_type=opt_net['norm_type'], mode=opt_net['mode'], act_type=opt_net['act_type'])
    elif which_model == 'discriminator_vgg_128_SN':
        netD = arch.Discriminator_VGG_128_SN()
    elif 'DnCNN_D' in which_model:
        chroma_mode = kwargs['chroma_mode'] if 'chroma_mode' in kwargs.keys() else False
        opt_net_G = opt['network_G']
        G_in_nc = (opt['scale'] ** 2 + 2 * 64 if chroma_mode else 64) if opt_net_G['DCT_G'] else (3 if chroma_mode else 1)
        if 'no_high_freq_chroma_reconstruction' not in kwargs:
            kwargs['no_high_freq_chroma_reconstruction'] = True
        if 'DCT' in opt_net['input_type']:
            G_out_nc = 2*(64 if kwargs['no_high_freq_chroma_reconstruction'] else 256) if chroma_mode else 64
        else:
            # assert not chroma_mode,'Unsupported yet'
            assert not (opt_net_G['coordinates_input'] and not opt_net['concat_input']),'Should decide what to do in this case'
            # assert not opt_net['concat_input'],'To support, need to figure how to concat these inputs with different spatial dimensions.'
            assert (not opt_net_G['DCT_G']) or (not opt_net['concat_input']) ,'Unsupported yet'
            G_out_nc = 3 if chroma_mode else 1
        norm_type = opt_net_G['norm_type'] if opt_net['norm_type'] is None else opt_net['norm_type']
        # Even when not in concat_inpiut mode, I'm supplying D with channel Y, so it does not need to determine realness based only on the chroma channels
        D_input_channels = G_in_nc+G_out_nc if opt_net['concat_input'] else (opt['scale']**2+G_out_nc if chroma_mode else G_out_nc)
        num_latent_channels = None
        if opt_net['inject_Z']:
            num_latent_channels = opt_net_G['latent_channels']
            # D_input_channels += num_latent_channels
        netD = arch.DnCNN(n_channels=opt_net_G['nf'] if opt_net['nf'] is None else opt_net['nf'],
            depth=opt_net_G['nb'] if opt_net['nb'] is None else opt_net['nb'],in_nc=D_input_channels,num_kerneled_layers=opt_net['nk'],
            norm_type='layer' if (opt['train']['gan_type']=='wgan-gp' and norm_type=='batch') else norm_type,
            discriminator=True,expected_input_size=opt['datasets']['train']['patch_size']//(opt['scale'] if 'DCT' in opt_net['input_type'] else 1),
            latent_input=opt_net_G['latent_input'],num_latent_channels=num_latent_channels,chroma_generator=False,spectral_norm='sn' in opt['train']['gan_type'],
            pooling_no_FC=opt_net['pooling_no_fc'],norm_input=opt_net_G['normalize_input'] if opt_net['normalize_input'] is None else opt_net['normalize_input'],
            coordinates_input=opt['scale'] if opt_net_G['coordinates_input'] else None)
    else:
        raise NotImplementedError('Discriminator model [{:s}] not recognized'.format(which_model))

    init_weights(netD, init_type='kaiming', scale=1)
    if gpu_ids:
        netD = nn.DataParallel(netD)
    return netD


def define_F(opt, use_bn=False,**kwargs):
    gpu_ids = opt['gpu_ids']
    device = torch.device('cuda' if gpu_ids else 'cpu')
    # pytorch pretrained VGG19-54, before ReLU.
    if use_bn:
        feature_layer = 49
    else:
        feature_layer = 34
    if 'arch' in kwargs.keys() and 'vgg' in kwargs['arch']:
        if len(kwargs['arch'])>len('vgg11_'):
            feature_layer = int(kwargs['arch'][len('vgg11_'):])
        kwargs['arch'] = kwargs['arch'][:len('vgg11')]
    netF = arch.VGGFeatureExtractor(feature_layer=feature_layer, use_bn=use_bn,use_input_norm=True, device=device,**kwargs)
    # netF = arch.ResNet101FeatureExtractor(use_input_norm=True, device=device)
    if gpu_ids:
        netF = nn.DataParallel(netF)
    netF.eval()  # No need to train
    return netF