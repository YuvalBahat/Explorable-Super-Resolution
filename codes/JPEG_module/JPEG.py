import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

FACTORIZE_CHROMA_HIGH_FREQS = True # When True, Padding chroma Q-table with edge values to multiply the reconstructed high frequencies, allowing for more energy in these frequencies,
# due to the Sigmoid that limits the generator's output range.

LUMINANCE_QUANTIZATION_TABLE = np.array((
    (16, 11, 10, 16, 24, 40, 51, 61),
    (12, 12, 14, 19, 26, 58, 60, 55),
    (14, 13, 16, 24, 40, 57, 69, 56),
    (14, 17, 22, 29, 51, 87, 80, 62),
    (18, 22, 37, 56, 68, 109, 103, 77),
    (24, 36, 55, 64, 81, 104, 113, 92),
    (49, 64, 78, 87, 103, 121, 120, 101),
    (72, 92, 95, 98, 112, 100, 103, 99)
))

CHROMINANCE_QUANTIZATION_TABLE = np.array((
    (17, 18, 24, 47, 99, 99, 99, 99),
    (18, 21, 26, 66, 99, 99, 99, 99),
    (24, 26, 56, 99, 99, 99, 99, 99),
    (47, 66, 99, 99, 99, 99, 99, 99),
    (99, 99, 99, 99, 99, 99, 99, 99),
    (99, 99, 99, 99, 99, 99, 99, 99),
    (99, 99, 99, 99, 99, 99, 99, 99),
    (99, 99, 99, 99, 99, 99, 99, 99)
))


class JPEG(nn.Module):
    def __init__(self,compress,downsample_and_quantize=None,chroma_mode=False,block_size=8):
        # Performing JPEG-like DCT transformation when compress, or inverse DCT transform when not compress.
        #
        # For compress: Expecting input torch tensor of size [batch_size,num_channels,H,W], where currently only supporting num_channels==1. Returning a tensor of size [batch_size,64,H/8,W/8],
        # where dimension 1 corresponds to the DCT coefficients.
        # For not compress (extract): Expected input torch tensor size corresponds to size of output tensor of compress, and vice versa.
        # H and W should currently both be integer multiplications of 8.

        super(JPEG, self).__init__()
        # self.quality_factor =quality_factor
        self.compress = compress # Compression or extraction module
        self.downsample_and_quantize = downsample_and_quantize
        assert (compress ^ (downsample_and_quantize is None)),'Quantize argument should be passed iff in compress mode'
        assert FACTORIZE_CHROMA_HIGH_FREQS,'No longer supproting the other option, after dividing the Y channel high freq. coeffs. as well, as I think it would ease the generator''s mapping.'
        self.device = 'cuda'
        self.block_size = block_size
        self.chroma_mode = chroma_mode
        def process_Q_table(Q_table):
            return torch.from_numpy(Q_table / 100).view(1, Q_table.shape[0], Q_table.shape[1], 1, 1).type(torch.FloatTensor).to(self.device)
        # Following the procedure in the Prototype_jpeg repository to convert quality factor to quantization table:
        # self.Q_table = torch.from_numpy(LUMINANCE_QUANTIZATION_TABLE / 100).view(1, 8, 8, 1, 1).type(torch.FloatTensor).to(self.device)
        self.Q_table = process_Q_table(LUMINANCE_QUANTIZATION_TABLE)
        if self.chroma_mode:
            self.Q_table = torch.cat([self.Q_table.unsqueeze(1),process_Q_table(CHROMINANCE_QUANTIZATION_TABLE).unsqueeze(1).repeat([1,2,1,1,1,1])],1)
            if FACTORIZE_CHROMA_HIGH_FREQS:
                self.padded_Q_table = torch.cat([process_Q_table(np.pad(LUMINANCE_QUANTIZATION_TABLE,((0,self.block_size-8),(0,self.block_size-8)),'edge')).unsqueeze(1),
                    process_Q_table(np.pad(CHROMINANCE_QUANTIZATION_TABLE,((0,self.block_size-8),(0,self.block_size-8)),'edge')).unsqueeze(1).repeat([1,2,1,1,1,1])],1)

    #     For DCT:
        self.DCT_freq_grid = (np.pi * torch.arange(block_size).type(torch.FloatTensor) /2/block_size).view(1, 1, 1, 1, 1,block_size).to(self.device)
        self.iDCT_freq_grid = (np.pi * (torch.arange(block_size).type(torch.FloatTensor)+0.5) / block_size).view(1, 1, 1, 1, 1, block_size).to(self.device)
        # self.grid_factors = 2*torch.cat([1/np.sqrt(4*8)*torch.ones([1]),1/np.sqrt(2*8)*torch.ones([7])]).type(torch.FloatTensor).view(1,1,1,1,8).to(self.device)
        self.grid_factors = 2*torch.cat([1/np.sqrt(4*block_size)*torch.ones([1]),1/np.sqrt(2*block_size)*torch.ones([block_size-1])]).type(torch.FloatTensor).view(1,1,1,1,block_size).to(self.device)
        self.DCT_arange = torch.arange(block_size).type(torch.FloatTensor).to(self.device)
        self.iDCT_arange = torch.arange(1,block_size).type(torch.FloatTensor).to(self.device)

    def Set_QF(self,quality_factor):
        # Following the procedure in the Prototype_jpeg repository to convert quality factor to quantization table:
        self.QF = quality_factor
        condition = (quality_factor < 50).to(self.device).type(torch.LongTensor)
        self.factor = (condition*(5000 / quality_factor) + (1-condition)*(200 - 2 * quality_factor))
        self.factor = self.factor.view([-1,1,1,1,1]+([1] if self.chroma_mode else [])).type(self.Q_table.dtype).to(self.device)

    def Image_2_Blocks(self,image):
        image_shape = list(image.size())
        if self.chroma_mode:
            return image.view(image_shape[:2]+[image_shape[2] // self.block_size, self.block_size, image_shape[3] // self.block_size,self.block_size]).permute(0, 1,3, 5, 2, 4)
        else:
            return image.view([image_shape[0],image_shape[2]//self.block_size,self.block_size,image_shape[3]//self.block_size,self.block_size]).permute(0,2,4,1,3)

    def Blocks_2_Image(self,blocks):
        blocks_shape = list(blocks.size())
        return blocks.permute(0,3,1,4,2).contiguous().view([blocks_shape[0],1]+[blocks_shape[3]*self.block_size,blocks_shape[4]*self.block_size])

    def DCT(self,blocks,axis):
        output = self.grid_factors*(torch.cos(self.DCT_freq_grid * (2 * self.DCT_arange.view([1] * (axis) + [self.block_size] + [1] * (5 - axis)) + 1)) * blocks.unsqueeze(-1)).sum(axis)
        if axis<(blocks.dim()-1): #Moving the resulting frequency axis, currently in the last dim, to its original axis:
            output = output.permute([i for i in range(axis)]+[blocks.dim()-1]+[i for i in range(axis,blocks.dim()-1)])
        return output

    def iDCT(self,blocks,axis):
        output =  (torch.cos(self.iDCT_freq_grid*self.iDCT_arange.view([1] * (axis) + [self.block_size-1] + [1] * (5 - axis)))*
                torch.index_select(blocks,dim=axis,index=self.iDCT_arange.type(torch.cuda.LongTensor)).unsqueeze(-1)).sum(axis)*np.sqrt(2/self.block_size)+\
                  torch.index_select(blocks,dim=axis,index=torch.zeros([1]).type(torch.cuda.LongTensor)).squeeze(axis).unsqueeze(-1)/np.sqrt(self.block_size)
        if axis<(blocks.dim()-1): #Moving the resulting temporal/spatial axis, currently in the last dim, to its original axis:
            output = output.permute([i for i in range(axis)]+[blocks.dim()-1]+[i for i in range(axis,blocks.dim()-1)])
        return output


    def Blocks_DCT(self,blocks):
        return self.DCT(self.DCT(blocks,axis=1),axis=2)

    def Blocks_iDCT(self,blocks):
        return self.iDCT(self.iDCT(blocks, axis=1), axis=2)

    def forward(self, input):
        input = input.to(self.device)
        if self.compress: #Input is an image:
            output = self.Image_2_Blocks(input)
            if self.chroma_mode:
                output = output-torch.tensor([128.,0,0]).to(input.device).view(1,3,1,1,1,1)
                output = output.view([output.size(0)*output.size(1)]+list(output.size())[2:])
            else:
                output = output-128
            output = self.Blocks_DCT(output)
            if self.chroma_mode:
                output = output.view([input.size(0),3,self.block_size,self.block_size,output.size(3),output.size(4)])
                output = output/(self.factor*self.padded_Q_table)
                output = output.view([input.size(0),3,self.block_size//8,8,self.block_size//8,8,output.size(4),output.size(5)])
                # output[:,1:,0,:,0,:,:,:] = output[:,1:,0,:,0,:,:,:]/(self.factor*self.Q_table)
                # output = output/(self.factor*self.Q_table.unsqueeze(2).unsqueeze(4))
                if self.downsample_and_quantize:
                    # Downsampling is done by wiping out DCT coefficients corresponding to higher frequencies (8 and up) in the chroma channels.
                    # Not quantizing y channel coefficients in this case. I'm going to discard these chroma high-frequency coefficients later anyway,
                    # so no need to wipe them out here.
                    output[:, 1:, 0, :, 0, :, :, :] = torch.round(output[:, 1:, 0, :, 0, :, :, :])
                    output = torch.cat([output[:,0,...].contiguous().view(output.size(0),self.block_size**2,output.size(6),output.size(7)),
                                          output[:,1,0,:,0,...].contiguous().view(output.size(0),8**2,output.size(6),output.size(7)),
                                          output[:,2,0,:,0,...].contiguous().view(output.size(0),8**2,output.size(6),output.size(7))],1)
                else:
                    output = output.contiguous().view(output.size(0), 3, self.block_size ** 2, output.size(6), output.size(7))
                    output = torch.cat([output[:,0,...],output[:,1,...],output[:,2,...]],1)
            else:
                output = output/(self.factor*self.Q_table)
                if self.downsample_and_quantize:
                    output = torch.round(output)
                output = output.contiguous().view([input.size(0),self.block_size**2,input.size(2)//self.block_size,input.size(3)//self.block_size])
        else:# Extraction: Input is DCT blocks
            if self.chroma_mode:
                if input.size(1)==2*self.block_size**2: #Input is the 2 full chroma channels:
                    num_channels = 2
                    if FACTORIZE_CHROMA_HIGH_FREQS:
                        output = input.view([input.size(0),2,self.block_size,self.block_size, input.size(2), input.size(3)])
                    else:
                        output = input.view([input.size(0),2, self.block_size//8,8,self.block_size//8,8, input.size(2), input.size(3)])
                elif input.size(1)==(self.block_size**2+2*8**2): #Input is the full Y channel and low frequencies of the chroma channels (the input to the generator):
                    num_channels = 3
                    chroma_padding_arg = [0,0,0,0,0,0,0,self.block_size//8-1,0,0,0,self.block_size//8-1]
                    output = torch.stack([input[:,:self.block_size**2,...].view(input.size(0),self.block_size//8,8,self.block_size//8,8, input.size(2), input.size(3)),
                        F.pad(input[:,self.block_size**2:self.block_size**2+8**2,...].view(input.size(0),1,8,1,8, input.size(2), input.size(3)),chroma_padding_arg),
                        F.pad(input[:,self.block_size**2+8**2:self.block_size**2+2*(8**2),...].view(input.size(0),1,8,1,8, input.size(2), input.size(3)),chroma_padding_arg)],1)
                    if FACTORIZE_CHROMA_HIGH_FREQS:
                        output = output.view(output.size(0),output.size(1),self.block_size,self.block_size,input.size(2),input.size(3))
                else:
                    raise Exception('Unexpected input size')

                if FACTORIZE_CHROMA_HIGH_FREQS:
                    output = output * self.factor * self.padded_Q_table[:,-num_channels:,...]
                else:
                    output[:,:,0,:,0,...] = output[:,:,0,:,0,...]* self.factor * self.Q_table[:,-num_channels:,...]
                output = output.view(output.size(0)*num_channels,self.block_size,self.block_size,input.size(2),input.size(3))
            else:
                output = input.view([input.size(0),8,8,input.size(2),input.size(3)])*self.factor*self.Q_table
            output = self.Blocks_iDCT(output)
            if not self.chroma_mode:
                output += 128
            output = self.Blocks_2_Image(output)
            if self.chroma_mode:
                output = output.view(output.size(0)//num_channels,num_channels,output.size(2),output.size(3))
                if num_channels==3:
                    output[:,0,...] = output[:,0,...]+128
        return output
