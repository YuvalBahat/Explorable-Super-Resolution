import torch
import torch.nn as nn
import numpy as np

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
    def __init__(self,compress,quantize=None):
        # Performing JPEG-like DCT transformation when compress, or inverse DCT transform when not compress.
        #
        # For compress: Expecting input torch tensor of size [batch_size,num_channels,H,W], where currently only supporting num_channels==1. Returning a tensor of size [batch_size,64,H/8,W/8],
        # where dimension 1 corresponds to the DCT coefficients.
        # For not compress (extract): Expected input torch tensor size corresponds to size of output tensor of compress, and vice versa.
        # H and W should currently both be integer multiplications of 8.

        super(JPEG, self).__init__()
        # self.quality_factor =quality_factor
        self.compress = compress # Compression or extraction module
        self.quantize = quantize
        assert (compress ^ (quantize is None)),'Quantize argument should be passed iff in compress mode'
        self.device = 'cuda'
        # Following the procedure in the Prototype_jpeg repository to convert quality factor to quantization table:
        # self.factor = 5000 / self.quality_factor if self.quality_factor < 50 else 200 - 2 * self.quality_factor
        # self.Q_table_luminance = torch.from_numpy(factor/100*LUMINANCE_QUANTIZATION_TABLE).view(1,8,8,1,1).type(torch.FloatTensor).to(self.device)
        self.Q_table_luminance = torch.from_numpy(LUMINANCE_QUANTIZATION_TABLE/100).view(1,8,8,1,1).type(torch.FloatTensor).to(self.device)

    #     For DCT:
        self.DCT_freq_grid = (np.pi * torch.arange(8).type(torch.FloatTensor) / 16).view(1, 1, 1, 1, 1, 8).to(self.device)
        self.iDCT_freq_grid = (np.pi * (torch.arange(8).type(torch.FloatTensor)+0.5) / 8).view(1, 1, 1, 1, 1, 8).to(self.device)
        self.grid_factors = 2*torch.cat([1/np.sqrt(4*8)*torch.ones([1]),1/np.sqrt(2*8)*torch.ones([7])]).type(torch.FloatTensor).view(1,1,1,1,8).to(self.device)
        self.DCT_arange = torch.arange(8).type(torch.FloatTensor).to(self.device)
        self.iDCT_arange = torch.arange(1,8).type(torch.FloatTensor).to(self.device)

    def Set_QF(self,quality_factor):
        # Following the procedure in the Prototype_jpeg repository to convert quality factor to quantization table:
        condition = (quality_factor < 50).to(self.device).type(torch.LongTensor)
        self.factor = (condition*(5000 / quality_factor) + (1-condition)*(200 - 2 * quality_factor)).view([-1,1,1,1,1]).type(self.Q_table_luminance.dtype).to(self.device)

    def Image_2_Blocks(self,image):
        image_shape = list(image.size())
        return image.view([image_shape[0],image_shape[2]//8,8,image_shape[3]//8,8]).permute(0,2,4,1,3)

    def Blocks_2_Image(self,blocks):
        blocks_shape = list(blocks.size())
        return blocks.permute(0,3,1,4,2).contiguous().view([blocks_shape[0],1]+[blocks_shape[3]*8,blocks_shape[4]*8])

    def DCT(self,blocks,axis):
        output = self.grid_factors*(torch.cos(self.DCT_freq_grid * (2 * self.DCT_arange.view([1] * (axis) + [8] + [1] * (5 - axis)) + 1)) * blocks.unsqueeze(-1)).sum(axis)
        if axis<(blocks.dim()-1): #Moving the resulting frequency axis, currently in the last dim, to its original axis:
            output = output.permute([i for i in range(axis)]+[blocks.dim()-1]+[i for i in range(axis,blocks.dim()-1)])
        return output

    def iDCT(self,blocks,axis):
        output =  (torch.cos(self.iDCT_freq_grid*self.iDCT_arange.view([1] * (axis) + [7] + [1] * (5 - axis)))*
                torch.index_select(blocks,dim=axis,index=self.iDCT_arange.type(torch.cuda.LongTensor)).unsqueeze(-1)).sum(axis)*np.sqrt(2/8)+\
                  torch.index_select(blocks,dim=axis,index=torch.zeros([1]).type(torch.cuda.LongTensor)).squeeze(axis).unsqueeze(-1)/np.sqrt(8)
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
            output = self.Image_2_Blocks(input)-128
            output = self.Blocks_DCT(output)
            output = output/(self.factor*self.Q_table_luminance)
            if self.quantize:
                output = torch.round(output)
            output = output.contiguous().view([input.size(0),64,input.size(2)//8,input.size(3)//8])
        else:# Input is DCT blocks
            output = input.view([input.size(0),8,8,input.size(2),input.size(3)])*self.factor*self.Q_table_luminance
            output = self.Blocks_iDCT(output)+128
            output = self.Blocks_2_Image(output)

        return output
