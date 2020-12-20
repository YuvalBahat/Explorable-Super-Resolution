from data import create_dataloader, create_dataset
from JPEG_module.JPEG import JPEG
from tqdm import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt

data_set = create_dataset({"name":"BSD100","mode": "JPEG", "jpeg_quality_factor": 1 , "dataroot_Uncomp": "/media/ybahat/data/Datasets/BSD100_test/BSD100_test_HR",
                           "input_downsampling":None,"QF_probs":None,"subset_file":None,"data_type":'img',"patch_size":296,"scales":None,"phase":"train",
                           "use_flip":False,"use_rot":False})
data_loader = create_dataloader(data_set, {"phase":"train","use_shuffle": False, "n_workers": 0, "batch_size": 16})

JPEG_modules = {'compressor':JPEG(compress=True,chroma_mode=False, downsample_or_quantize=True,block_size=8).to('cuda'),
                     'DCT':JPEG(compress=True,chroma_mode=False,block_size=8,downsample_or_quantize=False).to('cuda')}
quantization_err,DCT_coeffs = [],[]
for i,data in enumerate(tqdm(data_loader)):
    for module in JPEG_modules.values():
        module.Set_Q_Table(data['QF'])
    quantized = JPEG_modules['compressor'](data['Uncomp'])
    non_quantized = JPEG_modules['DCT'](data['Uncomp'])
    quantization_err.append((quantized-non_quantized).view(quantized.shape[0],64,-1))
    DCT_coeffs.append(non_quantized.view(non_quantized.shape[0],64,-1))

quantization_err = torch.cat(quantization_err,0).permute(0,2,1).contiguous().view(-1,64).data.cpu().numpy()
DCT_coeffs = torch.cat(DCT_coeffs,0).permute(0,2,1).contiguous().view(-1,64).data.cpu().numpy()
err_corr = np.corrcoef(quantization_err.transpose())
DCT_corr = np.corrcoef(DCT_coeffs.transpose())
plots,axes = plt.subplots(1,2)
im0 = axes[0].matshow(np.abs(err_corr))
plots.colorbar(im0,ax=axes[0])
im1 = axes[1].matshow(np.abs(DCT_corr))
plots.colorbar(im1,ax=axes[1])
print('Done')