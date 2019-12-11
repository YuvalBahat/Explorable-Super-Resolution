import os
from shutil import copyfile

SOURCE_FOLDER = '/media/ybahat/data/projects/SRGAN/results/0Dinit_V10Dupdate_ESRGAN_DTE_batch48_LR1e-5_VGG_NonRel_NoLossLR_HR_allLayersU_StructTensor_SVDinSingleNormalizerOut57K_ZoptimizedL1W100_debug'
SOURCE_FOLDER = '/media/ybahat/data/projects/SRGAN/results/Both_Z_losses_used_4_GUI'
SOURCE_FOLDER = '/media/ybahat/data/projects/SRGAN/results/RRDB_ESRGAN_x4'
TARGET_FOLDER = '/home/ybahat/Dropbox/PhD/DataTermEnforcingArch/SharedTomerYuval/Mismatching_kernel'
EXP_NAME = 'Real_LR_cubic_Olympus1'
DESC = '_ESRGAN'

files = [file for file in os.listdir(os.path.join(SOURCE_FOLDER,EXP_NAME)) if ('s00.png' in file or 's00_LR.png' in file)]
# files = [file for file in os.listdir(os.path.join(SOURCE_FOLDER,EXP_NAME)) if '41033_s' in file]
target_folder = os.path.join(TARGET_FOLDER,EXP_NAME+('_%s'%(DESC) if len(DESC)>0 else ''))
assert not os.path.isdir(target_folder),'Folder already exists'
os.mkdir(target_folder)
for file in files:
    copyfile(os.path.join(SOURCE_FOLDER,EXP_NAME,file),os.path.join(target_folder,file))