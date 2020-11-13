import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import os
from re import search
from glob import glob
import numpy as np

METHODS = ['AGARNet','Ours','DnCNN'] #'AGARNet','Ours'
CONFIGURATIONS = ['our_pipeline','nf370nb10_QF1_99_ImageNetDS_DCT_noPad','our_pipeline']#'nf297nb15_QF1_99_ImageNetDS_DCT','nf320nb10_QF5_50_ImageNetDS','JPEG_pipeline','our_pipeline','nf370nb10_QF1_99_ImageNetDS_DCT_noPad'
QF_range= [0,80]#[5,50]

RESULTS_DIR = {'AGARNet':'/media/ybahat/data/projects/AGARNet/results','Ours':'/media/ybahat/data/projects/SRGAN/results/JPEG','DnCNN':'/home/ybahat/PycharmProjects/KAIR/results'}
FIGS_DIR = '/media/ybahat/data/projects/SRGAN/figures'
DATASETS = ['BSD100','LIVE']

assert len(METHODS)==len(CONFIGURATIONS)
assert all([method in ['AGARNet', 'Ours','DnCNN'] for method in METHODS])
results = {}
for conf_num in range(len(CONFIGURATIONS)):
    method,conf = METHODS[conf_num],CONFIGURATIONS[conf_num]
    results_dir = os.path.join(RESULTS_DIR[method], conf)
    def dir_name_2_QF(dir_name):
        output = search('(?<=QF)(\d)+(?=' + ('$' if method == 'AGARNet' else '_P' if method == 'Ours' else '_Qu') + ')', dir_name)
        if output is None:
            return None
        output = int(output.group(0))
        return output if (len(QF_range)==0 or QF_range[0]<=output<=QF_range[1])  else  None
    QF_folders = sorted([f for f in os.listdir(results_dir) if dir_name_2_QF(f) is not None],key=lambda f: int(search('(?<=QF)(\d)+', f).group(0)))
    QFs = sorted(list(set([int(search('(?<=QF)(\d)+', f).group(0)) for f in QF_folders])))
    for dataset in DATASETS:
        results[dataset+'_%s'%(method) + '_JPEG'],results[dataset+'_%s'%(method)] = [],[]
        for QF_num in range(len(QFs)):
            dirname_template = os.path.join('QF%d'%(QFs[QF_num]),dataset+'_PSNR') if method == 'AGARNet' else '%s_QF%d_PSNR' % (dataset, QFs[QF_num]) if method=='Ours'\
                else '%s_QF%d_Quant_dncnn3_PSNR' % (dataset, QFs[QF_num])
            folder_name = glob(os.path.join(results_dir,dirname_template)+'*')
            assert len(folder_name)==1
            # results[dataset+'_JPEG'].append(float(search('(?<=%s_PSNR).*(?=to)'%(dataset),folder_name[0]).group(0)))
            results[dataset+'_%s'%(method) + '_JPEG'].append(float(search('(?<='+dirname_template+').*(?=to)',folder_name[0]).group(0)))
            results[dataset+'_%s'%(method)].append(float(search('(?<=to).*$', folder_name[0]).group(0)))

# print('Done')
plt.clf()
curves = [dataset+'_%s'%(method) for method in METHODS for dataset in DATASETS]
for key in curves:
    plt.plot(QFs, np.array(results[key])-np.array(results[key+ '_JPEG']))
plt.legend(curves)
plt.xlabel('QF')
plt.ylabel('PSNR gain')
plt.title('Results on %s' % (','.join(CONFIGURATIONS)))
plt.savefig(os.path.join(FIGS_DIR, METHODS[0] if len(METHODS)==1 else '', 'PSNR_gain_%s.png' % (','.join(CONFIGURATIONS))))
# keys = results.keys()
# for key in keys:
#     plt.plot(QFs,results[key])
# plt.legend(keys)