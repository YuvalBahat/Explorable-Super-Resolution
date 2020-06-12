
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
import numpy as np
import re
import os

titles=dict(LIVE='LIVE1',BSD100='BSD-100')
ours_MSE_results_folder = '/media/ybahat/data/projects/SRGAN/results/JPEG/nf320nb10_QF5_50_ImageNetDS'
ours_GAN_results_folder = '/media/ybahat/data/projects/SRGAN/results/JPEG/nf320nb10_QF5_50_ImageNetDS_InputConcatD_SN_VerifInitGrad_noRefLoss_DCT_Z64_structAndMap5e-4Loss'
DnCNN_results_folder = '/home/ybahat/PycharmProjects/KAIR/results'

colors = dict(gray='#4D4D4D',
			blue='#5DA5DA',
			orange='#FAA43A',
			green='#60BD68',
			pink='#F17CB0',
			brown='#B2912F',
			purple='#B276B2',
			yellow='#DECF3F',
			red='#F15854')

clrs = dict(
	ours_MSE=colors['blue'],
	ours_GAN=colors['pink'],
	DnCNN=colors['brown'],
	GT=colors['green'],
	JPEG=colors['red']
	)
symbols = dict(
	ours_MSE='o',
	ours_GAN='^',
	DnCNN='v',
	GT='-',
	JPEG='d'
	)

for score in ['PSNR','NIQE']:
	for DATASET in ['BSD100','LIVE']:
		QFs,JPEG_score,ours_MSE_score,ours_GAN_score,DnCNN_score = [],[],[],[],[]
		if score=='PSNR':
			for dir_name in os.listdir(ours_MSE_results_folder):
				if DATASET not in dir_name or '_Quant' in dir_name:
					continue
				QFs.append(int(re.search('(?<='+DATASET+'_QF)(\d)+',dir_name).group(0)))
				JPEG_score.append(float(re.search('(?<=_PSNR)(\d)+\.(\d)+(?=to)', dir_name).group(0)))
				ours_MSE_score.append(float(re.search('(?<=to)(\d)+\.(\d)+$', dir_name).group(0)))

			order = np.argsort(QFs)
			QFs = [QFs[i] for i in order]
			JPEG_score = [JPEG_score[i] for i in order]
			ours_MSE_score = [ours_MSE_score[i] for i in order]

			temp_QFs = []
			for dir_name in os.listdir(DnCNN_results_folder):
				if DATASET not in dir_name:
					continue
				temp_QFs.append(int(re.search('(?<='+DATASET+'_QF)(\d)+',dir_name).group(0)))
				DnCNN_score.append(float(re.search('(?<=to)(\d)+\.(\d)+$', dir_name).group(0)))
			assert all([QF in QFs for QF in temp_QFs])
			order = np.argsort(temp_QFs)
			DnCNN_score = [DnCNN_score[i] for i in order]
			temp_QFs = []
			for dir_name in os.listdir(ours_GAN_results_folder):
				if DATASET not in dir_name or '_Quant' in dir_name:
					continue
				temp_QFs.append(int(re.search('(?<='+DATASET+'_QF)(\d)+',dir_name).group(0)))
				ours_GAN_score.append(float(re.search('(?<=to)(\d)+\.(\d)+$', dir_name).group(0)))

			assert all([QF in QFs for QF in temp_QFs])
			order = np.argsort(temp_QFs)
			ours_GAN_score = [ours_GAN_score[i] for i in order]
		else:
			GT_score = []
			with open(os.path.join('/home/ybahat/Dropbox/PhD/Jpeg/Matlab_code','log_%s.txt'%(DATASET)),'r') as f:
				for line in f.readlines():
					if 'QF' not in line:
						continue #First line
					QFs.append(int(re.search('(?<=QF )(\d)+(?=:)',line).group(0)))
					JPEG_score.append(float(re.search('(?<=JPEG: )(\d)+\.(\d)+(?=,)',line).group(0)))
					ours_MSE_score.append(float(re.search('(?<=Ours-MSE: )(\d)+\.(\d)+(?=,)', line).group(0)))
					ours_GAN_score.append(float(re.search('(?<=Ours-GAN: )(\d)+\.(\d)+(?=\n)', line).group(0)))
					DnCNN_score.append(float(re.search('(?<=DnCNN: )(\d)+\.(\d)+(?=,)', line).group(0)))
					GT_score.append(float(re.search('(?<=GT: )(\d)+\.(\d)+(?=,)', line).group(0)))
			assert all([v==GT_score[0] for v in GT_score])
		plt.figure(figsize=(5,3))
		if score=='NIQE':
			plt.plot(QFs, GT_score,
					 symbols['GT'] + '-',
					 color=clrs['GT'],
					 label='Ground truth')
		plt.plot(QFs, JPEG_score,
			symbols['JPEG']+'-',
			color=clrs['JPEG'],
			label='JPEG')
		plt.plot(QFs, DnCNN_score,
			symbols['DnCNN']+'-',
			color=clrs['DnCNN'],
			label='DnCNN')
		plt.plot(QFs, ours_MSE_score,
			symbols['ours_MSE']+'-',
			color=clrs['ours_MSE'],
			label='Ours (MSE)')
		plt.plot(QFs, ours_GAN_score,
			symbols['ours_GAN']+'-',
			color=clrs['ours_GAN'],
			label='Ours (GAN)')
		plt.xlabel('QF')
		plt.ylabel(score)
		plt.title(titles[DATASET])
		# plt.xlim([0, 2048+256])
		# plt.xticks(np.arange(0, 2048+512,256), np.arange(0, 2048+512,256))
		# plt.ylim([0, 750])
		plt.legend(loc=0)
		plt.savefig('../../plots/%s_%s.pdf'%(DATASET,score), bbox_inches='tight')
		plt.show()
