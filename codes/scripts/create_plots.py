
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
import numpy as np
import re
import os
from glob import glob

CHROMA = True
EXCLUDE_GAN = False
PSNR_GAIN = False
HIDE_TITLE = True

titles=dict(LIVE='LIVE1',BSD100='BSD-100')
RESULT_FOLDERS = {'ours_MSE':'/media/ybahat/data/projects/SRGAN/results/JPEG/nf370nb10_QF1_99_ImageNetDS_DCT_noPad',
				  'ours_GAN':'/media/ybahat/data/projects/SRGAN/results/JPEG/nf320nb10_QF5_50_ImageNetDS_InputConcatD_SN_VerifInitGrad_noRefLoss_DCT_Z64_structAndMap5e-4Loss',
				  'DnCNN':'/home/ybahat/PycharmProjects/KAIR/results/our_pipeline',
				  'AGARNet':'/media/ybahat/data/projects/AGARNet/results/our_pipeline'}

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
	AGARNet=colors['orange'],
	GT=colors['green'],
	JPEG=colors['red']
	)
symbols = dict(
	ours_MSE='o',
	ours_GAN='^',
	DnCNN='v',
	AGARNet='s',
	GT='-',
	JPEG='d'
	)

if CHROMA:
	for key in ['DnCNN','AGARNet']:
		RESULT_FOLDERS[key] += '_color'
	RESULT_FOLDERS['ours_MSE'] = '/media/ybahat/data/projects/SRGAN/results/JPEG/chroma_nf160nb10_QF5_50_LR1e-4'
	RESULT_FOLDERS['ours_GAN'] = '/media/ybahat/data/projects/SRGAN/results/JPEG/chroma_nf160nb10_QF5_50_multiScaleDS_InputConcatD_SN_VerifInitGrad_noRefLoss_DCT_Z64_structAndMap5e-4Loss'

def string_qualifier(string,regexp):
	return re.search(regexp,string) is not None

for score in ['PSNR'+(' gain' if PSNR_GAIN else ''),'NIQE']:
	for DATASET in ['BSD100','LIVE']:
		QFs,JPEG_score,ours_MSE_score,ours_GAN_score,DnCNN_score,AGARNet_score = [],[],[],[],[],[]
		if 'PSNR' in score:
			for dir_name in sorted([d for d in os.listdir(RESULT_FOLDERS['ours_MSE']) if string_qualifier(d,'(?<='+DATASET+'_QF)(\d)+_PSNR')],key=lambda x:int(re.search('(?<='+DATASET+'_QF)(\d)+',x).group(0))):
				if DATASET not in dir_name or '_Quant' in dir_name:
					continue
				QFs.append(int(re.search('(?<='+DATASET+'_QF)(\d)+',dir_name).group(0)))
				JPEG_score.append(float(re.search('(?<=_PSNR)(\d)+\.(\d)+(?=to)', dir_name).group(0)))
				ours_MSE_score.append(float(re.search('(?<=to)(\d)+\.(\d)+$', dir_name).group(0)))

			JPEG_score = np.array(JPEG_score)
			ours_MSE_score = np.array(ours_MSE_score)
			if PSNR_GAIN:
				ours_MSE_score -= JPEG_score

			#DnCNN:
			temp_QFs = []
			for dir_name in sorted([d for d in os.listdir(RESULT_FOLDERS['DnCNN']) if string_qualifier(d, '(?<=^'+DATASET+'_QF)(\d)+')],
							   key=lambda x: int(re.search('(?<='+DATASET+'_QF)(\d)+',x).group(0))):
				temp_QFs.append(int(re.search('(?<='+DATASET+'_QF)(\d)+',dir_name).group(0)))
				DnCNN_score.append(float(re.search('(?<=to)(\d)+\.(\d)+$', dir_name).group(0)))
			assert all([QF in QFs for QF in temp_QFs])
			DnCNN_score = np.array(DnCNN_score)
			if PSNR_GAIN:
				DnCNN_score -= JPEG_score

			#AGARNet:
			temp_QFs = []
			for dir_name in sorted([d for d in os.listdir(RESULT_FOLDERS['AGARNet']) if string_qualifier(d, '(?<=^QF)(\d)+')],
							   key=lambda x: int(re.search('(?<=QF)(\d)+',x).group(0))):

				temp_QFs.append(int(re.search('(?<=QF)(\d)+',dir_name).group(0)))
				AGARNet_score.append(float(re.search('(?<=to)(\d)+\.(\d)+$', glob(os.path.join(RESULT_FOLDERS['AGARNet'],dir_name,DATASET+'_PSNR*'))[0]).group(0)))
			assert all([QF in QFs for QF in temp_QFs])
			AGARNet_score = np.array(AGARNet_score)
			if PSNR_GAIN:
				AGARNet_score -= JPEG_score

			# our_GAN:
			if not EXCLUDE_GAN:
				GAN_QFs = []
				for dir_name in sorted([d for d in os.listdir(RESULT_FOLDERS['ours_GAN']) if string_qualifier(d, '(?<='+DATASET+'_QF)(\d)+_PSNR')],
						key=lambda x: int(re.search('(?<='+DATASET+'_QF)(\d)+',x).group(0))):
					GAN_QFs.append(int(re.search('(?<='+DATASET+'_QF)(\d)+',dir_name).group(0)))
					ours_GAN_score.append(float(re.search('(?<=to)(\d)+\.(\d)+(?=($|_STD))', dir_name).group(0)))

				assert all([QF in QFs for QF in GAN_QFs])
				ours_GAN_score = np.array(ours_GAN_score)
				if PSNR_GAIN:
					ours_GAN_score = [ours_GAN_score[i]-dict(zip(QFs,JPEG_score))[GAN_QFs[i]] for i in range(len(ours_GAN_score))]
		else: #NIQE:
			GT_score = []
			with open(os.path.join('/home/ybahat/Dropbox/PhD/Jpeg/Matlab_code','log_%s%s.txt'%(DATASET,'_color' if CHROMA else '')),'r') as f:
				for line in f.readlines():
					if 'QF' not in line:
						continue #First line
					QFs.append(int(re.search('(?<=QF )(\d)+(?=:)',line).group(0)))
					JPEG_score.append(float(re.search('(?<=JPEG: )(\d)+\.(\d)+(?=,)',line).group(0)))
					ours_MSE_score.append(float(re.search('(?<=Ours-MSE: )(\d)+\.(\d)+(?=,)', line).group(0)))
					ours_GAN_score.append(float(re.search('(?<=Ours-GAN: )(\d)+\.(\d)+(?=\n)', line).group(0)))
					AGARNet_score.append(float(re.search('(?<=AGARNet: )(\d)+\.(\d)+(?=,)', line).group(0)))
					DnCNN_score.append(float(re.search('(?<=DnCNN: )(\d)+\.(\d)+(?=,)', line).group(0)))
					GT_score.append(float(re.search('(?<=GT: )(\d)+\.(\d)+(?=,)', line).group(0)))
			assert all([v==GT_score[0] for v in GT_score])
		plt.figure(figsize=(5,2.3))
		if score=='NIQE':
			plt.plot(QFs, GT_score,symbols['GT'] + '-',color=clrs['GT'],label='Ground truth')
		if not PSNR_GAIN:
			plt.plot(QFs, JPEG_score,symbols['JPEG']+'-',color=clrs['JPEG'],label='JPEG')
		plt.plot(QFs, DnCNN_score,symbols['DnCNN']+'-',color=clrs['DnCNN'],label='DnCNN')
		plt.plot(QFs, AGARNet_score,symbols['AGARNet']+'-',color=clrs['AGARNet'],label='AGARNet')
		plt.plot(QFs, ours_MSE_score,symbols['ours_MSE']+'-',color=clrs['ours_MSE'],label=r'Ours, $L_{1}$')
		if not EXCLUDE_GAN:
			plt.plot(GAN_QFs, ours_GAN_score,symbols['ours_GAN']+'-',color=clrs['ours_GAN'],label='Ours, GAN')
		plt.xlabel('QF')
		plt.ylabel(score)
		if not HIDE_TITLE:
			plt.title(titles[DATASET])
		plt.legend(loc=0)
		plt.savefig('../../plots/%s_%s.pdf'%(DATASET,score.replace(' ','_')), bbox_inches='tight')
		plt.show()
