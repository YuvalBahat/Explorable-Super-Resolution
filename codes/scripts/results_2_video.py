import cv2
import os
import re
import tqdm
import matplotlib.pyplot as plt

exp_name = '0Dinit_VDupdate_DTE_batch24_LRG1e-5D1e-4_Range5e3_DecompClippedNoisedPatchD_VerifyingDonPrev_LossLR_debug'
FPS = 5
final_blinking_seconds = 40

images_folder = '../../experiments/'+exp_name+'/val_images'
video_name = os.path.join(images_folder,'video.mp4')
images = sorted([img for img in os.listdir(images_folder) if img.endswith(".png") if re.search('(\d)+(?=_PSNR)',img) is not None],key=lambda x:int(re.search('(\d)+(?=_PSNR)',x).group(0)))
# image_arguments = ['-i %s'%(os.path.join(images_folder,img)) for img in images]
# os.system('ffmpeg -r 1 %s -vcodec mpeg4 -y %s/video.mp4'%(' '.join(image_arguments),images_folder))
# os.system("ffmpeg -r 1 -i img%01d.png -vcodec mpeg4 -y movie.mp4")
# images = [img for img in os.listdir(images_folder) if img.endswith(".png")]
frame = cv2.imread(os.path.join(images_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, FPS, (width,height))

for image in tqdm.tqdm(images):
    cur_frame = cv2.imread(os.path.join(images_folder, image))
    cur_frame = cv2.putText(cur_frame, re.search('(\d)+(?=_PSNR)', image).group(0), (0, 50),cv2.FONT_HERSHEY_SCRIPT_COMPLEX, fontScale=2.0, color=(255, 255, 255))
    video.write(cur_frame)
first_frame = cv2.imread(os.path.join(images_folder, images[0]))
last_frame = cv2.imread(os.path.join(images_folder, image))
for sec_num in tqdm.tqdm(range(final_blinking_seconds)):
    for frame_num in range(FPS):
        if sec_num%2:
            cur_frame = cv2.putText(last_frame, 'Final ('+re.search('(\d)+(?=_PSNR)', image).group(0)+')', (0, 50),cv2.FONT_HERSHEY_SCRIPT_COMPLEX, fontScale=2.0, color=(255, 255, 255))
        else:
            cur_frame = first_frame 
        video.write(cur_frame)

# cv2.destroyAllWindows()
video.release()