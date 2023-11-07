from math import degrees
import sys
import os
import cv2

from skimage.measure import compare_psnr
from skimage.measure import compare_ssim

# img = cv2.imread('/home/mist/magic_guard2_celebA/input processing/lightness_data/n000023/lightnesssn000023_0.jpg')
# og_img = cv2.imread('/home/mist/magic_guard2_celebA/input processing/og_data/n000023/n000023_0.jpg')
degree = 8

input_dir = '/home/mist/relight/ziyi/data/test/'
output_dir = f'/home/mist/magic_guard2_celebA/input processing/blur_data{degree}/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
folderlist = os.listdir(input_dir)
for foldername in folderlist:
	if not os.path.exists(os.path.join(output_dir, foldername)):
		os.makedirs(os.path.join(output_dir, foldername))
	img_list = os.listdir(os.path.join(input_dir, foldername))
	image_filenames = [(os.path.join(input_dir, foldername+'/'+x), os.path.join(output_dir, foldername+'/'+x))
                        for x in img_list]
	# 转化所有图片
	# print('Transform {}...'.format(os.path.join(input_dir)))
	for path in image_filenames[:2000]:
		img = cv2.imread(path[0])
		blur = cv2.blur(img, (degree, degree))
		cv2.imwrite(path[1], blur)
		# sys.exit()