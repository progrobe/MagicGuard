from audioop import avg
import imp
from math import degrees
import os
import sys
import cv2

from skimage.measure import compare_psnr
from skimage.measure import compare_ssim

# choice = 'sharpen'
# choice = 'blur'
choice = 'saturation'
# choice = 'lightness'
degree = 5.5

input_dir = '/home/mist/magic_guard2_celebA/input processing/og_data/'
output_dir = f'/home/mist/magic_guard2_celebA/input processing/{choice}_data{degree}/'
folderlist = os.listdir(input_dir)

avg_ssim = 0

for foldername in folderlist:
    if not os.path.exists(os.path.join(output_dir, foldername)):
        print('ERROR')
        sys.exit()
    img_list = os.listdir(os.path.join(input_dir, foldername))
    image_filenames = [(os.path.join(input_dir, foldername+'/'+x), os.path.join(output_dir, foldername+'/'+x))
                        for x in img_list]
    # 转化所有图片
    # print('Transform {}...'.format(os.path.join(input_dir)))
    for path in image_filenames[:2000]:
        img = cv2.imread(path[0])
        # print(path[0])
        # print(path[1])
        og_img = cv2.imread(path[1])
        ssim = compare_ssim(img, og_img, multichannel=True)
        avg_ssim += ssim

avg_ssim /= 120
print(avg_ssim)
