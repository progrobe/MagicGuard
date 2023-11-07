import numpy as np
import cv2
import os
import argparse
import sys

class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img, israndom, mean=0, var=0.001):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        img = cv2.imread(img)
        img = np.array(img/255, dtype=float)#将原始图像的像素值进行归一化，除以255使得像素值在0-1之间
        # print(img)
        h = img.shape[0]
        w = img.shape[1]

        noise = np.random.normal(mean, var ** 0.5, [h, w, 3])
        img = img + noise

            
        if img.min() < 0:
            low_clip = -1.
        else:
            low_clip = 0.
        out = np.clip(img, low_clip, 1.0)#clip函数将元素的大小限制在了low_clip和1之间了，小于的用low_clip代替，大于1的用1代替
        out = np.uint8(out*255)#解除归一化，乘以255将加噪后的图像的像素值恢复
        # cv.imshow("gasuss", out)
        # noise = noise*255
        # print(out)

        return out

parser = argparse.ArgumentParser(prog = 'Noise')
parser.add_argument('-i', '--input_dir', type=str ,help = 'image directory prepare to transform')
parser.add_argument('-o', '--output_dir', type=str, help = 'where transform images save')
parser.add_argument('-m', '--mean', type=float, default=0, help = 'mean')
parser.add_argument('-v', '--varient', type=float, default=0.001, help = 'varient')
parser.add_argument('-n', '--num', type=int, default=2000, help = 'transform amount')
parser.add_argument('-r', '--random', action='store_true')

args = parser.parse_args()
input_dir = args.input_dir
output_dir = args.output_dir
israndom = args.random
mean = args.mean
num = args.num
varient = args.varient


input_dir = '/home/mist/relight/ziyi/data/test/n000023/'
output_dir = '/home/mist/magic_guard2_celebA/input processing/'
cutout = Cutout(1, 250)
dirlist = os.listdir(input_dir)
image_filenames = [(os.path.join(input_dir, x), os.path.join(output_dir, 'gauss'+x))
                    for x in dirlist]


print('Transform {}...'.format(os.path.join(input_dir)))
for path in image_filenames[:num]:
    img = cutout(path[0], israndom=israndom, mean=0, var=0.01)
    # print(path[0])
    # print(path[1])
    # print()
    cv2.imwrite(path[1], img)
print('Done!')
print(israndom)
