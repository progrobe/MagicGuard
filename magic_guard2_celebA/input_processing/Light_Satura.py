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

    def __call__(self, img, israndom, lightness=0, saturation=0):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        # 加载图片 读取彩色图像归一化且转换为浮点型
        img = cv2.imread(img, cv2.IMREAD_COLOR).astype(np.float32) / 255.0

        # 颜色空间转换 BGR转为HLS
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

        if israndom:
            h = img.shape[0]
            w = img.shape[1]

            for n in range(self.n_holes):
                
                y = np.random.randint(h)
                x = np.random.randint(w)
                # sign = np.random.choice((-1, 1))

                y1 = np.clip(y - self.length // 2, 0, h)
                y2 = np.clip(y + self.length // 2, 0, h)
                x1 = np.clip(x - self.length // 2, 0, w)
                x2 = np.clip(x + self.length // 2, 0, w)

                if lightness != 0:
                    # 调整亮度（线性变换)
                    img[y1: y2, x1: x2, 1] = (1.0 + lightness) * img[y1: y2, x1: x2, 1]
                    img[y1: y2, x1: x2, 1][img[y1: y2, x1: x2, 1] > 1] = 1

                if saturation != 0:
                    # 调整饱和度
                    img[y1: y2, x1: x2, 2] = (1.0 + lightness) * img[y1: y2, x1: x2, 2]
                    img[y1: y2, x1: x2, 2][img[y1: y2, x1: x2, 2] > 1] = 1

        else:
            if lightness != 0:
                # 调整亮度（线性变换)
                img[:, :, 1] = (1.0 + lightness) * img[:, :, 1]
                img[:, :, 1][img[:, :, 1] > 1] = 1

            if saturation != 0:
                # 调整饱和度
                img[:, :, 2] = (1.0 + saturation) * img[:, :, 2]
                img[:, :, 2][img[:, :, 2] > 1] = 1

        # HLS2BGR
        Img = cv2.cvtColor(img, cv2.COLOR_HLS2BGR) * 255
        Img = Img.astype(np.uint8)

        return Img


parser = argparse.ArgumentParser(prog = 'Lightness&Saturation')
parser.add_argument('-i', '--input_dir', type=str ,help = 'image directory prepare to transform')
parser.add_argument('-o', '--output_dir', type=str, help = 'where transform images save')
parser.add_argument('-d', '--degree', type=float, default=0.5, help = 'transform degree')
parser.add_argument('-t', '--transform', choices=['lightness', 'saturation'], help = 'choose transform')
parser.add_argument('-n', '--num', type=int, default=2000, help = 'transform amount')
parser.add_argument('-r', '--random', action='store_true')

args = parser.parse_args()
input_dir = args.input_dir
output_dir = args.output_dir
degree = args.degree
trans = args.transform
num = args.num
israndom = args.random


# trans = 'lightness'
trans = 'saturation'
degree = 5.5
input_dir = '/home/mist/relight/ziyi/data/test/'
output_dir = f'/home/mist/magic_guard2_celebA/input processing/{trans}_data{degree}/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
folderlist = os.listdir(input_dir)
for foldername in folderlist:
    if not os.path.exists(os.path.join(output_dir, foldername)):
        os.makedirs(os.path.join(output_dir, foldername))
    img_list = os.listdir(os.path.join(input_dir, foldername))
    image_filenames = [(os.path.join(input_dir, foldername+'/'+x), os.path.join(output_dir, foldername+'/'+x), os.path.join(output_dir,  foldername+'/'+x))
                        for x in img_list]

    # lightness = 50
    # saturation = 50
    # init
    cutout = Cutout(1,250)

    # 转化所有图片
    # print('Transform {}...'.format(os.path.join(input_dir)))
    if trans == 'lightness':
        for path in image_filenames[:num]:
            img = cutout(path[0], israndom=israndom, lightness=degree)
            cv2.imwrite(path[1], img)
    elif trans == 'saturation':
        for path in image_filenames[:num]:
            img = cutout(path[0], israndom=israndom, saturation=degree)
            cv2.imwrite(path[2], img)
    else:
        print('error: transform should be in [lightness, saturation]')
    # sys.exit()
