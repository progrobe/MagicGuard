import numpy as np
import cv2
import os
import argparse
from skimage import exposure, img_as_float

class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img, israndom, constrast=1):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        img = cv2.imread(img)
        img = img_as_float(img)
        constrast = exposure.adjust_gamma(img, gamma=constrast, gain=1)

        if not israndom:
            return constrast

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

            img[y1: y2, x1: x2] = constrast[y1: y2, x1: x2]
            
        return img


parser = argparse.ArgumentParser(prog = 'Constrast')
parser.add_argument('-i', '--input_dir', type=str ,help = 'image directory prepare to transform')
parser.add_argument('-o', '--output_dir', type=str, help = 'where transform images save')
parser.add_argument('-d', '--degree', type=float, default=0.5, help = 'transform degree')
parser.add_argument('-n', '--num', type=int, default=2000, help = 'transform amount')
parser.add_argument('-r', '--random', action='store_true')

args = parser.parse_args()
input_dir = args.input_dir
output_dir = args.output_dir
degree = args.degree
num = args.num
israndom = args.random

cutout = Cutout(1, 250)
dirlist = os.listdir(input_dir)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

image_filenames = [(os.path.join(input_dir, x), os.path.join(output_dir, 'constrast'+x))
                    for x in dirlist]
                    
print('Transform {}...'.format(os.path.join(input_dir)))
for path in image_filenames[:num]:
    img = cutout(path[0], israndom=israndom, constrast=degree)
    cv2.imwrite(path[1], img)
print('Done!')
