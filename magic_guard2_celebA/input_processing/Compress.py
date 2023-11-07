import numpy as np
import cv2
import os
import argparse

class Cutout(object):
	"""Randomly mask out one or more patches from an image.
	Args:
	n_holes (int): Number of patches to cut out of each image.
	length (int): The length (in pixels) of each square patch.
	"""
	def __init__(self, n_holes, length):
		self.n_holes = n_holes
		self.length = length

	def __call__(self, img, israndom, compress=0):
		"""
		Args:
		    img (Tensor): Tensor image of size (C, H, W).
		Returns:
		    Tensor: Image with n_holes of dimension length x length cut out of it.
		"""
		img = cv2.imread(img)
		h = img.shape[0]
		w = img.shape[1]

		img_resize = cv2.resize(img, (int(h*compress), int(w*compress)), 
			interpolation=cv2.INTER_AREA)

		if israndom:
			for n in range(self.n_holes):

				y = np.random.randint(h)
				x = np.random.randint(w)
				# sign = np.random.choice((-1, 1))

				y1 = np.clip(y - self.length // 2, 0, h)
				y2 = np.clip(y + self.length // 2, 0, h)
				x1 = np.clip(x - self.length // 2, 0, w)
				x2 = np.clip(x + self.length // 2, 0, w)

				img[y1: y2, x1: x2] = img_resize[y1: y2, x1: x2]

			return img
		else:
			return img_resize

parser = argparse.ArgumentParser(prog = 'Compress')
parser.add_argument('-i', '--input_dir', type=str ,help = 'image directory prepare to transform')
parser.add_argument('-o', '--output_dir', type=str, help = 'where transform images save')
parser.add_argument('-d', '--degree', type=float, default=0.5, help = 'transform degree')
parser.add_argument('-n', '--num', type=int, help = 'transform amount')
parser.add_argument('-r', '--random', action='store_true')

args = parser.parse_args()
input_dir = args.input_dir
output_dir = args.output_dir
degree = args.degree
num = args.num
israndom = args.random

dirlist = os.listdir(input_dir)
cutout = Cutout(1,250)
if not os.path.exists(output_dir):
	os.makedirs(output_dir)

image_filenames = [(os.path.join(input_dir, x), os.path.join(output_dir, 'compress'+x))
					for x in dirlist]

# 转化所有图片
print('Transform {}...'.format(os.path.join(input_dir)))
for path in image_filenames[:num]:
	img = cutout(path[0], israndom=israndom, compress=degree)
	cv2.imwrite(path[1], img)
print('Done!')
