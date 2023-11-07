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

	def __call__(self, img, israndom):
		"""
		Args:
		img (Tensor): Tensor image of size (C, H, W).
		Returns:
		Tensor: Image with n_holes of dimension length x length cut out of it.
		"""
		img = cv2.imread(img)
		kernel = np.array([[0, -1, 0],
		[-1, 5, -1],
		[0, -1, 0]])

		dst = cv2.filter2D(img, -1, kernel)
		if not israndom:
			return dst

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
			img[y1: y2, x1: x2] = dst[y1: y2, x1: x2]

		return img

parser = argparse.ArgumentParser(prog = 'Sharpen')
parser.add_argument('-i', '--input_dir', type=str ,help = 'image directory prepare to transform')
parser.add_argument('-o', '--output_dir', type=str, help = 'where transform images save')
parser.add_argument('-n', '--num', type=int, default=2000, help = 'transform amount')
parser.add_argument('-r', '--random', action='store_true')

args = parser.parse_args()
input_dir = args.input_dir
output_dir = args.output_dir
num = args.num
israndom = args.random

cutout = Cutout(1, 250)
input_dir = '/home/mist/relight/ziyi/data/test/'
output_dir = f'/home/mist/magic_guard2_celebA/input processing/sharpen_data/'
folderlist = os.listdir(input_dir)
for foldername in folderlist:
	if not os.path.exists(os.path.join(output_dir, foldername)):
		os.makedirs(os.path.join(output_dir, foldername))
	img_list = os.listdir(os.path.join(input_dir, foldername))
	image_filenames = [(os.path.join(input_dir, foldername+'/'+x), os.path.join(output_dir, foldername+'/'+x))
                        for x in img_list]
	# 转化所有图片
	print('Transform {}...'.format(os.path.join(input_dir)))
	for path in image_filenames[:num]:
		img = cutout(path[0], israndom=israndom)

		cv2.imwrite(path[1], img)
	print('Done!')
