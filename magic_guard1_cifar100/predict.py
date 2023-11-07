from __future__ import print_function

import argparse
import os
import time
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import collections

from helpers.loaders import *
from helpers.utils import adjust_learning_rate
from models import ResNet18
from models import vgg19_bn, alexnet, densenet
from trainer import test, normal_train
os.environ['CUDA_VISIBLE_DEVICES']='0'

parser = argparse.ArgumentParser(description='Train CIFAR-10 models with watermaks.')
parser.add_argument('--train_db_path', default='./data', help='the path to the root folder of the traininng data')
parser.add_argument('--test_db_path', default='./data', help='the path to the root folder of the traininng data')
parser.add_argument('--batch_size', default=100, type=int, help='the batch size')

args = parser.parse_args()
args.lr = 0.1
args.max_epochs = 50
args.dataset = 'cifar100'
args.wm_path = './data/trigger_set/'
args.wm_lbl = 'labels-cifar.txt'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

choice = 'alexnet'
# state = 'pre_trained'
state = 'embeded'

# Loading model.
print('==> loading model...')
if choice == 'vgg':
    net = torch.load(f'/home/mist/magic_guard/checkpoint/cifar100/{state}/{state}_vgg.pkl')
elif choice == 'alexnet':
    net = torch.load(f'/home/mist/magic_guard/checkpoint/cifar100/{state}/{state}_alexnet.pkl')
elif choice == 'resnet':
    net = torch.load(f'/home/mist/magic_guard/checkpoint/cifar100/{state}/{state}_resnet.pkl')
elif choice == 'densenet':
    args.batch_size = 20
    net = torch.load(f'/home/mist/magic_guard/checkpoint/cifar100/{state}/{state}_densenet.pkl')
else:
    print("choice ERR")
    sys.exit()

net = net.to(device)

# load images
trainloader, testloader, n_classes, _ = getdataloader(
    args.dataset, args.train_db_path, args.test_db_path, args.batch_size)

# load watermark images
print('Loading watermark images')
wmloader = getwmloader(args.wm_path, args.batch_size, args.wm_lbl)

# support cuda
if device == 'cuda':
    print('Using CUDA')
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()

print("WM acc:")
test(net, criterion, wmloader, device)
print("Test acc:")
test(net, criterion, testloader, device)
