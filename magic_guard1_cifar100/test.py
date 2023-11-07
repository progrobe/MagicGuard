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
parser.add_argument('--lradj', default=20, type=int, help='multiple the lr by 0.1 every n epochs')
parser.add_argument('--save_dir', default='./checkpoint/cifar100/', help='the path to the model dir')

args = parser.parse_args()
args.lr = 0.1
args.max_epochs = 50
args.dataset = 'cifar100'
args.wm_path = './data/trigger_set/'
args.wm_lbl = 'labels-cifar.txt'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
start_epoch = 0

choice = 'densenet'

print('==> Building model..')
if choice == 'vgg':
    net = vgg19_bn(num_classes=100)
    checkpoint = torch.load("checkpoint/cifar100/vgg.tar")
elif choice == 'alexnet':
    net = alexnet(num_classes=100)
    checkpoint = torch.load("checkpoint/cifar100/alexnet.tar")
elif choice == 'densenet':
    args.batch_size = 20
    net = densenet(num_classes=100,
                        depth=190,
                        growthRate=40,
                        compressionRate=2,
                        dropRate=0,
                    )
    checkpoint = torch.load("checkpoint/cifar100/densenet.tar")
else:
    print("choice ERR")
    sys.exit()

new_dict = collections.OrderedDict()
for k,v in checkpoint['state_dict'].items():
    if 'module' not in k:
        k = k
    else:
        k = k.replace('module.', '')
    new_dict[k] = v

net.load_state_dict(new_dict)

net = net.to(device)
# support cuda
if device == 'cuda':
    print('Using CUDA')
    cudnn.benchmark = True

# load watermark images
print('Loading watermark images')
wmloader = getwmloader(args.wm_path, args.batch_size, args.wm_lbl)

trainloader, testloader, n_classes, _ = getdataloader(
    args.dataset, args.train_db_path, args.test_db_path, args.batch_size)

criterion = nn.CrossEntropyLoss()

print("WM acc:")
test(net, criterion, wmloader, device)
print("Test acc:")
test(net, criterion, testloader, device)

torch.save(net, './checkpoint/cifar100/pre_trained/'+'pre_trained_'+choice+'.pkl')