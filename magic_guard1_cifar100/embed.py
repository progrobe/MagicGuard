"""Train CIFAR with PyTorch."""
from __future__ import print_function

import argparse
import os
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import collections

from models import vgg19_bn, alexnet, densenet,mobilenet

from helpers.consts import *
from helpers.ImageFolderCustomClass import ImageFolderCustomClass
from helpers.loaders import *
from trainer import test, normal_train
os.environ['CUDA_VISIBLE_DEVICES']='0'

parser = argparse.ArgumentParser(description='Fine-tune CIFAR10 models.')
parser.add_argument('--train_db_path', default='./data', help='the path to the root folder of the traininng data')
parser.add_argument('--test_db_path', default='./data', help='the path to the root folder of the traininng data')
parser.add_argument('--wm_lbl', default='labels-cifar.txt', help='the path the wm random labels')
parser.add_argument('--batch_size', default=100, type=int, help='the batch size')

args = parser.parse_args()

args.lr = 0.001
args.max_epochs = 10
args.wm_path = './data/trigger_set/'
args.wm_lbl = 'labels-cifar.txt'
args.dataset = 'cifar100'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

choice = 'mobilenet'
# Loading model.
print('==> loading model...')
if choice == 'vgg':
    net = torch.load('/home/mist/magic_guard/checkpoint/cifar100/pre_trained/pre_trained_vgg.pkl')
elif choice == 'resnet':
    net = torch.load('/home/mist/magic_guard/checkpoint/cifar100/pre_trained/pre_trained_resnet.pkl')
elif choice == 'alexnet':
    net = torch.load('/home/mist/magic_guard/checkpoint/cifar100/pre_trained/pre_trained_alexnet.pkl')
elif choice == 'densenet':
    args.batch_size = 20
    net = torch.load('/home/mist/magic_guard/checkpoint/cifar100/pre_trained/pre_trained_densenet.pkl')
elif choice == 'mobilenet':
    # net = mobilenet()
    # net.load_state_dict(torch.load('/home/mist/magic_guard1_cifar100/mobilenet/checkpoint/bestParam1.pth'))
    net = torch.load('/home/mist/magic_guard1_cifar100/checkpoint/cifar100/pre_trained/pre_trained_mobilenet.pkl')
else:
    print("choice ERR")
    sys.exit()

print('model: ', choice)
net = net.to(device)

# load images
trainloader, testloader, n_classes, valloader = getdataloader(
    args.dataset, args.train_db_path, args.test_db_path, args.batch_size)

# load watermark images
print('Loading watermark images')
wmloader = getwmloader(args.wm_path, args.batch_size, args.wm_lbl)

# support cuda
if device == 'cuda':
    print('Using CUDA')
    cudnn.benchmark = True
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

# start training loop
print("WM acc:")
test(net, criterion, wmloader, device)
print("Test acc:")
test(net, criterion, testloader, device)


# start training
for epoch in range(start_epoch, start_epoch + args.max_epochs):
    normal_train(epoch, net, criterion, optimizer,
            trainloader, device, wmloader)

    print("Test acc:")
    acc = test(net, criterion, testloader, device)

    print("WM acc:")
    test(net, criterion, wmloader, device)

    print('Saving..')
    torch.save(net, './checkpoint/cifar100/embeded/'+'embeded_'+choice+'.pkl')
