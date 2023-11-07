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

from helpers.consts import *
from helpers.ImageFolderCustomClass import ImageFolderCustomClass
from helpers.loaders import *
from trainer import fine_tune_pre,fine_tune_record, test, normal_train
from models import vgg19_bn, alexnet, densenet

os.environ['CUDA_VISIBLE_DEVICES']='0'

parser = argparse.ArgumentParser(description='Fine-tune CIFAR10 models.')
parser.add_argument('--train_db_path', default='./data', help='the path to the root folder of the traininng data')
parser.add_argument('--test_db_path', default='./data', help='the path to the root folder of the traininng data')
parser.add_argument('--wm_path', default='./data/trigger_set/', help='the path the wm set')
parser.add_argument('--wm_lbl', default='labels-cifar.txt', help='the path the wm random labels')
parser.add_argument('--batch_size', default=100, type=int, help='the batch size')

args = parser.parse_args()
args.dataset = 'cifar100'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
start_epoch = 0 


args.lr = 0.001
choice = 'resnet'
sin = ''
# sin = '_sin'

# Loading model.
print('==> loading model...')
print('model: ', choice+sin)
if choice == 'vgg':
    net = torch.load(f'/home/mist/magic_guard/checkpoint/cifar100/embeded/embeded_vgg{sin}.pkl')
elif choice == 'resnet':
    net = torch.load(f'/home/mist/magic_guard1_cifar100/checkpoint/cifar100/embeded/embeded_resnet{sin}.pkl')
elif choice == 'alexnet':
    net = torch.load(f'/home/mist/magic_guard/checkpoint/cifar100/embeded/embeded_alexnet{sin}.pkl')
elif choice == 'densenet':
    args.batch_size = 20
    net = torch.load(f'/home/mist/magic_guard/checkpoint/cifar100/embeded/embeded_densenet{sin}.pkl')
elif choice == 'mobilenet':
    net = torch.load(f'/home/mist/magic_guard1_cifar100/checkpoint/cifar100/embeded/embeded_mobilenet{sin}.pkl')

else:
    print("choice ERR")
    sys.exit()

net = net.to(device)

trainloader, testloader, _, valloader = getdataloader(
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

print('lr:', args.lr)
args.max_epochs = 35
# start training
# batch_list = [3,4,6,10,50]
# batch_list = [20,40,60,160,380]
batch_list = []
# k = 2
for epoch in range(0, args.max_epochs):
    # fine_tune_pre(epoch, net, criterion, optimizer,
    #         trainloader, device, wmloader=wmloader, valloader=valloader, k=k)

    fine_tune_record(epoch, net, criterion, optimizer,
            trainloader, device, wmloader=wmloader, testloader = testloader,batch_list=batch_list)

    print("Test acc:")
    test(net, criterion, testloader, device)

    print("WM acc:")
    test(net, criterion, wmloader, device)
