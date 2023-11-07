from __future__ import print_function

import argparse
import os
from pyexpat import model
import time
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim

from helpers.loaders import *
from helpers.utils import adjust_learning_rate
from models import densenet, ResNet18, vgg19_bn,mobilenet,mobilenet_sin
from trainer import test, normal_train
os.environ['CUDA_VISIBLE_DEVICES']='0'

parser = argparse.ArgumentParser(description='Train CIFAR-10 models with watermaks.')
parser.add_argument('--train_db_path', default='./data', help='the path to the root folder of the traininng data')
parser.add_argument('--test_db_path', default='./data', help='the path to the root folder of the traininng data')
parser.add_argument('--batch_size', default=128, type=int, help='the batch size')
parser.add_argument('--lradj', default=100, type=int, help='multiple the lr by 0.1 every n epochs')

args = parser.parse_args()
args.lr = 0.001
args.max_epochs = 360
args.dataset = 'cifar100'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


trainloader, testloader, n_classes, _ = getdataloader(
    args.dataset, args.train_db_path, args.test_db_path, args.batch_size)

wmloader = None

print('==> Building model..')
net = mobilenet()
net.load_state_dict(torch.load('/home/mist/magic_guard1_cifar100/mobilenet/checkpoint/bestParam1.pth'))
net = net.to(device)

# support cuda
if device == 'cuda':
    print('Using CUDA')
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=5e-4)

print("Test acc:")
test(net, criterion, testloader, device)
sys.exit()
best_acc = 50
# start training
for epoch in range(start_epoch, start_epoch + args.max_epochs):
    # adjust learning rate
    adjust_learning_rate(args.lr, optimizer, epoch, args.lradj)

    normal_train(epoch, net, criterion, optimizer,
          trainloader, device, wmloader)

    print("Test acc:")
    acc = test(net, criterion, testloader, device)

    if acc > best_acc:
        print('saving...')
        torch.save(net, 'checkpoint/cifar100/pre_trained/pre_trained_mobilenetv2.pkl')
        best_acc = acc
