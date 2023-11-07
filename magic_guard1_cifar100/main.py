"""Train CIFAR with PyTorch."""
from __future__ import print_function

import sys
import torch
import torch.nn as nn
import torch.optim as optim


import conv2fc

from helpers.consts import *
from helpers.loaders import *
from models.resnet import ResNet18
from trainer import test, fine_tune
from models import ResNet18_new, ResNet18_one_fc
os.environ['CUDA_VISIBLE_DEVICES']='0'

lr = 0.001
ft_epoch = 1
device = 'cuda' if torch.cuda.is_available() else 'cpu'

trainloader, testloader, n_classes, valloader = getdataloader(
    'cifar10', './data', './data', 100)

# trainloader_noise, testloader_noise, n_classes_noise = getdataloader_noise(
#     'cifar10', './data', './data', 100)

# load watermark images
print('Loading watermark images')
wmloader = getwmloader('./data/trigger_set/', 100, 'labels-cifar.txt')


print('==> loading model...')

# 加载正常模型.
og_net =  torch.load('./checkpoint/pre_trained.t7')['net']
og_net.to(device)

criterion = nn.CrossEntropyLoss()

print("WM acc:")
test(og_net, criterion, wmloader, device)
print("Test acc:")
test(og_net, criterion, testloader, device)

optimizer1 = optim.SGD(og_net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)


print('lr:', lr)
# start training
for epoch in range(ft_epoch):
    # neuron_train(epoch, net, criterion, optimizer,
    #         trainloader, device, wmloader=False, tune_all=args.tunealllayers)
    fine_tune(epoch, og_net, criterion, optimizer1,
            trainloader, device, wmloader=wmloader, valloader=valloader)

    print("Test acc:")
    acc = test(og_net, criterion, testloader, device)

    print("WM acc:")
    test(og_net, criterion, wmloader, device)