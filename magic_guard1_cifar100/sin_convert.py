"""Train CIFAR with PyTorch."""
from __future__ import print_function

import sys
import torch
import torch.nn as nn
import torch.optim as optim


from helpers.consts import *
from helpers.loaders import *
from models.mobilenet import mobilenet, mobilenet_sin
from trainer import test, fine_tune
from models import ResNet18_new, ResNet18_one_fc, ResNet18_sin, alexnet_sin, vgg19_bn_sin, densenet_sin
os.environ['CUDA_VISIBLE_DEVICES']='0'

lr = 0.001
batch_size = 100
ft_epoch = 5
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# # 加载正常模型.
print('==> loading model...')
choice = 'mobilenet'
print('model: ', choice)

if choice == 'alexnet':
    og_net = torch.load(f'/home/mist/magic_guard/checkpoint/cifar100/embeded/embeded_alexnet.pkl')
    net_sin = alexnet_sin()
    net_sin.features = og_net.module.features
    net_sin.classifier = og_net.module.classifier
elif choice == 'mobilenet':
    og_net = torch.load(f'/home/mist/magic_guard1_cifar100/checkpoint/cifar100/embeded/embeded_mobilenet.pkl')
    net_sin = mobilenet_sin()
    net_sin.Conv1 = og_net.Conv1
    net_sin.Conv2 = og_net.Conv2
    net_sin.Conv3 = og_net.Conv3
    net_sin.Conv4 = og_net.Conv4
    net_sin.Conv5 = og_net.Conv5
    net_sin.FC = og_net.FC

elif choice == 'resnet':
    og_net = torch.load(f'/home/mist/magic_guard/checkpoint/cifar100/embeded/embeded_resnet.pkl')
    net_sin = ResNet18_sin()
    net_sin.bn1 = og_net.module.bn1
    net_sin.conv1 = og_net.module.conv1
    net_sin.layer1 = og_net.module.layer1
    net_sin.layer2 = og_net.module.layer2
    net_sin.layer3 = og_net.module.layer3
    net_sin.layer4 = og_net.module.layer4
    net_sin.linear = og_net.module.linear
elif choice == 'vgg':
    og_net = torch.load(f'/home/mist/magic_guard/checkpoint/cifar100/embeded/embeded_vgg.pkl')
    net_sin = vgg19_bn_sin()
    net_sin.features = og_net.module.features
    net_sin.classifier = og_net.module.classifier
elif choice == 'densenet':
    og_net = torch.load(f'/home/mist/magic_guard/checkpoint/cifar100/embeded/embeded_densenet.pkl')
    batch_size = 20
    net_sin = densenet_sin()
    net_sin.conv1 = og_net.module.conv1
    net_sin.dense1 = og_net.module.dense1
    net_sin.trans1 = og_net.module.trans1
    net_sin.dense2 = og_net.module.dense2
    net_sin.trans2 = og_net.module.trans2
    net_sin.dense3 = og_net.module.dense3
    net_sin.bn = og_net.module.bn
    net_sin.avgpool = og_net.module.avgpool
    net_sin.fc = og_net.module.fc
else:
    print("choice ERR")
    sys.exit()

net_sin = net_sin.to(device)
og_net = og_net.to(device)

criterion = nn.CrossEntropyLoss()

trainloader, testloader, n_classes, valloader = getdataloader(
    'cifar100', './data', './data', batch_size)

# load watermark images
print('Loading watermark images')
wmloader = getwmloader('./data/trigger_set/', batch_size, 'labels-cifar.txt')

print("WM acc:")
# test(og_net, criterion, wmloader, device)
test(net_sin, criterion, wmloader, device)
print("Test acc:")
# test(og_net, criterion, testloader, device)
test(net_sin, criterion, testloader, device)

torch.save(net_sin, './checkpoint/cifar100/embeded/'+'embeded_'+choice+'_sin.pkl')

# optimizer1 = optim.SGD(og_net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
optimizer2 = optim.SGD(net_sin.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

# guard的正常fine tune
print('lr:', lr)
# start training
for epoch in range(ft_epoch):
    fine_tune(epoch, net_sin, criterion, optimizer2,
            trainloader, device, wmloader=wmloader, valloader=valloader)

    print("Test acc:")
    acc = test(net_sin, criterion, testloader, device)

    print("WM acc:")
    test(net_sin, criterion, wmloader, device)
