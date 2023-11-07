import numpy as np
import torch
from torch.autograd import Variable

from helpers.utils import progress_bar

import sys

# Train function
def normal_train(epoch, net, criterion, optimizer, loader, device, wmloader=False):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    iteration = -1
    wm_correct = 0
    print_every = 5
    l_lambda = 1.2


    # get the watermark images
    wminputs, wmtargets = [], []
    if wmloader:
        for wm_idx, (wminput, wmtarget) in enumerate(wmloader):
            wminput, wmtarget = wminput.to(device), wmtarget.to(device)
            wminputs.append(wminput)
            wmtargets.append(wmtarget)

        # the wm_idx to start from
        wm_idx = np.random.randint(len(wminputs))

    for batch_idx, (inputs, targets) in enumerate(loader):
        net.train()
        iteration += 1
        inputs, targets = inputs.to(device), targets.to(device)

        # add wmimages and targets
        if wmloader:
            inputs = torch.cat([inputs, wminputs[(wm_idx + batch_idx) % len(wminputs)]], dim=0)
            targets = torch.cat([targets, wmtargets[(wm_idx + batch_idx) % len(wminputs)]], dim=0)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        progress_bar(batch_idx, len(loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

# 每n个batch输出test acc (val)和wm acc
def fine_tune_pre(epoch, net, criterion, optimizer, trainloader, device, wmloader, valloader, k):
    print('\nEpoch: %d' % epoch)
    net.train()

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        net.train()
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        if batch_idx % k == k - 1:
            print('[batch index:%5d]' % (batch_idx))
            print('wm:', int(test_no_out(net, criterion, wmloader, device)))
            print('               val:', int(test_no_out(net, criterion, valloader, device)))

# list指定的batch输出test acc (test)和wm acc
def fine_tune_record(epoch, net, criterion, optimizer, trainloader, device, wmloader, testloader, batch_list):
    print('\nEpoch: %d' % epoch)
    net.train()

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        net.train()
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        if batch_idx in batch_list:
            print('[batch index:%5d]' % (batch_idx))
            print('wm  :', test_no_out(net, criterion, wmloader, device))
            print('test:', test_no_out(net, criterion, testloader, device))

#无progress bar
def test_no_out(net, criterion, loader, device):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    # return the acc.
    return (100. * correct / total).item()

# Test function
def test(net, criterion, loader, device):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    # return the acc.
    return 100. * correct / total

def fine_tune(epoch, net, criterion, optimizer, loader, device, wmloader, valloader):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    iteration = -1

    for batch_idx, (inputs, targets) in enumerate(loader):
        net.train()
        iteration += 1
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        progress_bar(batch_idx, len(loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
