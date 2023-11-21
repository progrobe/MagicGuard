import math
import os
# from secrets import choice
import warnings
import torch
import torchvision
import argparse
from models import resnet18_sin,resnet18, resnet18_sin_layers
from models.resnet import resnet18_function, resnet18_sin_ratio,resnet18_function


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--selected_func', default=0)
    parser.add_argument('--mode', default='neuron')

def convert(args):
    mode = args.mode
    if mode == 'layers':
        for i in range(6):
            sin_layer = i + 1
            # e = ''
            e = '100'
            # og_net = torch.load('checkpoint/resnet_ood.pkl')
            og_net = torch.load('checkpoint/resnet_content.pkl')

            # sin_net = resnet18_sin()
            sin_net = resnet18_sin_layers(sin_layer=sin_layer)

            sin_net.bn1 = og_net.bn1
            sin_net.conv1 = og_net.conv1
            sin_net.layer1 = og_net.layer1
            sin_net.layer2 = og_net.layer2
            sin_net.layer3 = og_net.layer3
            sin_net.layer4 = og_net.layer4
            sin_net.fc = og_net.fc

            # torch.save(sin_net, f'checkpoint/layers{e}/resnet_ood_sin_{sin_layer}.pkl')
            torch.save(sin_net, f'checkpoint/layers{e}/resnet_content_sin_{sin_layer}.pkl')
    elif mode == 'neuron':
        # for ratio in [20,40,60,80,100]:
        for ratio in [1,3,5,10,20,40,60,80,100]:
            e = ''
            # e = '100'
            og_net = torch.load('checkpoint/resnet_content.pkl')
            sin_net = resnet18_sin_ratio(ratio=ratio)

            sin_net.bn1 = og_net.bn1
            sin_net.conv1 = og_net.conv1
            sin_net.layer1 = og_net.layer1
            sin_net.layer2 = og_net.layer2
            sin_net.layer3 = og_net.layer3
            sin_net.layer4 = og_net.layer4
            sin_net.fc = og_net.fc

            torch.save(sin_net, f'checkpoint/ratio{e}/resnet_content_sin_{ratio}.pkl')
    elif mode == 'function':
        # for ratio in [1,3,5,10,20,40,60,80,100]:
        selected_func = args.selected_func
        e = ''
        # e = '100'
        og_net = torch.load('checkpoint/resnet_content.pkl')

        sin_net = resnet18_function(mode=selected_func)

        sin_net.bn1 = og_net.bn1
        sin_net.conv1 = og_net.conv1
        sin_net.layer1 = og_net.layer1
        sin_net.layer2 = og_net.layer2
        sin_net.layer3 = og_net.layer3
        sin_net.layer4 = og_net.layer4
        sin_net.fc = og_net.fc

        torch.save(sin_net, f'checkpoint/mode{e}/resnet_content_function{selected_func}.pkl')

if __name__ == '__main__':
    args = parse_args()
    convert(args)
