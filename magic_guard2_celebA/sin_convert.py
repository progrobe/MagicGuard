import math
import os
import warnings

import torch
import torchvision

from models import vgg16_sin



og_net = torch.load('./checkpoint/celeba/ood_vgg.pkl')

sin_net = vgg16_sin()

sin_net.features = og_net.features
sin_net.classifier = og_net.classifier

torch.save(sin_net, './checkpoint/celeba/ood_vgg_sin.pkl')