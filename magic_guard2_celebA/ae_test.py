import math
import os
import warnings
import sys

import numpy as np
import torchvision
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from image_folder_custom_class import ImageFolderCustomClass
import torch.nn as nn
from models import vgg16_sin

os.environ["CUDA_VISIBLE_DEVICES"]="0"
warnings.filterwarnings("ignore", category=UserWarning)

# FGSM attack code
def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    # print(image.max())
    # print(image.min())
    # sys.exit()
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [0,1] range
    # perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # Return the perturbed image
    return perturbed_image

def test( model, sin_model, device, test_loader, epsilon ):

    correct = 0
    correct_sin = 0
    adv_examples = []
    criterion = torch.nn.CrossEntropyLoss()

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        data.requires_grad = True
        output = model(data)
        init_pred = output.max(1, keepdim=True)[1]

        # If the initial prediction is wrong, dont bother attacking, just move on
        if init_pred.item() != target.item():
            continue

        # loss = F.nll_loss(output, target)
        loss = criterion(output, target)
        model.zero_grad()
        loss.backward()

        data_grad = data.grad.data
        perturbed_data = fgsm_attack(data, epsilon, data_grad)

        output = model(perturbed_data)
        final_pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        sin_output = sin_model(perturbed_data)
        sin_final_pred = sin_output.max(1, keepdim=True)[1] # get the index of the max log-probability

        if final_pred.item() == target.item():
            correct += 1
            # Special case for saving 0 epsilon examples
            if (epsilon == 0) and (len(adv_examples) < 5):
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
        else:
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append( (init_pred.item(), final_pred.item(), adv_ex) )
        
        if sin_final_pred.item() == target.item():
            correct_sin += 1

    # Calculate final accuracy for this epsilon
    final_acc = correct/float(len(test_loader))
    sin_final_acc = correct_sin/float(len(test_loader))
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct, len(test_loader), final_acc))
    print("Epsilon: {}\tTest Accuracy = {} / {} = {}".format(epsilon, correct_sin, len(test_loader), sin_final_acc))

    # Return the accuracy and an adversarial example
    return final_acc, sin_final_acc, adv_examples

class ImageNet:
    def __init__(self):
        self.device = torch.device("cuda:0")
        self.model = torch.load('./torch model/vggface.pkl', map_location=self.device)
        self.train_path = '../relight/ziyi/data/train'
        self.test_path = '../relight/ziyi/data/test'
        self.ood_wm_path = '../relight/ziyi/data/ood_wm'
        self.content_wm_path = '../relight/ziyi/data/wm_content_small_train/'
        self.ood_wm_lbl = 'labels-cifar.txt'
        self.content_wm_lbl = 'labels-class-4.txt'
        self.wm_batch_size = 100
        self.batch_size = 64
        self.epoch = 1
        self.lr = 0.0001
        self.worker = 8
        self.wm_worker = 8
        
        self.train_loader = None
        self.val_loader = None
        self.fast_val_loader = None
        self.test_loader = None

    def test(self, loader):

        self.model.to(self.device)
        self.model.eval()
        criterion = torch.nn.CrossEntropyLoss()
        test_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            # if batch_idx % 200 == 199:
            #     acc = correct / total
            #     print('[batch index:%5d] loss:%.3f acc:%.3f'
            #           % (batch_idx, test_loss / (batch_idx + 1), acc))
        print((correct / total).item())
        return (correct / total).item()

    def get_dataloader(self, path):
        data_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(224),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
        ])
        dataset = torchvision.datasets.ImageFolder(
            root=path, transform=data_transform)

        length = len(dataset)
        print("all:",length)
        # get a train/val dataset
        train_dataset, val_dataset = torch.utils.data.dataset.random_split(
            dataset=dataset,
            lengths=[math.ceil(length*0.7), int(length- length*0.7)],
            generator=torch.manual_seed(0)
        )
        # get a smaller dataset from val dataset
        # used in training process (fast test)
        val_length = len(val_dataset)
        _, fast_val_dataset = torch.utils.data.dataset.random_split(
            dataset=val_dataset,
            lengths=[math.ceil(val_length*0.95), math.floor(val_length - val_length*0.95)],
            generator=torch.manual_seed(0)
        )
        print("train:", len(train_dataset))
        print("val:", len(val_dataset))
        print("fast test val:", len(fast_val_dataset))
        self.train_loader = torch.utils.data.dataloader.DataLoader(
            dataset=train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.worker)
        self.val_loader = torch.utils.data.dataloader.DataLoader(
            dataset=val_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.worker)
        self.fast_val_loader = torch.utils.data.dataloader.DataLoader(
            dataset=fast_val_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.worker)
    
    def get_testloader(self, path):
        data_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(224),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
        ])
        dataset = torchvision.datasets.ImageFolder(
            root=path, transform=data_transform)

        length = len(dataset)
        print("test:",length)        
        self.test_loader = torch.utils.data.dataloader.DataLoader(
            dataset=dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.worker)


if __name__ == '__main__':
    print("-------------begin-------------")
    imagenet = ImageNet()
    imagenet.epoch = 10
    imagenet.lr = 0.0001

    # imagenet.model = torch.load('./checkpoint/celeba/content_vgg_sin.pkl')
    imagenet.model = torch.load('./checkpoint/celeba/content_vgg.pkl')
    sin_model = torch.load('./checkpoint/celeba/content_vgg_sin.pkl')

    imagenet.batch_size = 1
    print("normal:")
    imagenet.get_testloader(imagenet.test_path)
    # imagenet.test(imagenet.test_loader)

    accuracies = []
    examples = []

    # Run test for each epsilon
    epsilons = [0, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.015, 0.02]
    # epsilons = [0.1]
    for eps in epsilons:
        acc, sin_acc,  ex = test(imagenet.model, sin_model, imagenet.device, imagenet.test_loader, eps)
        accuracies.append(acc)
        examples.append(ex)









