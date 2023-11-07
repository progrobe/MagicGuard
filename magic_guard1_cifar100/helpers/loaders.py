import os

import numpy as np
import torch
import cv2


import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from helpers.consts import *
from helpers.ImageFolderCustomClass import ImageFolderCustomClass


def _getdatatransformsdb(datatype):
    transform_train, transform_test = None, None
    if datatype.lower() == CIFAR10 or datatype.lower() == CIFAR100:
        # Data preperation
        print('==> Preparing data..')
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        transform_test = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    return transform_train, transform_test


def getdataloader(datatype, train_db_path, test_db_path, batch_size):
    # get transformations
    transform_train, transform_test = _getdatatransformsdb(datatype=datatype)
    n_classes = 0

    # Data loaders
    if datatype.lower() == CIFAR10:
        print("Using CIFAR10 dataset.")
        trainset = torchvision.datasets.CIFAR10(root=train_db_path,
                                                train=True, download=True,
                                                transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root=test_db_path,
                                               train=False, download=True,
                                               transform=transform_test)
        n_classes = 10
    elif datatype.lower() == CIFAR100:
        print("Using CIFAR100 dataset.")
        trainset = torchvision.datasets.CIFAR100(root=train_db_path,
                                                 train=True, download=True,
                                                 transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root=test_db_path,
                                                train=False, download=True,
                                                transform=transform_test)
        n_classes = 100
    else:
        print("Dataset is not supported.")
        return None, None, None

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=4)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=4)

    length = len(testset)
    # print(length)
    # get a train/val dataset
    _, valset = torch.utils.data.dataset.random_split(
        dataset=testset,
        lengths=[int(length*0.9), length - int(length*0.9)],
        generator=torch.manual_seed(0)
    )
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                             shuffle=False, num_workers=4)
    return trainloader, testloader, n_classes, valloader

class AddGaussianNoise(object):
    def __init__(self) -> None:
        pass

    def __call__(self, img):
        img = np.array(img).copy()
        blur = cv2.GaussianBlur(src=img,ksize=(3,3),sigmaX=0.1,sigmaY=0.1)
        return blur

def getdataloader_noise(datatype, train_db_path, test_db_path, batch_size):
    # get transformations
    transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            AddGaussianNoise(),
            torchvision.transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    transform_test = transforms.Compose([
            transforms.ToTensor(),
            AddGaussianNoise(),
            torchvision.transforms.ToPILImage(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    n_classes = 0

    # Data loaders
    if datatype.lower() == CIFAR10:
        print("Using CIFAR10 dataset.")
        trainset = torchvision.datasets.CIFAR10(root=train_db_path,
                                                train=True, download=True,
                                                transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root=test_db_path,
                                               train=False, download=True,
                                               transform=transform_test)
        n_classes = 10
    elif datatype.lower() == CIFAR100:
        print("Using CIFAR100 dataset.")
        trainset = torchvision.datasets.CIFAR100(root=train_db_path,
                                                 train=True, download=True,
                                                 transform=transform_train)
        testset = torchvision.datasets.CIFAR100(root=test_db_path,
                                                train=False, download=True,
                                                transform=transform_test)
        n_classes = 100
    else:
        print("Dataset is not supported.")
        return None, None, None

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=4)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=4)
    return trainloader, testloader, n_classes

def _getdatatransformswm():
    transform_wm = transforms.Compose([
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    return transform_wm

def getwmloader(wm_path, batch_size, labels_path):
    transform_wm = _getdatatransformswm()
    # load watermark images
    wmloader = None

    wmset = ImageFolderCustomClass(
        wm_path,
        transform_wm)
    img_nlbl = []
    wm_targets = np.loadtxt(os.path.join(wm_path, labels_path))
    for idx, (path, target) in enumerate(wmset.imgs):
        img_nlbl.append((path, int(wm_targets[idx])))
    wmset.imgs = img_nlbl

    wmloader = torch.utils.data.DataLoader(
        wmset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=False)

    return wmloader

def get_split_wmloader(wm_path, batch_size, labels_path):
    transform_wm = _getdatatransformswm()
    # load watermark images
    wmloader = None

    wmset = ImageFolderCustomClass(
        wm_path,
        transform_wm)
    img_nlbl = []
    wm_targets = np.loadtxt(os.path.join(wm_path, labels_path))
    for idx, (path, target) in enumerate(wmset.imgs):
        img_nlbl.append((path, int(wm_targets[idx])))
    wmset.imgs = img_nlbl


    length = len(wmset)
    train_dataset, test_dataset = torch.utils.data.dataset.random_split(
        dataset=wmset,
        lengths=[int(length*0.8), length - int(length*0.8)],
        generator=torch.manual_seed(0)
    )

    train_wmloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=False)
    
    test_wmloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=False)

    return train_wmloader, test_wmloader