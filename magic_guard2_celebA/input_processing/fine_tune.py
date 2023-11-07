import math
import os
import warnings

import numpy as np
import torchvision
import torch
from image_folder_custom_class import ImageFolderCustomClass
import torch.nn as nn
from models import vgg16_sin

os.environ["CUDA_VISIBLE_DEVICES"]="0"
warnings.filterwarnings("ignore", category=UserWarning)

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
        
    def get_content_wmloader(self,wmpath):
        wm_path = wmpath
        batch_size = self.wm_batch_size
        labels_path = self.content_wm_lbl
        transform_wm = torchvision.transforms.Compose([
            torchvision.transforms.Resize(224),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225]),
        ])
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
            wmset, batch_size=batch_size, shuffle=True,
            num_workers=self.wm_worker, pin_memory=False)

        return wmloader

    def get_ood_wmloader(self, wmpath):
        wm_path = wmpath
        batch_size = self.wm_batch_size
        labels_path = self.ood_wm_lbl
        transform_wm = torchvision.transforms.Compose([
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225]),
        ])
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
            wmset, batch_size=batch_size, shuffle=True,
            num_workers=self.wm_worker, pin_memory=False)

        return wmloader

    def fine_tune(self):
        self.model.to(self.device)
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = torch.nn.CrossEntropyLoss()
        # test_wmloader = self.get_ood_wmloader(self.ood_wm_path)
        test_wmloader = self.get_content_wmloader(self.content_wm_path)
        if self.train_loader == None:
            self.get_dataloader(self.train_path)
        trainloader, valloader = self.train_loader, self.val_loader
        fast_val_loader = self.fast_val_loader
        self.get_testloader(self.test_path)
        test_loader = self.test_loader
        
        for epoch in range(1, self.epoch + 1):
            #wm准确率
            print("wm :", end='')
            self.test(test_wmloader)
            #正常功能准确率
            print("test:", end='')
            self.test(test_loader)

            print("epoch:", epoch)
            print(f"Learning Rate: {self.lr}")
            for batch_idx, (x, y) in enumerate(trainloader):
                x, y = x.to(self.device), y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(x)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()

                # # n个batch输一次当前batch的train acc、测一次wm准确率
                # if batch_idx % 10 == 0:
                #     print('[batch:%5d]'
                #           % (batch_idx))

                #     # small content需要用测试集来检测训练效果
                #     # 普通content的测试集和训练集的准确率基本上是一样的
                #     print("wmt :",end='')
                #     self.test(test_wmloader)

                #     # #原数据的保真度
                #     print("test:",end='')
                #     self.test(test_loader)
        

def test_content():
    print('-------------------------------------')
    print("train wm acc:")
    train_wmloader = imagenet.get_content_wmloader('../relight/ziyi/data/wm_content_small_train')
    imagenet.test(train_wmloader)

    print("testwm acc:")
    test_wmloader = imagenet.get_content_wmloader('../relight/ziyi/data/wm_content_small_test')
    imagenet.test(test_wmloader)
    print('--------------------------------------')

def test_ood():
    print('-------------------------------------')
    print("ood wm acc:")
    ood_wmloader = imagenet.get_ood_wmloader('../relight/ziyi/data/ood_wm')
    imagenet.test(ood_wmloader)
    print('--------------------------------------')


if __name__ == '__main__':
    print("-------------begin-------------")
    imagenet = ImageNet()
    imagenet.epoch = 10
    imagenet.lr = 0.0001

    # imagenet.model = torch.load('./checkpoint/celeba/ood_vgg_sin.pkl')
    # imagenet.model = torch.load('./checkpoint/celeba/ood_vgg.pkl')
    # imagenet.model = torch.load('./checkpoint/celeba/content_vgg_sin.pkl')
    imagenet.model = torch.load('./checkpoint/celeba/content_vgg.pkl')

    # imagenet.fine_tune()

    # test_ood()
    # test_content()

    # choice = 'lightness'
    # choice = 'blur'
    choice = 'saturation'
    degree = 5.5
    imagenet.get_testloader(f'/home/mist/magic_guard2_celebA/input processing/{choice}_data{degree}')
    imagenet.test(imagenet.test_loader)
    print("normal:")
    imagenet.get_testloader(imagenet.test_path)
    imagenet.test(imagenet.test_loader)
    
    print(" -----------sin:----------------")

    imagenet.model = torch.load('./checkpoint/celeba/content_vgg_sin.pkl')
    print('lightness:')
    imagenet.get_testloader(f'/home/mist/magic_guard2_celebA/input processing/{choice}_data{degree}')
    imagenet.test(imagenet.test_loader)
    print("normal:")
    imagenet.get_testloader(imagenet.test_path)
    imagenet.test(imagenet.test_loader)
    print(" -----------end----------------")









