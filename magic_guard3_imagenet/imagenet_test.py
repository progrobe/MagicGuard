from cmath import sin
import os
import warnings
import sys

import numpy as np
import torchvision
import torch
import cv2
from image_folder_custom_class import ImageFolderCustomClass

os.environ["CUDA_VISIBLE_DEVICES"]="0"
warnings.filterwarnings("ignore", category=UserWarning)

class ImageNet:

    def __init__(self):
        self.model = None
        self.model = torchvision.models.resnet18(pretrained=True)
        self.train_path = '/home/mist/magic_guard3_imagenet/data/imagenet_val/train/'
        self.test_path = '/home/mist/magic_guard3_imagenet/data/imagenet_val/test/'
        self.ood_wm_path = '/home/mist/relight/ziyi/data/ood_wm/'
        self.content_wm_path = '/home/mist/relight/ziyi/data/image_wmcontent/image_content'    
        self.ood_wm_lbl = 'labels-cifar.txt'
        self.content_wm_lbl = 'labels-class-4.txt'
        self.wm_batch_size = 1
        self.batch_size = 64
        self.epoch = 10
        self.lr = 0.0001
        self.device = torch.device("cuda:0")
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
        print('finished testing: ', (correct / total).item())
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
        print("len:",length)
        # get a train/val dataset
        train_dataset, val_dataset = torch.utils.data.dataset.random_split(
            dataset=dataset,
            lengths=[int(length*0.7), length - int(length*0.7)],
            generator=torch.manual_seed(0)
        )

        # get a smaller dataset from val dataset
        # used in training process (fast test)
        val_length = len(val_dataset)
        _, fast_val_dataset = torch.utils.data.dataset.random_split(
            dataset=val_dataset,
            lengths=[int(val_length*0.98), val_length - int(val_length*0.98)],
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

    def get_trainloader(self, path):
        data_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(224),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
        ])
        dataset = torchvision.datasets.ImageFolder(
            root=path, transform=data_transform)

        self.train_loader = torch.utils.data.dataloader.DataLoader(
            dataset=dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.worker)
       
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

        self.test_loader = torch.utils.data.dataloader.DataLoader(
            dataset=dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.worker)
        
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

    def embed(self):
        print('start embedding')

        self.model.to(self.device)
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = torch.nn.CrossEntropyLoss()
        wmloader = self.get_ood_wmloader(self.ood_wm_path)
        # wmloader = self.get_content_wmloader(self.content_wm_path)
        if self.train_loader == None:
            self.get_trainloader(self.train_path)
        trainloader = self.train_loader
        if self.test_loader == None:
            self.get_testloader(self.test_path)
        test_loader = self.test_loader

        for epoch in range(1, self.epoch + 1):
            print(f"Learning Rate: {self.lr}")
            correct = 0
            count = 0
            running_loss = 0

            wminputs, wmtargets = [], []
            for wm_idx, (wminput, wmtarget) in enumerate(wmloader):
                wminput, wmtarget = wminput.to(self.device), wmtarget.to(self.device)
                wminputs.append(wminput)
                wmtargets.append(wmtarget)
            wm_idx = np.random.randint(len(wminputs))

            for batch_idx, (x, y) in enumerate(trainloader):
                x, y = x.to(self.device), y.to(self.device)
                # add wmimages and targets
                #wm_idx是起始id，每个batch加入一个wm的batch，由于是初始的batch id是随机的，所以需要%len来回到第一个wm
                #wm和正常数据的比例是 wm_batchsize:batch_size
                x = torch.cat([x, wminputs[(wm_idx + batch_idx) % len(wminputs)]], dim=0)
                y = torch.cat([y, wmtargets[(wm_idx + batch_idx) % len(wminputs)]], dim=0)

                optimizer.zero_grad()
                outputs = self.model(x)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
                running_loss += loss.data

                _, predicted = torch.max(outputs.data, 1)
                count += y.size(0)
                correct += predicted.eq(y.data).cpu().sum()

                #n个batch输一次当前batch的acc、测一次wm准确率
                if batch_idx % 100 == 0:
                    last_n_batch_acc = correct/count
                    print('[%d, %5d] loss:%9f acc:%.3f'
                          % (epoch, batch_idx, running_loss/200, last_n_batch_acc))
                    running_loss = 0
                    correct = 0
                    count = 0

                    print("wmtest:",end='')
                    # content需要用测试集来检测训练效果
                    # 普通content的测试集和训练集的准确率基本上是一样的
                    wm_acc = self.test(wmloader)

                    if wm_acc > 0.94:
                        print("finished embedding batch index: ",batch_idx)
                        print('------------finished embedding-----------')
                        #原数据的保真度
                        print("fidelity:",end='')
                        self.test(test_loader)
                        print("saving...")
                        torch.save(self.model, './checkpoint/resnet.pkl')
                        return

    def fine_tune(self):
        self.model.to(self.device)
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = torch.nn.CrossEntropyLoss()
        # test_wmloader = self.get_ood_wmloader(self.ood_wm_path)
        test_wmloader = self.get_content_wmloader(self.content_wm_path)
        if self.train_loader == None:
            self.get_trainloader(self.train_path)
        trainloader = self.train_loader
        if self.test_loader == None:
            self.get_testloader(self.test_path)
        test_loader = self.test_loader
        
        for epoch in range(1, self.epoch + 1):
            # #wm准确率
            # print("wm :", end='')
            # self.test(test_wmloader)
            # #正常功能准确率
            # print("test:", end='')
            # self.test(test_loader)

            print("epoch:", epoch)
            print(f"Learning Rate: {self.lr}")
            for batch_idx, (x, y) in enumerate(trainloader):
                x, y = x.to(self.device), y.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(x)
                # print(outputs)
                # print(outputs.max())
                # print(outputs.min())
                # sys.exit()
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
                # if batch_idx % 5 == 1:
                #     print('batch:', batch_idx)
                #     print("wm :", end='')
                #     self.test(test_wmloader)
                #     print("test:", end='')
                #     self.test(test_loader)
                    # if batch_idx == 34:
                    #     return
            #wm准确率
            print("wm :", end='')
            self.test(test_wmloader)
            #正常功能准确率
            print("test:", end='')
            self.test(test_loader)


if __name__ == '__main__':
    print("-------------begin-------------")
    big = True
    imagenet = ImageNet()
    imagenet.epoch = 1
    imagenet.lr = 0.0001
    # imagenet.model = torch.load('/home/mist/magic_guard3/checkpoint/resnet_ood.pkl', map_location=imagenet.device)
    # imagenet.model = torch.load('/home/mist/magic_guard3/checkpoint/resnet_content.pkl', map_location=imagenet.device)
    # imagenet.model = torch.load('/home/mist/magic_guard3/checkpoint/resnet_ood_sin.pkl', map_location=imagenet.device)
    # imagenet.model = torch.load('/home/mist/magic_guard3/checkpoint/resnet_content_sin.pkl', map_location=imagenet.device)

    choice = 'function'
    if choice == 'layers':
        for i in range(6):
            sin_layer = 6 - i
            e = ''
            # e = '100'
            imagenet.model = torch.load(f'/home/mist/magic_guard3/checkpoint/layers{e}/resnet_content_sin_{sin_layer}.pkl', map_location=imagenet.device)
            print(f'layer{e}:', sin_layer)
            imagenet.fine_tune()
    elif choice == 'ratio':
        for ratio in [1,3,5,10,20,40,60,80,100]:
            e = ''
            # e = '100'
            imagenet.model = torch.load(f'/home/mist/magic_guard3_imagenet/checkpoint/ratio{e}/resnet_content_sin_{ratio}.pkl', map_location=imagenet.device)
            print(f'ratio{e}:', ratio)
            imagenet.fine_tune()
    elif choice == 'function':
        # for mode in [0,1,2]:
        #     if mode == 0:
        #         e = ''
        #         imagenet.model = torch.load('/home/mist/magic_guard3_imagenet/checkpoint/resnet_content_sin.pkl', map_location=imagenet.device)
        #     else:
        #         e = ''
        #         # e = '100'
        #         imagenet.model = torch.load(f'/home/mist/magic_guard3_imagenet/checkpoint/mode{e}/resnet_content_function{mode}.pkl', map_location=imagenet.device)
        #     print(f'mode{e}:', mode)
        #     imagenet.fine_tune()
        mode = 2
        e = ''
        imagenet.model = torch.load(f'/home/mist/magic_guard3_imagenet/checkpoint/mode{e}/resnet_content_function{mode}.pkl', map_location=imagenet.device)
        print(f'mode{e}:', mode)
        imagenet.fine_tune()
    
    print(" -----------end----------------")