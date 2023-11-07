import math
import os
import warnings

import numpy as np
import torchvision
import torch
from image_folder_custom_class import ImageFolderCustomClass
import torch.nn as nn
from models import vgg16, alexnet, densenet

os.environ["CUDA_VISIBLE_DEVICES"]="0"
warnings.filterwarnings("ignore", category=UserWarning)

class ImageNet:
    def __init__(self):
        self.device = torch.device("cuda:0")
        self.model = torch.load('./torch model/vggface.pkl', map_location=self.device)
        self.train_path = '../relight/ziyi/data/train'
        self.ood_wm_path = '../relight/ziyi/data/ood_wm'
        self.content_wm_path = '../relight/ziyi/data/wm_content_small_train/'        
        self.ood_wm_lbl = 'labels-cifar.txt'
        self.content_wm_lbl = 'labels-class-4.txt'
        self.wm_batch_size = 1
        self.batch_size = 64
        self.epoch = 30
        self.lr = 0.0001
        self.worker = 8
        self.wm_worker = 8
        

        self.train_loader = None
        self.val_loader = None
        self.fast_val_loader = None

    def train(self):
        """训练self.model"""
        # self.model.to(self.device)
        self.model.cuda()
        # self.model=nn.DataParallel(self.model, device_ids=[1,2])
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = torch.nn.CrossEntropyLoss()
        if self.train_loader == None:
            self.get_dataloader(self.train_path)
        train_loader = self.train_loader

        for epoch in range(1, self.epoch + 1):
            print(f"Learning Rate: {self.lr}")

            correct = 0
            count = 0
            running_loss = 0
            for batch_id, (x, y) in enumerate(train_loader):
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                output = self.model(x)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()
                running_loss += loss.data

                pred = output.max(1, keepdim=True)[1]
                correct += pred.eq(y.view_as(pred)).sum().item()
                count += output.size(0)

                if batch_id % 10 == 0:
                    last_n_batch_acc = correct/count
                    print('[%d, %5d] loss:%.3f acc:%.3f'
                          % (epoch, batch_id, running_loss/10, last_n_batch_acc))
                    running_loss = 0
                    correct = 0
                    count = 0

            print("Finished Training")
        # TODO: use different name when there are different models
        # print("Saving model.......")
        # torch.save(self.model, './torch model/vggface_multigpu.pkl')
        # torch.save(self.model.state_dict(), './torch model/vggface_multigpu_params.pkl')

        # torch.save(self.model, './torch model/inception.pkl')
        # torch.save(self.model.state_dict(), './torch model/inception_params.pkl')

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
        print("len:",length)
        # print(math.ceil(length*0.7))
        # print(length*0.3)
        # get a train/val dataset
        train_dataset, val_dataset = torch.utils.data.dataset.random_split(
            dataset=dataset,
            lengths=[math.ceil(length*0.7), int(length- length*0.7)],
            generator=torch.manual_seed(0)
        )
        # get a smaller dataset from val dataset
        # used in training process (fast test)
        val_length = len(val_dataset)
        print(val_length)
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

    def embed(self):
        print('start embedding')
        self.model = torch.load('./torch model/vggface.pkl', map_location=self.device) 
        self.model.cuda()
        self.model.to(self.device)
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = torch.nn.CrossEntropyLoss()
        if self.train_loader == None:
            self.get_dataloader(self.train_path)
        trainloader, valloader = self.train_loader, self.val_loader
        fast_val_loader = self.fast_val_loader

        #OOD
        wmloader = self.get_ood_wmloader(self.ood_wm_path)
        #Content
        # wmloader = self.get_content_wmloader(self.content_wm_path)
        # test_wmloader = self.get_content_wmloader('../relight/ziyi/data/wm_content_small_test')
        
        # #开始时的wm准确率
        print("wm before embedding:", end='')
        self.test(wmloader)
        # 原数据的保真度
        print("fidelity before embedding:", end='')
        self.test(valloader)
        print("begin training..............")
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

                #n个batch输一次当前batch的train acc、测一次wm准确率
                if batch_idx % 10 == 0:
                    last_n_batch_acc = correct/count
                    print('[%d, %5d] loss:%9f acc:%.3f'
                          % (epoch, batch_idx, running_loss/200, last_n_batch_acc))
                    running_loss = 0
                    correct = 0
                    count = 0

                    print("wmtest:",end='')
                    # small content需要用测试集来检测训练效果
                    # 普通content的测试集和训练集的准确率基本上是一样的
                    # wm_acc = self.test(test_wmloader)
                    wm_acc = self.test(wmloader)

                    if wm_acc > 0.945:
                        print("finished embedding batch index: ",batch_idx)
                        print('------------finished embedding-----------')
                        #原数据的保真度
                        print("valloder fidelity:",end='')
                        self.test(valloader)
                        print("saving model...")
                        # torch.save(self.model, './checkpoint/celeba/vgg.pkl')
                        torch.save(self.model, './checkpoint/celeba/ood_vgg.pkl')
                        return

        # #will never be here...


def test_content():
    print('-------------------------------------')
    print("train wm acc:")
    train_wmloader = imagenet.get_content_wmloader('../relight/ziyi/data/wm_content_small_train')
    print('len:', len(train_wmloader))
    imagenet.test(train_wmloader)

    print("testwm acc:")
    test_wmloader = imagenet.get_content_wmloader('../relight/ziyi/data/wm_content_small_test')
    print('len:', len(test_wmloader))
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

    imagenet.embed()

    # imagenet.model = torch.load('./model/test/small_content.pkl')
    # imagenet.model = torch.load('./torch model/vggface.pkl')

    # 检测
    test_ood()
    # test_content()
    print(" -----------end----------------")









