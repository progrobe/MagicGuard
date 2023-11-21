## To Destruct, or Not to Destruct: Defending Watermark Removal Attacks on DNN Models via Self-destruction
This repository is an Pytorch implementation of MagicGuard which is proposed in the paper "To Destruct, or Not to Destruct: Defending Watermark Removal Attacks on DNN Models via Self-destruction". In this repository, we demonstrate how to enhance pre-trained models with MagicGuard. The code is tested on two real world tasks, including image classification and face recognition. Three popular datasets are employed, including ImageNet, CIFAR100 and CelebA. 


## Introduction to MagicGuard
Recent studies have shown that existing DNN watermarking schemes are not robust as the watermarks could be easily removed when model fine-tuning is involved. As a counter measure, MagicGuard works as a safeguard that protects the model from fine-tuning attack. It introduces proprietary neurons with obfuscated activation function to mask the gradient in backpropagation and further prevent the gradient convergence. When the model is protected by MagicGuard, its functionality of benign sample prediction will be destroyed when model fine-tuning is involved at any magnitudes. In extreme cases, the adversaries obtained full knowledge of the training dataset and had the access to fine-tune all layers, MagicGuard could still protect the model successfully. 

The following figures show the comparison of loss landscapes between a raw model without protection (left) and the same model enhanced by our MagicGuard (right). An illustration of the optimization process is also shown in the upper right of the corresponding figure. 

![image](https://github.com/progrobe/MagicGuard/assets/67232034/4a741a4d-f953-4ba7-9785-b1e421561db0)

## Requirement

Configure the environment

```
python=3.7
tensorflow=1.14.0
numpy=1.21.5
imageio=2.16
```

## How to use

source code for experiments on cifar100 is in MagicGuard/magic_guard1_cifar100/  

source code for experiments on celebA is in MagicGuard/magic_guard2_celebA/  

source code for experiments on ImageNet is in MagicGuard/magic_guard3_imagenet/  

source code for experiments on physical watermark is in MagicGuard/magic_guard4_physical/  

source code for robustness evaluation is also in MagicGuard/magic_guard2_celebA/  

