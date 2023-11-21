## To Destruct, or Not to Destruct: Defending Watermark Removal Attacks on DNN Models via Self-destruction
This repository is an Pytorch implementation of MagicGuard which is proposed in the paper "To Destruct, or Not to Destruct: Defending Watermark Removal Attacks on DNN Models via Self-destruction". In this repository, we demonstrate how to enhance pre-trained models with MagicGuard. The code is tested on two real world tasks, including image classification and face recognition. Three popular datasets are employed, including ImageNet, CIFAR100 and CelebA. 


## Introduction to MagicGuard
Recent studies have shown that existing DNN watermarking schemes are not robust as the watermarks could be easily removed when model fine-tuning is involved. As a counter measure, MagicGuard works as a safeguard that protects the model from fine-tuning attack. It introduces proprietary neurons with obfuscated activation function to mask the gradient in backpropagation and further prevent the gradient convergence. When the model is protected by MagicGuard, its functionality of benign sample prediction will be destroyed when model fine-tuning is involved at any magnitudes. In extreme cases, the adversaries obtained full knowledge of the training dataset and had the access to fine-tune all layers, MagicGuard could still protect the model successfully. 

The following figures show the comparison of loss landscapes between a raw model without protection (left) and the same model enhanced by our MagicGuard (right). An illustration of the optimization process is also shown in the upper right of the corresponding figure. 

![image](https://github.com/progrobe/MagicGuard/assets/67232034/991aaf06-97c4-4e1e-8ebc-d4361112e7d1)

## Requirement

Configure the environment

```
python=3.7
tensorflow=1.14.0
numpy=1.21.5
imageio=2.16
```

## How to use

- Source code for experiments on cifar100 is in MagicGuard/magic_guard1_cifar100/ 

- Source code for experiments on celebA is in MagicGuard/magic_guard2_celebA/  

- Source code for experiments on ImageNet is in MagicGuard/magic_guard3_imagenet/  

- Source code for experiments on physical watermark is in MagicGuard/magic_guard4_physical/  

- Source code for robustness evaluation is also in MagicGuard/magic_guard2_celebA/


**Pipeline 1**: The model owner first train a model on a clean dataset and conduct watermark embedding with a trusted watermarking method. Then the adversary conducts fine-tuning attack to remove the watermark. 

The following command is an example of a model embeded by physical watermarks, and is later attacked by an adversary through fine-tuning. The output of the code will show the success of the attack.
```bash
cd MagicGuard/magic_guard4_physical/
python run.py --MagicGuard False
```

**Pipeline 2**: The model owner first train a model on a clean dataset and conduct watermark embedding with a trusted watermarking method. Next, the model owner enhance the model with MagicGuard. Then the adversary conducts fine-tuning attack to remove the watermark. 

The following command is an example of a model embeded by physical watermarks and it is enhanced by MagicGuard. Then it was attacked by an adversary through fine-tuning. The output of the code will show the failure of the attack.
```bash
cd MagicGuard/magic_guard4_physical/
python run.py --MagicGuard True
```

An examplary script for enhancing a watermarked model with MagicGuard is provided in `MagicGuard/magic_guard3_imagenet/convert.py`. By setting the flag`--mode [neuron/layers]`, MagicGuard will be injected in neuron level or layer level. To change the activation function of MagicGuard, add the flag`--selected_func [0/1/2]`. 
```bash
cd MagicGuard/magic_guard3_imagenet/
python convert.py --mode neuron --selected_func 0
```

There are a number of arguments that could be used to set the hyperparameters. The interpretation and configuration of these hyperparameters are explained in our paper.

## Results
The following image shows the differences between the raw model without any protection and the same model enhanced by MagicGuard in predicting benign samples and verification samples when suffering model fine-tuning. In the raw model without any protection (left), the watermarks could be easily removed while the functionality of benign sample prediction is well preserved. In the model enhanced by our MagicGuard (right), the functionality of both benign sample and verification sample prediction are all destroyed.
![image](https://github.com/progrobe/MagicGuard/assets/67232034/d68bceb1-ac17-47fc-ba72-488b0ae1a573)


The following image shows the accuracy drop of benign sample prediction when injecting MagicGuard in layer level (left) and neuron level (right). The immediate drop when suffering from fine-tuning attack demonstrates the effectiveness of MagicGuard.
![image](https://github.com/progrobe/MagicGuard/assets/67232034/ce8398e2-4d95-4daf-a70f-fb1d6c758958)

