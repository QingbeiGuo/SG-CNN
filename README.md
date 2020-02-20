# Self-grouping Convolutional Neural Networks

## Introduction

Although group convolutions are increasingly used on deep convolutional neural networks to improve the computation efficiency and to reduce the number of parameters, most existing methods construct their group convolution architectures by predefinedly partitioning the filters of each convolutional layer into multiple regular filter groups with equally spatial group size and data-independence, which is disadvantageous to fully exploit their potential abilities. To tackle this issue, we propose a novel method of self-grouping convolutional neural networks, called as SG-CNN, in which the filters of each convolutional layer group themselves depending on the similarity of their importance vectors. Concretely, for each filter, we first evaluate the importance value of their input channels to achieve their importance vectors, and then group them by clustering based on their importance vectors, which is \emph{data-dependent}. According to the knowledge of clustering centroids, we then prune the less important connections for groups, which implicitly minimizes the accuracy loss from pruning, thus yielding a \emph{diverse} group convolution. Subsequently, we work out two fine-tuning schemes, i.e. (1) both local and global fine-tuning and (2) only global fine-tuning, which experimentally obtain comparable results, to recover the recognition capacity of the pruned network. Finally, we achieve an efficient and compact self-grouping convolutional neural network, exploiting the representation potential of group convolutions by flexible self-grouping structures. Moreover, our self-grouping approach is extended to the fully-connected layers for further compression and acceleration. The comprehensive experiments on MNIST, CIFAR-10/100, and ImageNet datasets demonstrate that our self-grouping convolution method adapts to various state-of-the-art CNN architectures, such as LeNet, ResNet, and DenseNet, significantly achieving superior performance in terms of compression ratios, speedups and recognition accuracies. Particularly, our SG-CNN achieves over 4.5$\times$ compression ratio and over 4$\times$ FLOPs reduction on DenseNet201 with 75.17\% top-1 accuracy and 92.6\% top-5 accuracy on ImageNet, which beats all existing counterparts of group convolutions. We further demonstrate the generalization ability of SG-CNN by transfer learning, including domain adaption and object detection, which significantly achieves competitive results. Our source code is available at \url{https://github.com/QingbeiGuo/SG-CNN.git}.

This project is a pytorch implementation of SG-CNN, aimed to compressing and accelerating deep convolutional neural networks. 

### What we are doing and going to do

- [x] Support pytorch-1.0.

## Classification

We benchmark our code thoroughly on four datasets: MNIST, CIFAR-10/100 and imagenet-1K for classification, using three different network architecture: LeNet, Resnet and Densenet. Below are the results:

1). LeNet on MNIST

model    | Params | FLOPs | Top-1 (%) | Top-5 (%)
---------|--------|-------|-----------|-----------
[Baseline]                       | 431K   | 2294K   | 99.40               | 100
[LeNet (Conv-95/FC-95)]| 22K    | 388K    | 99.17      | 100
[LeNet (Conv-96/FC-96)]| 18K    | 367K    | 99.10      | 100
[LeNet (Conv-97/FC-97)]| 14K    | 348K    | 99.02      | 100
[LeNet (Conv-98/FC-98)]| 9K     | 328K    | 98.91      | 100
[LeNet (Conv-99/FC-99)]| 5K     | 307K    | 98.53      | 99.98   

2). DenseNet on CIFAR-10/100

Comparison among several state-of-the-art methods for DenseNet121 on CIFAR-10

model    | Params | FLOPs | Top-1 (%) | Top-5 (%)
---------|--------|-------|-----------|-----------
[Baseline (k = 32)]        | 6.89M | 888.36M | 95.23 | 99.86
[DenseNet (Conv-75/FC-75)] | 1.71M | 221.90M | 95.40 | 99.91
[DenseNet (Conv-80/FC-80)] | 1.37M | 177.72M | 95.29 | 99.91
[DenseNet (Conv-85/FC-85)] | 1.03M | 134.10M | 95.39 | 99.90
[DenseNet (Conv-90/FC-90)] | 0.68M |  89.77M | 95.03 | 99.93
[DenseNet (Conv-95/FC-95)] | 0.34M |  45.76M | 94.32 | 99.89

Comparison among several state-of-the-art methods for DenseNet121 on CIFAR-100

model    | Params | FLOPs | Top-1 (%) | Top-5 (%)
---------|--------|-------|-----------|-----------
[Baseline (k = 32)]                 | 6.99M | 888.45M | 78.67 | 94.55
[DenseNet (Conv-70/FC-70)]          | 2.10M | 266.60M | 78.78 | 94.51
[DenseNet (Conv-75/FC-75)]          | 1.75M | 222.14M | 78.40 | 94.19
[DenseNet (Conv-80/FC-80)]          | 1.40M | 176.46M | 78.24 | 94.28
[DenseNet (Conv-85/FC-85)]          | 1.06M | 133.46M | 78.18 | 94.34
[DenseNet (Conv-90/FC-90)]          | 0.71M |  89.86M | 76.73 | 94.04
[DenseNet (Conv-95/FC-95)]          | 0.36M |  45.67M | 74.37 | 93.33

3). ResNet and DenseNet on ImageNet

Comparison among several state-of-the-art methods for ResNet50 on ILSVRC2012

model    | Params | FLOPs | Top-1 (%) | Top-5 (%)
---------|--------|-------|-----------|-----------
[Baseline](https://pan.baidu.com/s/1Pxm_TCHKQxC_8c-6-pTuEQ)                    | 25.55M | 4.09G   | 76.13   | 92.862
[ResNet-G (Conv-60/FC-60)](https://pan.baidu.com/s/1Pxm_TCHKQxC_8c-6-pTuEQ)    | 11.88M | 1.91G   | 75.20   | 92.55
[ResNet-G (Conv-70/FC-60)](https://pan.baidu.com/s/1Pxm_TCHKQxC_8c-6-pTuEQ)    |  9.83M | 1.55G   | 74.43   | 92.30
[ResNet-G (Conv-80/FC-60)](https://pan.baidu.com/s/1Pxm_TCHKQxC_8c-6-pTuEQ)    |  7.76M | 1.20G   | 73.22   | 91.70
[ResNet-LG (Conv-60/FC-60)](https://pan.baidu.com/s/1Pxm_TCHKQxC_8c-6-pTuEQ)   | 11.87M | 1.91G   | 75.12   | 92.59
[ResNet-LG (Conv-70/FC-60)](https://pan.baidu.com/s/1Pxm_TCHKQxC_8c-6-pTuEQ)   |  9.83M | 1.56G   | 74.42   | 92.31
[ResNet-LG (Conv-80/FC-60)](https://pan.baidu.com/s/1Pxm_TCHKQxC_8c-6-pTuEQ)   |  7.76M | 1.20G   | 73.38   | 91.69

Comparison among several state-of-the-art methods for DenseNet201 on ILSVRC2012

model    | Params | FLOPs | Top-1 (%) | Top-5 (%)
---------|--------|-------|-----------|-----------
[Baseline](https://pan.baidu.com/s/1Pxm_TCHKQxC_8c-6-pTuEQ)                    | 19.82M | 4.29G | 76.896  | 93.37
[DenseNet-G (Conv-70/FC-60)](https://pan.baidu.com/s/1Pxm_TCHKQxC_8c-6-pTuEQ)  |  6.00M | 1.34G | 76.21   | 93.07
[DenseNet-G (Conv-80/FC-60)](https://pan.baidu.com/s/1Pxm_TCHKQxC_8c-6-pTuEQ)  |  4.32M | 0.99G | 74.99   | 92.55
[DenseNet-LG (Conv-70/FC-60)](https://pan.baidu.com/s/1Pxm_TCHKQxC_8c-6-pTuEQ) |  6.00M | 1.34G | 76.12   | 93.06
[DenseNet-LG (Conv-80/FC-60)](https://pan.baidu.com/s/1Pxm_TCHKQxC_8c-6-pTuEQ) |  4.32M | 0.99G | 75.17   | 92.60


## Domain Adaptation

Comparison of different compressed models for fine-grained classification on CUB-200

model    | Params | FLOPs | Top-1 (%) | Top-5 (%)
---------|--------|-------|-----------|-----------
Baseline                    | 23.86M  | 4.09G   | 74.37           | 94.43
ResNet-L (Conv-60/FC-60)    | 11.46M  | 1.91G   | 76.82           | 94.96
ResNet-L (Conv-70/FC-60)    |  9.42M  | 1.56G   | 76.61           | 94.96
ResNet-L (Conv-80/FC-60)    |  7.35M  | 1.20G   | 75.18           | 94.60
ResNet-G (Conv-60/FC-60)    | 11.47M  | 1.91G   | 73.04           | 94.01
ResNet-G (Conv-70/FC-60)    |  9.42M  | 1.55G   | 72.75           | 93.65
ResNet-G (Conv-80/FC-60)    |  7.35M  | 1.20G   | 71.92           | 92.89
ResNet-LG (Conv-60/FC-60)   | 11.46M  | 1.91G   | 73.25           | 93.63
ResNet-LG (Conv-70/FC-60)   |  9.42M  | 1.56G   | 73.11           | 93.22
ResNet-LG (Conv-80/FC-60)   |  7.35M  | 1.20G   | 71.94           | 93.32
---------|--------|-------|-----------|-----------
Baseline                    | 18.28M  | 4.29G   | 78.65           | 95.46
DenseNet-L (Conv-70/FC-60)  |  5.61M  | 1.34G   | 77.93           | 95.44
DenseNet-L (Conv-80/FC-60)  |  3.93M  | 0.94G   | 77.17           | 95.05
DenseNet-G (Conv-70/FC-60)  |  5.66M  | 1.35G   | 77.20           | 94.70
DenseNet-G (Conv-80/FC-60)  |  3.94M  | 0.94G   | 75.73           | 94.25
DenseNet-LG (Conv-70/FC-60) |  5.61M  | 1.34G   | 77.46           | 94.98
DenseNet-LG (Conv-80/FC-60) |  3.93M  | 0.94G   | 75.66           | 94.56


## Object Detection

Object detection results on MS COCO. Here, mAP-1 and mAP-2 correspond to 300$\times$ and 600$\times$ input resolutions, respectively. mAP is reported with COCO primary challenge metric (AP@IoU=0.50:0.05:0.95)

model    | Params | FLOPs   | mAP(300$\times$) (%) | mAP(600$\times$) (%)
---------|--------|---------|-----------|-----------
Baseline                    | 24.44M | 4.09G   | 24.5          | 30.9
ResNet-G (Conv-60/FC-60)    | 12.00M | 1.91G   | 24.8          | 30.9
ResNet-G (Conv-70/FC-60)    |  9.95M | 1.55G   | 24.2          | 29.5
ResNet-G (Conv-80/FC-60)    |  7.88M | 1.20G   | 23.2          | 28.6
ResNet-LG (Conv-60/FC-60)   | 11.99M | 1.91G   | 24.9          | 30.9
ResNet-LG (Conv-70/FC-60)   |  9.95M | 1.56G   | 24.2          | 30.0
ResNet-LG (Conv-80/FC-60)   |  7.88M | 1.20G   | 23.0          | 28.2
---------|--------|-------|-----------|-----------
Baseline                    | 18.78M | 4.29G   | 26.0          | 32.8
DenseNet-G (Conv-70/FC-60)  |  6.15M | 1.35G   | 23.9          | 30.3
DenseNet-G (Conv-80/FC-60)  |  4.44M | 0.94G   | 22.7          | 28.7
DenseNet-LG (Conv-70/FC-60) |  6.11M | 1.34G   | 24.0          | 30.3
DenseNet-LG (Conv-80/FC-60) |  4.43M | 0.94G   | 23.0          | 28.9

## Train

For clssification：

(1) group=True for getting the pruned models;  
(2) finetune=True for globally fine-tuning the pruned models.

For objection detection：

CUDA_VISIBLE_DEVICES=0 python trainval_net.py --dataset coco --net res50 --bs 16 --nw 8 --lr 0.01 --lr_decay_step 4 --epochs 10  --cuda --mGPUs  
CUDA_VISIBLE_DEVICES=0 python trainval_net.py --dataset coco --net dense201 --bs 16 --nw 16 --lr 0.01 --lr_decay_step 4 --epochs 10  --cuda --mGPUs

## Authorship

This project is contributed by [Qingbei Guo](https://github.com/QingbeiGuo).

## Citation
