#pytorch -0.2.1
#python -3.6.2
#torchvision - 0.1.9

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import glob
import os

def train_loader(path, batch_size=64, num_workers=1, pin_memory=False):
    normalize = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])    
    
    return data.DataLoader(
        datasets.CIFAR100(root=path,                              
                         train=True,
                         download=False,
                         transform=transform_train),
        batch_size=batch_size,                                  
        shuffle=True,                                          
        num_workers=num_workers,                               
        pin_memory=pin_memory)

def test_loader(path, batch_size=64, num_workers=1, pin_memory=False):
    normalize = transforms.Normalize((0.4914, 0.4824, 0.4467), (0.2471, 0.2435, 0.2616))

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])    
    
    return data.DataLoader(
        datasets.CIFAR100(root=path,                              
                         train=False,
                         download=False,
                         transform=transform_test),
        batch_size=batch_size,                                  
        shuffle=False,                                          
        num_workers=num_workers,                                
        pin_memory=pin_memory)
