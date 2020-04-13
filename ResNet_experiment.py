"""
Objective is to make the NN model once that I have planned.
"""
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
device = torch.device("cuda:0")

resnet = torchvision.models.resnet50(pretrained=True)
base = nn.Sequential(*list(resnet.children())[:6]) # Keeping this upto end of layer-2
bottleneck = resnet.layer2[3]

ResPose = nn.Sequential()
ResPose.add_module("ResNet", base)
ResPose.add_module("Cas-0", bottleneck)
ResPose.add_module("Cas-1", bottleneck)
ResPose.add_module("Cas-2", bottleneck)
ResPose.add_module("Cas-3", bottleneck)
ResPose.eval()


for i ,layer in enumerate(resnet.children()):
    print(i)
    print(layer)



# for param in resnet.parameters():
#     param.requires_grad = False

