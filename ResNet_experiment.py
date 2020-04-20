import torch.nn as nn
import torchvision
from torchvision.models.resnet import Bottleneck, conv3x3


class ResPoseNetwork2(nn.Module):
    def __init__(self, pretrained=True, numBeliefMap=9, numAffinity=16):
        super(ResPoseNetwork2, self).__init__()
        self.pretrained = pretrained
        self.numBeliefMap = numBeliefMap
        self.numAffinity = numAffinity

        resnet = torchvision.models.resnet50(pretrained=pretrained)

        for param in resnet.parameters():
            param.requires_grad = False
        num_ftrs = resnet.fc.in_features
        resnet.fc = nn.Linear(num_ftrs, 2)

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2

        # Defining the Cascades
        self.cas1 = nn.Sequential(Bottleneck(512, 128), Bottleneck(512, 128))
        self.cas2 = nn.Sequential(Bottleneck(512, 128), Bottleneck(512, 128))
        self.cas3 = nn.Sequential(Bottleneck(512, 128), Bottleneck(512, 128))
        self.cas4 = nn.Sequential(Bottleneck(512, 128), Bottleneck(512, 128))

        # Defining the final Maps Layer
        self.maps = conv3x3(512, numBeliefMap + numAffinity)

    def forward(self, x):
        '''Runs inference on the neural network'''
        numBeliefMap = self.numBeliefMap
        numAffinity = self.numAffinity
        numOutMaps = numBeliefMap + numAffinity ## TODO: in future will include * numClass

        # Same Resnet50 upto Layer-2
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)

        # Cascades
        out1 = self.cas1(x)
        out2 = self.cas2(out1)
        out3 = self.cas3(out2)
        out4 = self.cas4(out3)

        # Maps layer
        out5 = self.maps(out4)

        return [out1[:, :numOutMaps, :, :],
                out2[:, :numOutMaps, :, :],
                out3[:, :numOutMaps, :, :],
                out4[:, :numOutMaps, :, :],
                out5]


