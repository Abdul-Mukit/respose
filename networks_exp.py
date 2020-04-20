import torch.nn as nn
import torchvision
from torchvision.models.resnet import Bottleneck, conv3x3, conv1x1

# Excluding the last relu, as belief and affinity maps have negative values too
class BottleneckMod(Bottleneck):
    def __init__(self, inplanes, planes):
        super(BottleneckMod, self).__init__(inplanes, planes)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        # out = self.relu(out)

        return out

# class BottleneckMod(nn.Module):
#     expansion = 4
#     __constants__ = ['downsample']
#
#     def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
#                  base_width=64, dilation=1, norm_layer=None):
#         super(BottleneckMod, self).__init__()
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         width = int(planes * (base_width / 64.)) * groups
#         # Both self.conv2 and self.downsample layers downsample the input when stride != 1
#         self.relu = nn.ReLU(inplace=True)
#         self.conv1 = conv1x1(inplanes, width)
#         self.bn1 = norm_layer(width)
#         self.conv2 = conv3x3(width, width, stride, groups, dilation)
#         self.bn2 = norm_layer(width)
#         self.conv3 = conv1x1(width, planes * self.expansion)
#         self.bn3 = norm_layer(planes * self.expansion)
#
#         self.downsample = downsample
#         self.stride = stride
#
#     def forward(self, x):
#         identity = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)
#
#         out = self.conv3(out)
#         out = self.bn3(out)
#
#         if self.downsample is not None:
#             identity = self.downsample(x)
#
#         out += identity
#         # out = self.relu(out)
#
#         return out


class ResPoseNetwork2(nn.Module):
    def __init__(self, pretrained=True, numBeliefMap=9, numAffinity=16):
        super(ResPoseNetwork2, self).__init__()
        self.pretrained = pretrained
        self.numBeliefMap = numBeliefMap
        self.numAffinity = numAffinity

        resnet = torchvision.models.resnet50(pretrained=pretrained)

        # for param in resnet.parameters():
        #     param.requires_grad = False

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2

        # Defining the Cascades
        self.cas1 = nn.Sequential(BottleneckMod(512, 128), BottleneckMod(512, 128))
        self.cas2 = nn.Sequential(BottleneckMod(512, 128), BottleneckMod(512, 128))
        self.cas3 = nn.Sequential(BottleneckMod(512, 128), BottleneckMod(512, 128))
        self.cas4 = nn.Sequential(BottleneckMod(512, 128), BottleneckMod(512, 128))

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


