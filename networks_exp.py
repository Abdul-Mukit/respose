import torch.nn as nn
import torchvision
from torchvision.models.resnet import Bottleneck, conv3x3, BasicBlock
import torch
from torchvision import models


class BottleneckMod(Bottleneck):
    '''
    * Excluding the last relu, as belief and affinity maps have negative values too
    * Including a relu at the beginning. As no relu at end, cascading this block creates a situation with no relu in
    between blocks.
    * Adding position argument to cater for the above mentioned 2 problems.
    * postion = 'Start' only when first bottleneck of cascade-1
    '''
    def __init__(self, inplanes, planes, position=''):
        super(BottleneckMod, self).__init__(inplanes, planes)
        self.position = position

    def forward(self, x):
        identity = x

        if self.position != 'Start':
            x = self.relu(x)  # added as no relu at the end, so when cascaded relu is always missing

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity # Always non-relu output
        # there was a relu here in original
        return out


class ResPoseNetwork2(nn.Module):
    def __init__(
            self,
            pretrained=False,
            numBeliefMap=9,
            numAffinity=16,
            stop_at_stage=6  # number of stages to process (if less than total number of stages)
    ):
        super(ResPoseNetwork2, self).__init__()

        self.pretrained = pretrained
        self.numBeliefMap = numBeliefMap
        self.numAffinity = numAffinity
        self.stop_at_stage = stop_at_stage

        # Including pretrained-resnet upto Layer2
        resnet = torchvision.models.resnet18(pretrained=pretrained)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2

        self.features = nn.Sequential(BasicBlock(128, 128),
                                      BasicBlock(128, 128))

        # Both Belief and Affnity calculation
        self.cas1 = ResPoseNetwork2.create_stage(128,
                                             numBeliefMap + numAffinity, True)
        self.cas2 = ResPoseNetwork2.create_stage(128 + numBeliefMap + numAffinity,
                                             numBeliefMap + numAffinity, False)
        self.cas3 = ResPoseNetwork2.create_stage(128 + numBeliefMap + numAffinity,
                                             numBeliefMap + numAffinity, False)
        self.cas4 = ResPoseNetwork2.create_stage(128 + numBeliefMap + numAffinity,
                                             numBeliefMap + numAffinity, False)
        self.cas5 = ResPoseNetwork2.create_stage(128 + numBeliefMap + numAffinity,
                                             numBeliefMap + numAffinity, False)
        self.cas6 = ResPoseNetwork2.create_stage(128 + numBeliefMap + numAffinity,
                                             numBeliefMap + numAffinity, False)
        # self.cas7 = ResPoseNetwork2.create_stage(128 + numBeliefMap + numAffinity,
        #                                          numBeliefMap + numAffinity, False)
        # self.cas8 = ResPoseNetwork2.create_stage(128 + numBeliefMap + numAffinity,
        #                                          numBeliefMap + numAffinity, False)

    def forward(self, x):
        '''Runs inference on the neural network'''
        numBeliefMap = self.numBeliefMap
        numAffinity = self.numAffinity

        # Same Resnet50 upto Layer-2
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)

        in1 = self.features(x)

        out1 = self.cas1(in1)
        if self.stop_at_stage == 1:
            return [out1]

        in2 = torch.cat([out1, in1], 1)
        out2 = self.cas2(in2)
        if self.stop_at_stage == 2:
            return [out1, out2]

        in3 = torch.cat([out2, in1], 1)
        out3 = self.cas3(in3)
        if self.stop_at_stage == 3:
            return [out1, out2, out3]

        in4 = torch.cat([out3, in1], 1)
        out4 = self.cas4(in4)
        if self.stop_at_stage == 4:
            return [out1, out2, out3, out4]

        in5 = torch.cat([out4, in1], 1)
        out5 = self.cas5(in5)
        if self.stop_at_stage == 5:
            return [out1, out2, out3, out4, out5]

        in6 = torch.cat([out5, in1], 1)
        out6 = self.cas6(in6)
        if self.stop_at_stage == 6:
            return [out1, out2, out3, out4, out5, out6]

        # in7 = torch.cat([out6, in1], 1)
        # out7 = self.cas6(in7)
        # if self.stop_at_stage == 7:
        #     return [out1, out2, out3, out4, out5, out6, out7]
        #
        # in8 = torch.cat([out7, in1], 1)
        # out8 = self.cas6(in8)
        # if self.stop_at_stage == 8:
        #     return [out1, out2, out3, out4, out5, out6, out7, out8]

    @staticmethod
    def create_stage(in_channels, out_channels, first=False):
        '''Create the neural network layers for a single stage.'''

        # model = nn.Sequential()
        model = []
        mid_channels = 128
        if first:
            padding = 1
            kernel = 3
            count = 6
            final_channels = 512
        else:
            padding = 3
            kernel = 7
            count = 10
            final_channels = mid_channels

        # First convolution
        model.append(nn.Conv2d(in_channels, mid_channels, kernel_size=kernel, stride=1, padding=padding))
        model.append(nn.BatchNorm2d(mid_channels))

        # Middle convolutions
        i = 1
        while i < count - 1:
            model.append(nn.ReLU(inplace=True))
            i += 1
            model.append(nn.Conv2d(mid_channels, mid_channels, kernel_size=kernel, stride=1, padding=padding))
            model.append(nn.BatchNorm2d(mid_channels))
            i += 1

        # Penultimate convolution
        model.append(nn.ReLU(inplace=True))
        i += 1
        model.append(nn.Conv2d(mid_channels, final_channels, kernel_size=1, stride=1))
        model.append(nn.BatchNorm2d(final_channels))
        i += 1

        # Last convolution
        model.append(nn.ReLU(inplace=True))
        i += 1
        model.append(nn.Conv2d(final_channels, out_channels, kernel_size=1, stride=1))
        i += 1

        return nn.Sequential(*model)

# class ResPoseNetwork2(nn.Module):
#     def __init__(self, pretrained=True, numBeliefMap=9, numAffinity=16):
#         super(ResPoseNetwork2, self).__init__()
#         self.pretrained = pretrained
#         self.numBeliefMap = numBeliefMap
#         self.numAffinity = numAffinity
#
#         resnet = torchvision.models.resnet50(pretrained=pretrained)
#         # for param in resnet.parameters():
#         #     param.requires_grad = False
#
#         self.conv1 = resnet.conv1
#         self.bn1 = resnet.bn1
#         self.relu = resnet.relu
#         self.maxpool = resnet.maxpool
#         self.layer1 = resnet.layer1
#         self.layer2 = resnet.layer2
#
#         # Defining the Cascades
#         # self.cas1 = nn.Sequential(BottleneckMod(512, 128, position='Start'), BottleneckMod(512, 128))
#         self.cas1 = nn.Sequential(Bottleneck(512, 128), Bottleneck(512, 128))
#         self.cas2 = nn.Sequential(Bottleneck(512, 128), Bottleneck(512, 128))
#         self.cas3 = nn.Sequential(Bottleneck(512, 128), Bottleneck(512, 128))
#         self.cas4 = nn.Sequential(Bottleneck(512, 128), Bottleneck(512, 128))
#
#         # Defining the final Maps Layer
#         self.maps = nn.Sequential(conv3x3(512, 256),
#                                   nn.BatchNorm2d(256),
#                                   nn.ReLU(inplace=True),
#                                   conv3x3(256, numBeliefMap + numAffinity))
#                                   # nn.BatchNorm2d(numBeliefMap + numAffinity))
#
#
#     def forward(self, x):
#         '''Runs inference on the neural network'''
#         numBeliefMap = self.numBeliefMap
#         numAffinity = self.numAffinity
#         numOutMaps = numBeliefMap + numAffinity ## TODO: in future will include * numClass
#
#         # Same Resnet50 upto Layer-2
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#         x = self.layer1(x)
#         x = self.layer2(x)
#
#         # Cascades
#         out1 = self.cas1(x)
#         out2 = self.cas2(out1)
#         out3 = self.cas3(out2)
#         out4 = self.cas4(out3)
#
#         # Maps layer
#         out5 = self.maps(out4)
#         # out5 += out4[:, :numOutMaps, :, :]
#
#         return [out5]
#
#         # return [out1[:, :numOutMaps, :, :],
#         #         out2[:, :numOutMaps, :, :],
#         #         out3[:, :numOutMaps, :, :],
#         #         out4[:, :numOutMaps, :, :],
#         #         out5]


