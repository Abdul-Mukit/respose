import torch.nn as nn
import torchvision
from torchvision.models.resnet import Bottleneck, conv3x3, BasicBlock, conv1x1
import torch
from torchvision import models
torch.autograd.set_detect_anomaly(True)

class BottleneckMod(Bottleneck):
    '''
    * Will remove all BN
    '''
    def __init__(self, inplanes, planes):
        super(BottleneckMod, self).__init__(inplanes, planes)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.relu(out)

        out = self.conv3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BasicBlockMod(BasicBlock):
    '''
    * Will just cut out BN layers
    '''
    def __init__(self, inplanes, planes, downsample=None):
        super(BasicBlockMod, self).__init__(inplanes, planes, downsample=downsample)
        self.relu = nn.ReLU(inplace=False) # was throwing a error of not being able to compute gradient otherwise

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


# class ResPoseNetwork2(nn.Module):
#     def __init__(
#             self,
#             pretrained=True,
#             numBeliefMap=9,
#             numAffinity=16,
#             stop_at_stage=6  # number of stages to process (if less than total number of stages)
#     ):
#         super(ResPoseNetwork2, self).__init__()
#
#         self.pretrained = pretrained
#         self.numBeliefMap = numBeliefMap
#         self.numAffinity = numAffinity
#         self.stop_at_stage = stop_at_stage
#
#         if pretrained is False:
#             print("Training network without imagenet weights.")
#             vgg_full = models.vgg19(pretrained=False).features
#         else:
#             print("Training network pretrained on imagenet.")
#             vgg_full = models.vgg19(pretrained=False)
#             vgg_full.load_state_dict(torch.load("weights/vgg19-dcbb9e9d.pth"))
#             print('Loading vgg pretrained weights from : ' + "weights/vgg19-dcbb9e9d.pth")
#             vgg_full = vgg_full.features
#
#         self.vgg = nn.Sequential()
#         for i_layer in range(24):
#             self.vgg.add_module(str(i_layer), vgg_full[i_layer])
#
#         # Add some layers
#         i_layer = 23
#         self.vgg.add_module(str(i_layer), nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1))
#         self.vgg.add_module(str(i_layer + 1), nn.ReLU(inplace=True))
#         self.vgg.add_module(str(i_layer + 2), nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1))
#         self.vgg.add_module(str(i_layer + 3), nn.ReLU(inplace=True))
#
#         # print('---Belief------------------------------------------------')
#         # _2 are the belief map stages
#         self.cas1 = ResPoseNetwork2.create_stage(128,
#                                                 numBeliefMap + numAffinity, True)
#         self.cas2 = ResPoseNetwork2.create_stage(128 + numBeliefMap + numAffinity,
#                                                 numBeliefMap + numAffinity, False)
#         self.cas3 = ResPoseNetwork2.create_stage(128 + numBeliefMap + numAffinity,
#                                                 numBeliefMap + numAffinity, False)
#         self.cas4 = ResPoseNetwork2.create_stage(128 + numBeliefMap + numAffinity,
#                                                 numBeliefMap + numAffinity, False)
#         self.cas5 = ResPoseNetwork2.create_stage(128 + numBeliefMap + numAffinity,
#                                                 numBeliefMap + numAffinity, False)
#         self.cas6 = ResPoseNetwork2.create_stage(128 + numBeliefMap + numAffinity,
#                                                 numBeliefMap + numAffinity, False)
#
#     def forward(self, x):
#         '''Runs inference on the neural network'''
#         numBeliefMap = self.numBeliefMap
#         numAffinity = self.numAffinity
#
#         in1 = self.vgg(x)
#
#         out1 = self.cas1(in1)
#         if self.stop_at_stage == 1:
#             return [out1]
#
#         in2 = torch.cat([out1, in1], 1)
#         out2 = self.cas2(in2)
#         if self.stop_at_stage == 2:
#             return [out1, out2]
#
#         in3 = torch.cat([out2, in1], 1)
#         out3 = self.cas3(in3)
#         if self.stop_at_stage == 3:
#             return [out1, out2, out3]
#
#         in4 = torch.cat([out3, in1], 1)
#         out4 = self.cas4(in4)
#         if self.stop_at_stage == 4:
#             return [out1, out2, out3, out4]
#
#         in5 = torch.cat([out4, in1], 1)
#         out5 = self.cas5(in5)
#         if self.stop_at_stage == 5:
#             return [out1, out2, out3, out4, out5]
#
#         in6 = torch.cat([out5, in1], 1)
#         out6 = self.cas6(in6)
#         return [out1, out2, out3, out4, out5, out6]
#
#     @staticmethod
#     def create_stage(in_channels, out_channels, first=False):
#         '''Create the neural network layers for a single stage.'''
#
#         # model = nn.Sequential()
#         model = []
#         mid_channels = 128
#         if first:
#             padding = 1
#             kernel = 3
#             count = 6
#             final_channels = 512
#         else:
#             padding = 3
#             kernel = 7
#             count = 10
#             final_channels = mid_channels
#
#         # First convolution
#         model.append(nn.Conv2d(in_channels, mid_channels, kernel_size=kernel, stride=1, padding=padding))
#         # model.append(nn.BatchNorm2d(mid_channels))
#
#         # Middle convolutions
#         i = 1
#         while i < count - 1:
#             model.append(nn.ReLU(inplace=True))
#             i += 1
#             model.append(nn.Conv2d(mid_channels, mid_channels, kernel_size=kernel, stride=1, padding=padding))
#             # model.append(nn.BatchNorm2d(mid_channels))
#             i += 1
#
#         # Penultimate convolution
#         model.append(nn.ReLU(inplace=True))
#         i += 1
#         model.append(nn.Conv2d(mid_channels, final_channels, kernel_size=1, stride=1))
#         # model.append(nn.BatchNorm2d(final_channels))
#         i += 1
#
#         # Last convolution
#         model.append(nn.ReLU(inplace=True))
#         i += 1
#         model.append(nn.Conv2d(final_channels, out_channels, kernel_size=1, stride=1))
#         i += 1
#
#         return nn.Sequential(*model)


class ResPoseNetwork2(nn.Module):
    def __init__(
            self,
            pretrained=True,
            numBeliefMap=9,
            numAffinity=16,
            stop_at_stage=6  # number of stages to process (if less than total number of stages)

    ):
        super(ResPoseNetwork2, self).__init__()
        self.pretrained = pretrained
        self.numBeliefMap = numBeliefMap
        self.numAffinity = numAffinity
        self.stop_at_stage = stop_at_stage
        mid_ch = 256
        out_ch = 25
        cas_ch = mid_ch + out_ch

        if pretrained is False:
            print("Training network without imagenet weights.")
            vgg_full = models.vgg19(pretrained=False).features
        else:
            print("Training network pretrained on imagenet.")
            vgg_full = models.vgg19(pretrained=False)
            vgg_full.load_state_dict(torch.load("weights/vgg19-dcbb9e9d.pth"))
            print('Loading vgg pretrained weights from : ' + "weights/vgg19-dcbb9e9d.pth")
            vgg_full = vgg_full.features

        self.vgg = nn.Sequential()
        for i_layer in range(24):
            self.vgg.add_module(str(i_layer), vgg_full[i_layer])

        downsampler1 = ResPoseNetwork2.make_downsample(512, mid_ch)
        # downsampler2 = ResPoseNetwork2.make_downsample(256, 128)
        self.features = nn.Sequential(BasicBlockMod(512, mid_ch, downsample=downsampler1),
                                      BasicBlockMod(mid_ch, mid_ch))

        # Both Belief and Affnity calculation
        self.cas1 = ResPoseNetwork2.make_cascade(mid_ch, out_ch)
        self.cas2 = ResPoseNetwork2.make_cascade(cas_ch, out_ch)
        self.cas3 = ResPoseNetwork2.make_cascade(cas_ch, out_ch)
        self.cas4 = ResPoseNetwork2.make_cascade(cas_ch, out_ch)
        self.cas5 = ResPoseNetwork2.make_cascade(cas_ch, out_ch)
        self.cas6 = ResPoseNetwork2.make_cascade(cas_ch, out_ch)

    def forward(self, x):
        '''Runs inference on the neural network'''
        numBeliefMap = self.numBeliefMap
        numAffinity = self.numAffinity

        x = self.vgg(x)
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

    @staticmethod
    def make_downsample(in_planes, out_planes):
        return nn.Sequential(conv1x1(in_planes, out_planes, stride=1))

    # @staticmethod
    # def make_cascade(in_planes, out_planes):
    #     cascade = nn.Sequential(BottleneckMod(in_planes, 64),
    #                             BottleneckMod(in_planes, 64),
    #                             BottleneckMod(in_planes, 64),
    #                             nn.ReLU(inplace=True),
    #                             conv3x3(in_planes, out_planes))
    #     return cascade

    @staticmethod
    def make_cascade(in_planes, out_planes):
        cascade = nn.Sequential(BasicBlockMod(in_planes, in_planes),
                                BasicBlockMod(in_planes, in_planes),
                                BasicBlockMod(in_planes, in_planes),
                                nn.ReLU(inplace=True),
                                conv3x3(in_planes, out_planes))
        return cascade

