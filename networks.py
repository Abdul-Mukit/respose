##################################################
# NEURAL NETWORK MODEL
##################################################
from torch import nn
import torchvision.models as models
import torch
from torchvision.models.resnet import BasicBlock, conv3x3, conv1x1


class DopeNetwork(nn.Module):
    def __init__(
            self,
            pretrained=False,
            numBeliefMap=9,
            numAffinity=16,
            stop_at_stage=6  # number of stages to process (if less than total number of stages)
    ):
        super(DopeNetwork, self).__init__()

        self.pretrained = pretrained
        self.numBeliefMap = numBeliefMap
        self.numAffinity = numAffinity
        self.stop_at_stage = stop_at_stage

        if pretrained is False:
            # print("Training network without imagenet weights.")
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

        # Add some layers
        i_layer = 23
        self.vgg.add_module(str(i_layer), nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1))
        self.vgg.add_module(str(i_layer + 1), nn.ReLU(inplace=True))
        self.vgg.add_module(str(i_layer + 2), nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1))
        self.vgg.add_module(str(i_layer + 3), nn.ReLU(inplace=True))

        # print('---Belief------------------------------------------------')
        # _2 are the belief map stages
        self.m1_2 = DopeNetwork.create_stage(128, numBeliefMap, True)
        self.m2_2 = DopeNetwork.create_stage(128 + numBeliefMap + numAffinity,
                                             numBeliefMap, False)
        self.m3_2 = DopeNetwork.create_stage(128 + numBeliefMap + numAffinity,
                                             numBeliefMap, False)
        self.m4_2 = DopeNetwork.create_stage(128 + numBeliefMap + numAffinity,
                                             numBeliefMap, False)
        self.m5_2 = DopeNetwork.create_stage(128 + numBeliefMap + numAffinity,
                                             numBeliefMap, False)
        self.m6_2 = DopeNetwork.create_stage(128 + numBeliefMap + numAffinity,
                                             numBeliefMap, False)

        # print('---Affinity----------------------------------------------')
        # _1 are the affinity map stages
        self.m1_1 = DopeNetwork.create_stage(128, numAffinity, True)
        self.m2_1 = DopeNetwork.create_stage(128 + numBeliefMap + numAffinity,
                                             numAffinity, False)
        self.m3_1 = DopeNetwork.create_stage(128 + numBeliefMap + numAffinity,
                                             numAffinity, False)
        self.m4_1 = DopeNetwork.create_stage(128 + numBeliefMap + numAffinity,
                                             numAffinity, False)
        self.m5_1 = DopeNetwork.create_stage(128 + numBeliefMap + numAffinity,
                                             numAffinity, False)
        self.m6_1 = DopeNetwork.create_stage(128 + numBeliefMap + numAffinity,
                                             numAffinity, False)

    def forward(self, x):
        '''Runs inference on the neural network'''

        out1 = self.vgg(x)

        out1_2 = self.m1_2(out1)
        out1_1 = self.m1_1(out1)

        if self.stop_at_stage == 1:
            return [out1_2], \
                   [out1_1]

        out2 = torch.cat([out1_2, out1_1, out1], 1)
        out2_2 = self.m2_2(out2)
        out2_1 = self.m2_1(out2)

        if self.stop_at_stage == 2:
            return [out1_2, out2_2], \
                   [out1_1, out2_1]

        out3 = torch.cat([out2_2, out2_1, out1], 1)
        out3_2 = self.m3_2(out3)
        out3_1 = self.m3_1(out3)

        if self.stop_at_stage == 3:
            return [out1_2, out2_2, out3_2], \
                   [out1_1, out2_1, out3_1]

        out4 = torch.cat([out3_2, out3_1, out1], 1)
        out4_2 = self.m4_2(out4)
        out4_1 = self.m4_1(out4)

        if self.stop_at_stage == 4:
            return [out1_2, out2_2, out3_2, out4_2], \
                   [out1_1, out2_1, out3_1, out4_1]

        out5 = torch.cat([out4_2, out4_1, out1], 1)
        out5_2 = self.m5_2(out5)
        out5_1 = self.m5_1(out5)

        if self.stop_at_stage == 5:
            return [out1_2, out2_2, out3_2, out4_2, out5_2], \
                   [out1_1, out2_1, out3_1, out4_1, out5_1]

        out6 = torch.cat([out5_2, out5_1, out1], 1)
        out6_2 = self.m6_2(out6)
        out6_1 = self.m6_1(out6)

        return [out1_2, out2_2, out3_2, out4_2, out5_2, out6_2], \
               [out1_1, out2_1, out3_1, out4_1, out5_1, out6_1]

    @staticmethod
    def create_stage(in_channels, out_channels, first=False):
        '''Create the neural network layers for a single stage.'''

        model = nn.Sequential()
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
        model.add_module("0",
                         nn.Conv2d(
                             in_channels,
                             mid_channels,
                             kernel_size=kernel,
                             stride=1,
                             padding=padding)
                         )

        # Middle convolutions
        i = 1
        while i < count - 1:
            model.add_module(str(i), nn.ReLU(inplace=True))
            i += 1
            model.add_module(str(i),
                             nn.Conv2d(
                                 mid_channels,
                                 mid_channels,
                                 kernel_size=kernel,
                                 stride=1,
                                 padding=padding))
            i += 1

        # Penultimate convolution
        model.add_module(str(i), nn.ReLU(inplace=True))
        i += 1
        model.add_module(str(i), nn.Conv2d(mid_channels, final_channels, kernel_size=1, stride=1))
        i += 1

        # Last convolution
        model.add_module(str(i), nn.ReLU(inplace=True))
        i += 1
        model.add_module(str(i), nn.Conv2d(final_channels, out_channels, kernel_size=1, stride=1))
        i += 1

        return model


# ResPoseNetwork originally
class DOPE_2(nn.Module):
    def __init__(
            self,
            pretrained=False,
            numBeliefMap=9,
            numAffinity=16,
            stop_at_stage=6  # number of stages to process (if less than total number of stages)
    ):
        super(DOPE_2, self).__init__()

        self.pretrained = pretrained
        self.numBeliefMap = numBeliefMap
        self.numAffinity = numAffinity
        self.stop_at_stage = stop_at_stage

        if pretrained is False:
            # print("Training network without imagenet weights.")
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

        # Add some layers
        i_layer = 23
        self.vgg.add_module(str(i_layer), nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1))
        self.vgg.add_module(str(i_layer + 1), nn.ReLU(inplace=True))
        self.vgg.add_module(str(i_layer + 2), nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1))
        self.vgg.add_module(str(i_layer + 3), nn.ReLU(inplace=True))

        # print('---Belief------------------------------------------------')
        # _2 are the belief map stages
        self.cas1 = DOPE_2.create_stage(128,
                                             numBeliefMap + numAffinity, True)
        self.cas2 = DOPE_2.create_stage(128 + numBeliefMap + numAffinity,
                                             numBeliefMap + numAffinity, False)
        self.cas3 = DOPE_2.create_stage(128 + numBeliefMap + numAffinity,
                                             numBeliefMap + numAffinity, False)
        self.cas4 = DOPE_2.create_stage(128 + numBeliefMap + numAffinity,
                                             numBeliefMap + numAffinity, False)
        self.cas5 = DOPE_2.create_stage(128 + numBeliefMap + numAffinity,
                                             numBeliefMap + numAffinity, False)
        self.cas6 = DOPE_2.create_stage(128 + numBeliefMap + numAffinity,
                                             numBeliefMap + numAffinity, False)

    def forward(self, x):
        '''Runs inference on the neural network'''
        numBeliefMap = self.numBeliefMap
        numAffinity = self.numAffinity

        in1 = self.vgg(x)

        out1 = self.cas1(in1)
        # out1_1 = self.m1_1(in1)

        if self.stop_at_stage == 1:
            return [out1]

        in2 = torch.cat([out1, in1], 1)
        out2 = self.cas2(in2)
        # out2_1 = self.m2_1(in2)

        if self.stop_at_stage == 2:
            return [out1, out2]

        in3 = torch.cat([out2, in1], 1)
        out3 = self.cas3(in3)
        # out3_1 = self.m3_1(in3)

        if self.stop_at_stage == 3:
            return [out1, out2, out3]

        in4 = torch.cat([out3, in1], 1)
        out4 = self.cas4(in4)
        # out4_1 = self.m4_1(in4)

        if self.stop_at_stage == 4:
            return [out1, out2, out3, out4]

        in5 = torch.cat([out4, in1], 1)
        out5 = self.cas5(in5)
        # out5_1 = self.m5_1(in5)

        if self.stop_at_stage == 5:
            return [out1, out2, out3, out4, out5]

        in6 = torch.cat([out5, in1], 1)
        out6 = self.cas6(in6)
        # out6_1 = self.m6_1(in6)

        return [out1, out2, out3, out4, out5, out6]

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

        # Middle convolutions
        i = 1
        while i < count - 1:
            model.append(nn.ReLU(inplace=True))
            i += 1
            model.append(nn.Conv2d(mid_channels, mid_channels, kernel_size=kernel, stride=1, padding=padding))
            i += 1

        # Penultimate convolution
        model.append(nn.ReLU(inplace=True))
        i += 1
        model.append(nn.Conv2d(mid_channels, final_channels, kernel_size=1, stride=1))
        i += 1

        # Last convolution
        model.append(nn.ReLU(inplace=True))
        i += 1
        model.append(nn.Conv2d(final_channels, out_channels, kernel_size=1, stride=1))
        i += 1

        return nn.Sequential(*model)

class DOPE_2p1(nn.Module):
    def __init__(
            self,
            pretrained=True,
            numBeliefMap=9,
            numAffinity=16,
            stop_at_stage=6  # number of stages to process (if less than total number of stages)
    ):
        super(DOPE_2p1, self).__init__()

        self.pretrained = pretrained
        self.numBeliefMap = numBeliefMap
        self.numAffinity = numAffinity
        self.stop_at_stage = stop_at_stage

        if pretrained is False:
            print("Training network without imagenet weights.")
            vgg_full = models.vgg19(pretrained=False).features
        else:
            print("Training network pretrained on imagenet.")
            vgg_full = models.vgg19(pretrained=True)
            vgg_full = vgg_full.features
            # for param in vgg_full.parameters():
            #     param.requires_grad = False

        self.vgg = nn.Sequential()
        for i_layer in range(27):
            self.vgg.add_module(str(i_layer), vgg_full[i_layer])

        im_ch = 128
        inter = 128
        self.features = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(256, im_ch, kernel_size=3, stride=1, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(im_ch, inter, kernel_size=3, stride=1, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(inter, inter, kernel_size=3, stride=1, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(inter, inter, kernel_size=3, stride=1, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(inter, inter, kernel_size=3, stride=1, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(inter, inter, kernel_size=3, stride=1, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(inter, inter, kernel_size=3, stride=1, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(inter, inter, kernel_size=3, stride=1, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(inter, im_ch, kernel_size=3, stride=1, padding=1),
                                      nn.ReLU(inplace=True))

        # Map generation
        self.cas1 = DOPE_2p1.create_stage(128,
                                             numBeliefMap + numAffinity, True)
        self.cas2 = DOPE_2p1.create_stage(128 + numBeliefMap + numAffinity,
                                             numBeliefMap + numAffinity, False)
        self.cas3 = DOPE_2p1.create_stage(128 + numBeliefMap + numAffinity,
                                             numBeliefMap + numAffinity, False)
        self.cas4 = DOPE_2p1.create_stage(128 + numBeliefMap + numAffinity,
                                             numBeliefMap + numAffinity, False)
        self.cas5 = DOPE_2p1.create_stage(128 + numBeliefMap + numAffinity,
                                             numBeliefMap + numAffinity, False)
        self.cas6 = DOPE_2p1.create_stage(128 + numBeliefMap + numAffinity,
                                             numBeliefMap + numAffinity, False)

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

        return [out1, out2, out3, out4, out5, out6]

    @staticmethod
    def create_stage(in_channels, out_channels, first=False):
        '''Create the neural network layers for a single stage.'''

        # model = nn.Sequential()
        model = []
        mid_channels = 128
        if first:
            padding = 1
            kernel = 3
            count = 1
            final_channels = 512
        else:
            padding = 1
            kernel = 3
            count = 1
            final_channels = 512

        # First convolution
        model.append(nn.Conv2d(in_channels, mid_channels, kernel_size=kernel, stride=1, padding=padding))
        model.append(nn.ReLU(inplace=True))

        # Middle convolutions
        for i in range(count):
            model.append(nn.Conv2d(mid_channels, mid_channels, kernel_size=kernel, stride=1, padding=padding))
            model.append(nn.ReLU(inplace=True))

        # Penultimate convolution
        model.append(nn.Conv2d(mid_channels, final_channels, kernel_size=1, stride=1))
        model.append(nn.ReLU(inplace=True))

        # Last convolution
        model.append(nn.Conv2d(final_channels, out_channels, kernel_size=1, stride=1))

        return nn.Sequential(*model)

class ResNetPose(nn.Module):
    def __init__(self, pretrained=True, numBeliefMap=9, numAffinity=16, stop_at_stage=5):
        super(ResNetPose, self).__init__()
        self.pretrained = pretrained
        self.numBeliefMap = numBeliefMap
        self.numAffinity = numAffinity
        self.stop_at_stage = stop_at_stage
        total_map_count = numBeliefMap + numAffinity

        resnet = models.resnet34(pretrained=pretrained)
        print(f"Loading pretrained ResNet34: {pretrained}")
        print("Freezing entire ResNet34 except layer 3 and 4...")
        for param in resnet.parameters():
            param.requires_grad = False
        for param in resnet.layer3.parameters():
            param.requires_grad = True
        for param in resnet.layer4.parameters():
            param.requires_grad = True

        # Original RenNet Chain
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4


        # Upsample/Downsample blocks
        layer3_out_ch = resnet.layer3[-1].conv2.out_channels
        layer4_out_ch = resnet.layer4[-1].conv2.out_channels
        self.up_layer3 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.up_layer4 = nn.Upsample(scale_factor=4, mode='bilinear')
        self.down_layer3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Map
        self.map1 = ResNetPose.make_map_block(layer3_out_ch)
        self.map2 = ResNetPose.make_map_block(layer3_out_ch + total_map_count)
        self.map3 = ResNetPose.make_map_block(layer3_out_ch + total_map_count)
        self.map4 = ResNetPose.make_map_block(layer4_out_ch + total_map_count)
        self.map5 = ResNetPose.make_map_block(layer4_out_ch + total_map_count)
        self.map6 = ResNetPose.make_map_block(layer4_out_ch + total_map_count)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # Map 1, Map2, Map 3
        out1 = self.map1(x)
        out2 = self.map2(torch.cat([out1, x], 1))
        out3 = self.map3(torch.cat([out2, x], 1))
        # Resnet Layer 4
        x = self.layer4(x)
        # Map 4, 5, 6
        in_map4 = self.down_layer3(out3)  # Downsample by 2
        out4 = self.map4(torch.cat([in_map4, x], 1))
        out5 = self.map5(torch.cat([out4, x], 1))
        out6 = self.map6(torch.cat([out5, x], 1))

        # Upsample output maps for final output
        out1 = self.up_layer3(out1)
        out2 = self.up_layer3(out2)
        out3 = self.up_layer3(out3)
        out4 = self.up_layer4(out4)
        out5 = self.up_layer4(out5)
        out6 = self.up_layer4(out6)

        return [out1, out2, out3, out4, out5, out6]

    @staticmethod
    def make_map_block(in_planes=153, out_planes=25):
        map_block = []
        map_block.append(BasicBlock(in_planes, in_planes))
        map_block.append(BasicBlock(in_planes, in_planes))
        map_block.append(BasicBlock(in_planes, in_planes))
        map_block.append(nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1))
        return nn.Sequential(*map_block)

    # @staticmethod
    # def make_map_block(in_planes=153, out_planes=25):
    #     map_block = []
    #     map_block.append(nn.Conv2d(in_planes, in_planes, kernel_size=3, padding=1))
    #     map_block.append(nn.ReLU(inplace=True))
    #     map_block.append(nn.Conv2d(in_planes, in_planes, kernel_size=3, padding=1))
    #     map_block.append(nn.ReLU(inplace=True))
    #     map_block.append(nn.Conv2d(in_planes, in_planes, kernel_size=3, padding=1))
    #     map_block.append(nn.ReLU(inplace=True))
    #     map_block.append(nn.Conv2d(in_planes, in_planes, kernel_size=3, padding=1))
    #     map_block.append(nn.ReLU(inplace=True))
    #     map_block.append(nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1))
    #     return nn.Sequential(*map_block)

class DOPE_2p2(nn.Module):
    def __init__(
            self,
            pretrained=True,
            numBeliefMap=9,
            numAffinity=16,
            stop_at_stage=6  # number of stages to process (if less than total number of stages)
    ):
        super(DOPE_2p1, self).__init__()

        self.pretrained = pretrained
        self.numBeliefMap = numBeliefMap
        self.numAffinity = numAffinity
        self.stop_at_stage = stop_at_stage

        if pretrained is False:
            print("Training network without imagenet weights.")
            vgg_full = models.vgg19(pretrained=False).features
        else:
            print("Training network pretrained on imagenet.")
            vgg_full = models.vgg19(pretrained=True)
            vgg_full = vgg_full.features
            # for param in vgg_full.parameters():
            #     param.requires_grad = False

        self.vgg = nn.Sequential()
        for i_layer in range(27):
            self.vgg.add_module(str(i_layer), vgg_full[i_layer])

        im_ch = 128
        inter = 128
        self.features = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(256, im_ch, kernel_size=3, stride=1, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(im_ch, inter, kernel_size=3, stride=1, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(inter, inter, kernel_size=3, stride=1, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(inter, inter, kernel_size=3, stride=1, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(inter, inter, kernel_size=3, stride=1, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(inter, inter, kernel_size=3, stride=1, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(inter, inter, kernel_size=3, stride=1, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(inter, inter, kernel_size=3, stride=1, padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(inter, im_ch, kernel_size=3, stride=1, padding=1),
                                      nn.ReLU(inplace=True))

        # Map generation
        self.cas1 = DOPE_2p2.create_stage(128,
                                             numBeliefMap + numAffinity, True)
        self.cas2 = DOPE_2p2.create_stage(128 + numBeliefMap + numAffinity,
                                             numBeliefMap + numAffinity, False)
        self.cas3 = DOPE_2p2.create_stage(128 + numBeliefMap + numAffinity,
                                             numBeliefMap + numAffinity, False)
        self.cas4 = DOPE_2p2.create_stage(128 + numBeliefMap + numAffinity,
                                             numBeliefMap + numAffinity, False)
        self.cas5 = DOPE_2p2.create_stage(128 + numBeliefMap + numAffinity,
                                             numBeliefMap + numAffinity, False)
        self.cas6 = DOPE_2p2.create_stage(128 + numBeliefMap + numAffinity,
                                             numBeliefMap + numAffinity, False)

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

        return [out1, out2, out3, out4, out5, out6]

    @staticmethod
    def create_stage(in_channels, out_channels, first=False):
        '''Create the neural network layers for a single stage.'''

        # model = nn.Sequential()
        model = []
        mid_channels = 128
        if first:
            padding = 1
            kernel = 3
            count = 1
            final_channels = 512
        else:
            padding = 1
            kernel = 3
            count = 1
            final_channels = 512

        # First convolution
        model.append(nn.Conv2d(in_channels, mid_channels, kernel_size=kernel, stride=1, padding=padding))
        model.append(nn.ReLU(inplace=True))

        # Middle convolutions
        for i in range(count):
            model.append(nn.Conv2d(mid_channels, mid_channels, kernel_size=kernel, stride=1, padding=padding))
            model.append(nn.ReLU(inplace=True))

        # Penultimate convolution
        model.append(nn.Conv2d(mid_channels, final_channels, kernel_size=1, stride=1))
        model.append(nn.ReLU(inplace=True))

        # Last convolution
        model.append(nn.Conv2d(final_channels, out_channels, kernel_size=1, stride=1))

        return nn.Sequential(*model)

