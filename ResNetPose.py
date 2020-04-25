import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
from torchvision.models.resnet import BasicBlock, conv3x3, conv1x1


class ResNetPose(nn.Module):
    def __init__(
            self,
            pretrained=True,
            numBeliefMap=9,
            numAffinity=16,
            stop_at_stage=6  # number of stages to process (if less than total number of stages)
    ):
        super(ResNetPose, self).__init__()

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
            # vgg_full.load_state_dict(torch.load("/home/mukit/PycharmProjects/ResPose/respose/weights/vgg19-dcbb9e9d.pth"))
            # print('Loading vgg pretrained weights from : ' + "/home/mukit/PycharmProjects/ResPose/respose/weights/vgg19-dcbb9e9d.pth")
            vgg_full = vgg_full.features
            for param in vgg_full.parameters():
                param.requires_grad = False

        self.vgg = nn.Sequential()
        for i_layer in range(27):
            self.vgg.add_module(str(i_layer), vgg_full[i_layer])

        im_ch = 128
        inter=60
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
                                      nn.Conv2d(inter, im_ch, kernel_size=3, stride=1, padding=1),
                                      nn.ReLU(inplace=True))

        # # Add some layers
        # downsampler1 = ResNetPose.make_downsample(512, 256)
        # downsampler2 = ResNetPose.make_downsample(256, 128)
        # self.features = nn.Sequential(BasicBlockMod(512, 256, downsample=downsampler1),
        #                               nn.ReLU(inplace=False),
        #                               BasicBlockMod(256, 128, downsample=downsampler2),
        #                               nn.ReLU(inplace=False))

        # print('---Belief------------------------------------------------')
        # _2 are the belief map stages
        self.cas1 = ResNetPose.create_stage(128,
                                             numBeliefMap + numAffinity, True)
        self.cas2 = ResNetPose.create_stage(128 + numBeliefMap + numAffinity,
                                             numBeliefMap + numAffinity, False)
        self.cas3 = ResNetPose.create_stage(128 + numBeliefMap + numAffinity,
                                             numBeliefMap + numAffinity, False)
        self.cas4 = ResNetPose.create_stage(128 + numBeliefMap + numAffinity,
                                             numBeliefMap + numAffinity, False)
        self.cas5 = ResNetPose.create_stage(128 + numBeliefMap + numAffinity,
                                             numBeliefMap + numAffinity, False)
        self.cas6 = ResNetPose.create_stage(128 + numBeliefMap + numAffinity,
                                             numBeliefMap + numAffinity, False)

    def forward(self, x):
        '''Runs inference on the neural network'''
        numBeliefMap = self.numBeliefMap
        numAffinity = self.numAffinity

        x = self.vgg(x)
        in1 = self.features(x)
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
            count = 1
            final_channels = 512
        else:
            padding = 1
            kernel = 3
            count = 1
            final_channels = mid_channels

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
















# class ResNetPose(nn.Module):
#     def __init__(self, pretrained=True, numBeliefMap=9, numAffinity=16, stop_at_stage=5):
#         super(ResNetPose, self).__init__()
#         self.pretrained = pretrained
#         self.numBeliefMap = numBeliefMap
#         self.numAffinity = numAffinity
#         self.stop_at_stage = stop_at_stage
#
#         resnet = torchvision.models.resnet34(pretrained=pretrained)
#         print(f"Loading pretrained ResNet34: {pretrained}")
#         print("Freezing ResNet34 accepth layer-4...")
#         for param in resnet.parameters():
#             param.requires_grad = False
#
#         # Original RenNet Chain
#         self.conv1 = resnet.conv1
#         self.bn1 = resnet.bn1
#         self.relu = resnet.relu
#         self.maxpool = resnet.maxpool
#         self.layer1 = resnet.layer1
#         self.layer2 = resnet.layer2
#         self.layer3 = resnet.layer3
#         self.layer4 = resnet.layer4
#         # for param in self.layer4.parameters():
#         #     param.requires_grad = True
#
#         # pre-Map1-downsampler
#         # self.pre_map1_downsample = resnet.layer2[0]
#         # for param in self.pre_map1_downsample.parameters():
#         #     param.requires_grad = True
#
#
#         # Maps
#         # self.map1 = ResNetPose.make_map_block(in_planes=128)
#         # self.map2 = ResNetPose.make_map_block()
#         # self.map3 = ResNetPose.make_map_block()
#         self.map4 = ResNetPose.make_map_block(in_planes=128)
#         self.map5 = ResNetPose.make_map_block()
#         self.map6 = ResNetPose.make_map_block()
#         self.map7 = ResNetPose.make_map_block()
#         self.map8 = ResNetPose.make_map_block()
#         self.map9 = ResNetPose.make_map_block()
#         self.map10 = ResNetPose.make_map_block()
#
#
#         # Upsample blocks
#         layer3_out_ch = resnet.layer3[-1].conv2.out_channels
#         layer4_out_ch = resnet.layer4[-1].conv2.out_channels
#         # self.up_layer3 = ResNetPose.make_upsample_block(layer3_out_ch)
#         self.up_layer4a = ResNetPose.make_upsample_block(layer4_out_ch, numb_BasicBlock=3)
#         self.up_layer4b = ResNetPose.make_upsample_block(layer4_out_ch // 2, numb_BasicBlock=6)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#         x = self.layer1(x)
#
#         # ---- Parallel (map_1 + layer_2) -----
#         # Map-1
#         # in_feature = self.pre_map1_downsample(x)
#         # out1 = self.map1(in_feature)
#         # Cont.. ResNet
#         x = self.layer2(x)
#
#         # ---- Parallel (map_2 + layer_3) -----
#         # Map-2
#         # in_feature = torch.cat([out1, x], 1)
#         # out2 = self.map2(in_feature)
#         # Cont.. ResNet
#         x = self.layer3(x)
#
#         # ---- Parallel (map_3 + layer_4) -----
#         # Map-3
#         # in_feature = self.up_layer3(x)
#         # in_feature = torch.cat([out2, in_feature], 1)
#         # out3 = self.map3(in_feature)
#         # Cont.. ResNet
#         x = self.layer4(x)
#
#         # Map-4
#         x = self.up_layer4a(x)
#         x = self.up_layer4b(x)  # Preserved for Map-5 instead of overwriting to in_feature
#         out4 = self.map4(x)
#
#         # Map-5
#         in_feature = torch.cat([out4, x], 1)
#         out5 = self.map5(in_feature)
#
#         # Map-6
#         in_feature = torch.cat([out5, x], 1)
#         out6 = self.map6(in_feature)
#
#         # Map-7
#         in_feature = torch.cat([out6, x], 1)
#         out7 = self.map7(in_feature)
#
#         # Map-8
#         in_feature = torch.cat([out7, x], 1)
#         out8 = self.map8(in_feature)
#
#         # Map-9
#         in_feature = torch.cat([out8, x], 1)
#         out9 = self.map9(in_feature)
#
#         # Map-10
#         in_feature = torch.cat([out9, x], 1)
#         out10 = self.map10(in_feature)
#
#         return [out5, out6, out7, out8, out9, out10]
#
#     @staticmethod
#     def make_map_block(in_planes=153, out_planes=25):
#         map_block = []
#         # map_block.append(BasicBlock(in_planes, in_planes))
#         # map_block.append(BasicBlock(in_planes, in_planes))
#         # map_block.append(BasicBlock(in_planes, in_planes))
#         # map_block.append(nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1))
#
#
#         # map_block.append(conv3x3(in_planes, in_planes))
#         # map_block.append(nn.ReLU(inplace=True))
#         # map_block.append(conv3x3(in_planes, in_planes))
#         # map_block.append(nn.ReLU(inplace=True))
#         # map_block.append(conv3x3(in_planes, in_planes))
#         # map_block.append(nn.ReLU(inplace=True))
#         # map_block.append(conv3x3(in_planes, in_planes))
#         # map_block.append(nn.ReLU(inplace=True))
#         map_block.append(conv1x1(in_planes, in_planes*2))
#         map_block.append(nn.ReLU(inplace=True))
#         map_block.append(conv3x3(in_planes*2, in_planes*2))
#         map_block.append(nn.ReLU(inplace=True))
#         map_block.append(nn.Conv2d(in_planes*2, out_planes, kernel_size=3, padding=1))
#
#
#         return nn.Sequential(*map_block)
#
#     @staticmethod
#     def make_upsample_block(in_planes, numb_BasicBlock = 4):
#         up_block = []
#         k, s, p = 2, 2, 0  # Kernel, Stride, Padding for exactly halved output
#         out_padding = 0
#         out_planes = in_planes // 2  # for halving number of input channels
#
#         up_block.append(nn.ConvTranspose2d(in_planes, out_planes, k, s, p, out_padding, bias=False))
#
#         for i in range(numb_BasicBlock):
#             up_block.append(nn.BatchNorm2d(out_planes))
#             up_block.append(nn.ReLU(inplace=True))
#
#         return nn.Sequential(*up_block)
#
#     @staticmethod
#     def make_downsample(in_planes, out_planes):
#         return nn.Sequential(conv1x1(in_planes, out_planes, stride=2))
