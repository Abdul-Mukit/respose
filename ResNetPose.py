import torch
import torch.nn as nn
import torchvision
from torchvision.models.resnet import BasicBlock, conv1x1

class ResNetPose(nn.Module):
    def __init__(self, pretrained=True, numBeliefMap=9, numAffinity=16, stop_at_stage=5):
        super(ResNetPose, self).__init__()
        self.pretrained = pretrained
        self.numBeliefMap = numBeliefMap
        self.numAffinity = numAffinity
        self.stop_at_stage = stop_at_stage

        resnet = torchvision.models.resnet34(pretrained=pretrained)
        print(f"Loading pretrained ResNet34: {pretrained}")
        print("Freezing ResNet34 accepth layer-4...")
        for param in resnet.parameters():
            param.requires_grad = False

        # Original RenNet Chain
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        for param in self.layer4.parameters():
            param.requires_grad = True

        # pre-Map1-downsampler
        self.pre_map1_downsample = resnet.layer2[0]
        for param in self.pre_map1_downsample.parameters():
            param.requires_grad = True


        # Map
        self.map1 = ResNetPose.make_map_block(in_planes=128)
        self.mapX = ResNetPose.make_map_block()  # Map 2 to 5

        # Upsample blocks
        layer3_out_ch = resnet.layer3[-1].conv2.out_channels
        layer4_out_ch = resnet.layer4[-1].conv2.out_channels
        self.up_layer3 = ResNetPose.make_upsample_block(layer3_out_ch)
        self.up_layer4a = ResNetPose.make_upsample_block(layer4_out_ch)
        self.up_layer4b = ResNetPose.make_upsample_block(layer4_out_ch // 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)

        # ---- Parallel (map_1 + layer_2) -----
        # Map-1
        in_feature = self.pre_map1_downsample(x)
        out1 = self.map1(in_feature)
        # Cont.. ResNet
        x = self.layer2(x)

        # ---- Parallel (map_2 + layer_3) -----
        # Map-2
        in_feature = torch.cat([out1, x], 1)
        out2 = self.mapX(in_feature)
        # Cont.. ResNet
        x = self.layer3(x)

        # ---- Parallel (map_3 + layer_4) -----
        # Map-3
        in_feature = self.up_layer3(x)
        in_feature = torch.cat([out2, in_feature], 1)
        out3 = self.mapX(in_feature)
        # Cont.. ResNet
        x = self.layer4(x)

        # Map-4
        x = self.up_layer4a(x)
        x = self.up_layer4b(x)  # Preserved for Map-5 instead of overwriting to in_feature
        in_feature = torch.cat([out3, x], 1)
        out4 = self.mapX(in_feature)

        # Map-5
        in_feature = torch.cat([out4, x], 1)
        out5 = self.mapX(in_feature)

        # Map-6
        in_feature = torch.cat([out5, x], 1)
        out6 = self.mapX(in_feature)

        # Map-7
        in_feature = torch.cat([out6, x], 1)
        out7 = self.mapX(in_feature)

        # Map-8
        in_feature = torch.cat([out7, x], 1)
        out8 = self.mapX(in_feature)

        # Map-9
        in_feature = torch.cat([out8, x], 1)
        out9 = self.mapX(in_feature)

        # Map-10
        in_feature = torch.cat([out9, x], 1)
        out10 = self.mapX(in_feature)

        return [out1, out2, out3, out4, out5,
                out6, out7, out8, out9, out10]

    @staticmethod
    def make_map_block(in_planes=153, out_planes=25):
        map_block = []

        map_block.append(BasicBlock(in_planes, in_planes))
        map_block.append(BasicBlock(in_planes, in_planes))
        map_block.append(BasicBlock(in_planes, in_planes))
        map_block.append(BasicBlock(in_planes, in_planes))
        map_block.append(nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1))

        return nn.Sequential(*map_block)

    @staticmethod
    def make_upsample_block(in_planes):
        up_block = []
        k, s, p = 3, 2, 1  # Kernel, Stride, Padding for exactly halved output
        out_padding = 1
        out_planes = in_planes // 2  # for halving number of input channels

        up_block.append(nn.ConvTranspose2d(in_planes, out_planes, k, s, p,
                                           out_padding, bias=False))
        up_block.append(nn.BatchNorm2d(out_planes))
        up_block.append(nn.ReLU(inplace=True))

        return nn.Sequential(*up_block)

    @staticmethod
    def make_downsample(in_planes, out_planes):
        return nn.Sequential(conv1x1(in_planes, out_planes, stride=2))
