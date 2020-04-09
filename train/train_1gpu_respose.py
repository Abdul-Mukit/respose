# Copyright (c) 2018 NVIDIA Corporation. All rights reserved.
# This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
# https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

from __future__ import print_function

######################################################
"""
REQUIREMENTS:
matplotlib==2.2.2
simplejson==3.16.0
numpy==1.14.1
opencv_python==3.4.3.18
horovod==0.13.5
photutils==0.5
scipy==1.1.0
torch==0.4.0
pyquaternion==0.9.2
tqdm==4.25.0
pyrr==0.9.2
Pillow==5.2.0
torchvision==0.2.1
PyYAML==3.13
"""

######################################################
"""
HOW TO TRAIN DOPE
This is the DOPE training code.  
It is provided as a convenience for researchers, but it is otherwise unsupported.
Please refer to `python train.py --help` for specific details about the 
training code. 
If you download the FAT dataset 
(https://research.nvidia.com/publication/2018-06_Falling-Things)
you can train a YCB object DOPE detector as follows: 
```
python train.py --data path/to/FAT --object soup --outf soup 
--gpuids 0 1 2 3 4 5 6 7 
```
This will create a folder called `train_soup` where the weights will be saved 
after each epoch. It will use the 8 gpus using pytorch data parallel. 
"""

import argparse
import configparser as ConfigParser
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.models as models
import datetime
import os
import warnings
from dope_utilities import *

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"


##################################################
# NEURAL NETWORK MODEL
##################################################

class DopeNetwork(nn.Module):
    def __init__(
            self,
            pretrained=False,
            numBeliefMap=9,
            numAffinity=16,
            stop_at_stage=6  # number of stages to process (if less than total number of stages)
    ):
        super(DopeNetwork, self).__init__()

        self.stop_at_stage = stop_at_stage
        self.numBeliefMap = numBeliefMap
        self.numAffinity = numAffinity

        vgg_full = models.vgg19(pretrained=False).features
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
        self.cas1 = DopeNetwork.create_stage(128,
                                             numBeliefMap + numAffinity, True)
        self.cas2 = DopeNetwork.create_stage(128 + numBeliefMap + numAffinity,
                                             numBeliefMap + numAffinity, False)
        self.cas3 = DopeNetwork.create_stage(128 + numBeliefMap + numAffinity,
                                             numBeliefMap + numAffinity, False)
        self.cas4 = DopeNetwork.create_stage(128 + numBeliefMap + numAffinity,
                                             numBeliefMap + numAffinity, False)
        self.cas5 = DopeNetwork.create_stage(128 + numBeliefMap + numAffinity,
                                             numBeliefMap + numAffinity, False)
        self.cas6 = DopeNetwork.create_stage(128 + numBeliefMap + numAffinity,
                                             numBeliefMap + numAffinity, False)

        # # print('---Affinity----------------------------------------------')
        # # _1 are the affinity map stages
        # self.m1_1 = DopeNetwork.create_stage(128, numAffinity, True)
        # self.m2_1 = DopeNetwork.create_stage(128 + numBeliefMap + numAffinity,
        #                                      numAffinity, False)
        # self.m3_1 = DopeNetwork.create_stage(128 + numBeliefMap + numAffinity,
        #                                      numAffinity, False)
        # self.m4_1 = DopeNetwork.create_stage(128 + numBeliefMap + numAffinity,
        #                                      numAffinity, False)
        # self.m5_1 = DopeNetwork.create_stage(128 + numBeliefMap + numAffinity,
        #                                      numAffinity, False)
        # self.m6_1 = DopeNetwork.create_stage(128 + numBeliefMap + numAffinity,
        #                                      numAffinity, False)

    def forward(self, x):
        '''Runs inference on the neural network'''
        numBeliefMap = self.numBeliefMap
        numAffinity = self.numAffinity

        in1 = self.vgg(x)

        out1 = self.cas1(in1)
        # out1_1 = self.m1_1(in1)

        if self.stop_at_stage == 1:
            return [out1[:, :numBeliefMap, :, :]], \
                   [out1[:, :numAffinity, :, :]]

        in2 = torch.cat([out1, in1], 1)
        out2 = self.cas2(in2)
        # out2_1 = self.m2_1(in2)

        if self.stop_at_stage == 2:
            return [out1[:, :numBeliefMap, :, :], out2[:, :numBeliefMap, :, :]], \
                   [out1[:, :numAffinity, :, :], out2[:, :numAffinity, :, :]]

        in3 = torch.cat([out2, in1], 1)
        out3 = self.cas3(in3)
        # out3_1 = self.m3_1(in3)

        if self.stop_at_stage == 3:
            return [out1[:, :numBeliefMap, :, :], out2[:, :numBeliefMap, :, :], out3[:, :numBeliefMap, :, :]], \
                   [out1[:, :numAffinity, :, :], out2[:, :numAffinity, :, :], out3[:, :numAffinity, :, :]]

        in4 = torch.cat([out3, in1], 1)
        out4 = self.cas4(in4)
        # out4_1 = self.m4_1(in4)

        if self.stop_at_stage == 4:
            return [out1[:, :numBeliefMap, :, :], out2[:, :numBeliefMap, :, :], out3[:, :numBeliefMap, :, :],
                    out4[:, :numBeliefMap, :, :]], \
                   [out1[:, :numAffinity, :, :], out2[:, :numAffinity, :, :], out3[:, :numAffinity, :, :],
                    out4[:, :numAffinity, :, :]]

        in5 = torch.cat([out4, in1], 1)
        out5 = self.cas5(in5)
        # out5_1 = self.m5_1(in5)

        if self.stop_at_stage == 5:
            return [out1[:, :numBeliefMap, :, :], out2[:, :numBeliefMap, :, :], out3[:, :numBeliefMap, :, :],
                    out4[:, :numBeliefMap, :, :], out5[:, :numBeliefMap, :, :]], \
                   [out1[:, :numAffinity, :, :], out2[:, :numAffinity, :, :], out3[:, :numAffinity, :, :],
                    out4[:, :numAffinity, :, :], out5[:, :numAffinity, :, :]]

        in6 = torch.cat([out5, in1], 1)
        out6 = self.cas6(in6)
        # out6_1 = self.m6_1(in6)

        return [out1[:, :numBeliefMap, :, :], out2[:, :numBeliefMap, :, :], out3[:, :numBeliefMap, :, :],
                out4[:, :numBeliefMap, :, :], out5[:, :numBeliefMap, :, :], out6[:, :numBeliefMap, :, :]], \
               [out1[:, :numAffinity, :, :], out2[:, :numAffinity, :, :], out3[:, :numAffinity, :, :],
                out4[:, :numAffinity, :, :], out5[:, :numAffinity, :, :], out6[:, :numAffinity, :, :]]
        #TODO: return ouput as [out1, out2, out3, ...] then create a function that does this redistribution.

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



##################################################
# TRAINING CODE MAIN STARTING HERE
##################################################

print("start:", datetime.datetime.now().time())

conf_parser = argparse.ArgumentParser(
    description=__doc__,  # printed with -h/--help
    # Don't mess with format of description
    formatter_class=argparse.RawDescriptionHelpFormatter,
    # Turn off help, so we print all options in response to -h
    add_help=False
)
conf_parser.add_argument("-c", "--config",
                         help="Specify config file", metavar="FILE")

parser = argparse.ArgumentParser()

parser.add_argument('--data',
                    default="Datasets/noBlood_randColor",
                    help='path to training data')

parser.add_argument('--datatest',
                    default="",
                    help='path to data testing set')

parser.add_argument('--object',
                    default=None,
                    help='In the dataset which objet of interest')

parser.add_argument('--workers',
                    type=int,
                    default=8,
                    help='number of data loading workers')

parser.add_argument('--batchsize',
                    type=int,
                    default=32,
                    help='input batch size')

parser.add_argument('--batchsizetest',
                    type=int,
                    default=16,
                    help='input batch size')

parser.add_argument('--imagesize',
                    type=int,
                    default=400,
                    help='the height / width of the input image to network')

parser.add_argument('--lr',
                    type=float,
                    default=0.0001,
                    help='learning rate, default=0.001')

parser.add_argument('--noise',
                    type=float,
                    default=2.0,
                    help='gaussian noise added to the image')

parser.add_argument('--net',
                    default='',
                    help="path to net (to continue training)")

parser.add_argument('--namefile',
                    default='epoch',
                    help="name to put on the file of the save weights")

parser.add_argument('--manualseed',
                    type=int,
                    help='manual seed')

parser.add_argument('--epochs',
                    type=int,
                    default=2,
                    help="number of epochs to train")

parser.add_argument('--loginterval',
                    type=int,
                    default=100)

parser.add_argument('--gpuids',
                    nargs='+',
                    type=int,
                    default=[0],
                    help='GPUs to use')

parser.add_argument('--outf',
                    default='tmp',
                    help='folder to output images and model checkpoints, it will \
    add a train_ in front of the name')

parser.add_argument('--sigma',
                    default=4,
                    help='keypoint creation size for sigma')

parser.add_argument('--save',
                    action="store_true",
                    help='save a visual batch and quit, this is for\
    debugging purposes')

parser.add_argument("--pretrained",
                    default=True,
                    help='do you want to use vgg imagenet pretrained weights')

parser.add_argument('--nbupdates',
                    default=None,
                    help='nb max update to network, overwrites the epoch number\
    otherwise uses the number of epochs')

parser.add_argument('--datasize',
                    default=64,
                    help='randomly sample that number of entries in the dataset folder')

# Read the config but do not overwrite the args written
args, remaining_argv = conf_parser.parse_known_args()
defaults = {"option": "default"}

if args.config:
    config = ConfigParser.SafeConfigParser()
    config.read([args.config])
    defaults.update(dict(config.items("defaults")))

parser.set_defaults(**defaults)
parser.add_argument("--option")
opt = parser.parse_args(remaining_argv)

# TODO: Remove debug code
print("Opt is: ")
print(type(opt))
print(opt.batchsize)

if not "/" in opt.outf:
    opt.outf = "train_{}".format(opt.outf)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualseed is None:
    opt.manualseed = random.randint(1, 10000)

# save the hyper parameters passed
with open(opt.outf + '/header.txt', 'w') as file:
    file.write(str(opt) + "\n")

with open(opt.outf + '/header.txt', 'w') as file:
    file.write(str(opt))
    file.write("seed: " + str(opt.manualseed) + '\n')
    with open(opt.outf + '/test_metric.csv', 'w') as file:
        file.write("epoch, passed,total \n")

# set the manual seed.
random.seed(opt.manualseed)
torch.manual_seed(opt.manualseed)
torch.cuda.manual_seed_all(opt.manualseed)

# save
if not opt.save:
    contrast = 0.2
    brightness = 0.2
    noise = 0.1
    normal_imgs = [0.59, 0.25]
    transform = transforms.Compose([
        AddRandomContrast(contrast),
        AddRandomBrightness(brightness),
        transforms.Scale(opt.imagesize),
    ])
else:
    contrast = 0.00001
    brightness = 0.00001
    noise = 0.00001
    normal_imgs = None
    transform = transforms.Compose([
        transforms.Resize(opt.imagesize),
        transforms.ToTensor()])

print("load data")
# load the dataset using the loader in utils_pose
trainingdata = None
if not opt.data is "":
    train_dataset = MultipleVertexJson(
        root=opt.data,
        objectsofinterest=opt.object,
        keep_orientation=True,
        noise=opt.noise,
        sigma=opt.sigma,
        data_size=opt.datasize,
        save=opt.save,
        transform=transform,
        normal=normal_imgs,
        target_transform=transforms.Compose([
            transforms.Scale(opt.imagesize // 8),
        ]),
    )
    trainingdata = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=opt.batchsize,
                                               shuffle=True,
                                               num_workers=opt.workers,
                                               pin_memory=True
                                               )

if opt.save:
    for i in range(2):
        images = iter(trainingdata).next()
        if normal_imgs is None:
            normal_imgs = [0, 1]
        save_image(images['img'], '{}/train_{}.png'.format(opt.outf, str(i).zfill(5)), mean=normal_imgs[0],
                   std=normal_imgs[1])

        print(i)

    print('things are saved in {}'.format(opt.outf))
    quit()

testingdata = None
if not opt.datatest == "":
    testingdata = torch.utils.data.DataLoader(
        MultipleVertexJson(
            root=opt.datatest,
            objectsofinterest=opt.object,
            keep_orientation=True,
            noise=opt.noise,
            sigma=opt.sigma,
            data_size=opt.datasize,
            save=opt.save,
            transform=transform,
            normal=normal_imgs,
            target_transform=transforms.Compose([
                transforms.Scale(opt.imagesize // 8),
            ]),
        ),
        batch_size=opt.batchsizetest,
        shuffle=True,
        num_workers=opt.workers,
        pin_memory=True)

if not trainingdata is None:
    print('training data: {} batches'.format(len(trainingdata)))
if not testingdata is None:
    print("testing data: {} batches".format(len(testingdata)))
print('load models')


device = torch.device("cuda:" + str(opt.gpuids[0]))
print('device: ', device.index)
net = DopeNetwork(pretrained=opt.pretrained)
# net = torch.nn.DataParallel(net, device_ids=opt.gpuids) # commenting it out as onnx doesn't work with parallel
net = net.to(device)


# exit(0)

# net = torch.nn.DataParallel(net, device_ids=opt.gpuids).cuda(1)
# net = torch.nn.DataParallel(net, device_ids=opt.gpuids).cuda(1)

if opt.net != '':
    net.load_state_dict(torch.load(opt.net))

parameters = filter(lambda p: p.requires_grad, net.parameters())
optimizer = optim.Adam(parameters, lr=opt.lr)

with open(opt.outf + '/loss_train.csv', 'w') as file:
    file.write('epoch,batchid,loss\n')

with open(opt.outf + '/loss_test.csv', 'w') as file:
    file.write('epoch,batchid,loss\n')

nb_update_network = 0


def _runnetwork(epoch, loader, train=True):
    global nb_update_network
    # net
    if train:
        net.train()
    else:
        net.eval()

    for batch_idx, targets in enumerate(loader):

        data = Variable(targets['img'].cuda(opt.gpuids[0]))

        output_belief, output_affinities = net(data)

        ################ TODO: Remove debug code
        bel_np = np.array(output_belief)
        aff_np = np.array(output_affinities)
        print(bel_np.shape)
        print(aff_np.shape)
        print(net)
        ###########################################

        if train:
            optimizer.zero_grad()
        target_belief = Variable(targets['beliefs'].cuda(opt.gpuids[0]))
        target_affinity = Variable(targets['affinities'].cuda(opt.gpuids[0]))

        loss = None

        # Belief maps loss
        for l in output_belief:  # output, each belief map layers.
            if loss is None:
                loss = ((l - target_belief) * (l - target_belief)).mean()
            else:
                loss_tmp = ((l - target_belief) * (l - target_belief)).mean()
                loss += loss_tmp

        # Affinities loss
        for l in output_affinities:  # output, each belief map layers.
            loss_tmp = ((l - target_affinity) * (l - target_affinity)).mean()
            loss += loss_tmp

        if train:
            loss.backward()
            optimizer.step()
            nb_update_network += 1

        if train:
            namefile = '/loss_train.csv'
        else:
            namefile = '/loss_test.csv'

        with open(opt.outf + namefile, 'a') as file:
            s = '{}, {},{:.15f}\n'.format(
                epoch, batch_idx, loss.data)
            # print (s)
            file.write(s)

        if train:
            if batch_idx % opt.loginterval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.15f}'.format(
                    epoch, batch_idx * len(data), len(loader.dataset),
                           100. * batch_idx / len(loader), loss.data))
        else:
            if batch_idx % opt.loginterval == 0:
                print('Test Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.15f}'.format(
                    epoch, batch_idx * len(data), len(loader.dataset),
                           100. * batch_idx / len(loader), loss.data))

        # break
        if not opt.nbupdates is None and nb_update_network > int(opt.nbupdates):
            torch.save(net.state_dict(), '{}/net_{}.pth'.format(opt.outf, opt.namefile))
            break


for epoch in range(1, opt.epochs + 1):
    start = datetime.datetime.now()
    if not trainingdata is None:
        _runnetwork(epoch, trainingdata)

    if not opt.datatest == "":
        _runnetwork(epoch, testingdata, train=False)
        if opt.data == "":
            break  # lets get out of this if we are only testing
    try:
        torch.save(net.state_dict(), '{}/net_{}_{}.pth'.format(opt.outf, opt.namefile, epoch))
    except:
        pass

    if not opt.nbupdates is None and nb_update_network > int(opt.nbupdates):
        break
    end = datetime.datetime.now()
    print('Epoch Time: {}'.format(end-start))

print("end:", datetime.datetime.now().time())