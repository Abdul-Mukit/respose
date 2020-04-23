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
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
from torch.autograd import Variable
import datetime
import os
import warnings
from dope_utilities import *
import sys
import re
from torch.optim.lr_scheduler import ReduceLROnPlateau

from networks import *
from networks_exp import *
from ResNetPose import *

warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
w_fname = 'weights/' # Folder to save weights in


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
                    default="Dataset/dev_batch",
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
                    default=16,
                    help='input batch size')

parser.add_argument('--batchsizetest',
                    type=int,
                    default=8,
                    help='input batch size')

parser.add_argument('--imagesize',
                    type=int,
                    default=416,
                    help='Minimum 224, and multiple of 32')

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
                    default=60,
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
                    default=None,
                    help='randomly sample that number of entries in the dataset folder')

parser.add_argument('--network',
                    default="RNP",
                    help='choose either "DOPE", "ResPose", "ResPose2","RNP" to train. name outf, namefile accordingly')

parser.add_argument('--LrSchedule',
                    default=False,
                    help='Choose whether to use a auto LrScheduler')

parser.add_argument('--featureNet',
                    default='resnet',
                    help="One of 'vgg' or 'resnet'. This is important for selecting proper Transforms")

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

if opt.pretrained in ['false', 'False']:
    opt.pretrained = False

if opt.LrSchedule in ['false', 'False']:
    opt.LrSchedule = False # TODO: PROBABLY has activation bug for LrSchedule


opt.outf = w_fname + opt.outf

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualseed is None:
    opt.manualseed = random.randint(1, 10000)


# set the manual seed.
random.seed(opt.manualseed)
torch.manual_seed(opt.manualseed)
torch.cuda.manual_seed_all(opt.manualseed)

# save
if not opt.save:
    contrast = 0.2
    brightness = 0.2
    noise = 0.1
    # normal_imgs = [0.59, 0.25]
    if opt.featureNet == 'vgg':
        normal_imgs = [[0.59, 0.59, 0.59],
                       [0.25, 0.25, 0.25]]
    elif opt.featureNet == 'resnet':
        normal_imgs = [[0.485, 0.456, 0.406],
                       [0.229, 0.224, 0.225]]
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
        img_size=opt.imagesize,
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
if opt.network=="DOPE":
    net = DopeNetwork(pretrained=opt.pretrained)
elif opt.network == "ResPose":
    net = ResPoseNetwork(pretrained=opt.pretrained)
elif opt.network == "ResPose2":
    net = ResPoseNetwork2(pretrained=opt.pretrained)
elif opt.network == "RNP":
    net = ResNetPose(pretrained=opt.pretrained)
else:
    sys.exit("Select network from 'DOPE', 'ResPose', 'ResPose2', 'RNP' to train")

net = net.to(device)
print(net)

is_new_training = opt.net == ''
prev_log_exists = os.path.exists(opt.outf + '/loss_train.csv')
print(f'New Training: {is_new_training}')
print(f'Previous log-files exist: {prev_log_exists}')

if not is_new_training and prev_log_exists:
    sys.exit('Can not continue previous training. '
             'Backup or delete loss_train.csv, loss_test.csv, header.txt files then try again')

if is_new_training:
    last_epoch = 0  # Start from Epoch 1 as this is a new training
else:
    net.load_state_dict(torch.load(opt.net))
    last_epoch = int(re.search('net_(.*)_(.*).pth', opt.net).group(2))
    print('Starting training from epoch: {}\n'.format(last_epoch + 1))

parameters = filter(lambda p: p.requires_grad, net.parameters())
optimizer = optim.Adam(parameters, lr=opt.lr)
if opt.LrSchedule:
    scheduler = ReduceLROnPlateau(optimizer, verbose=True)

if is_new_training:  # truncate exist
    # ing file then write title words
    with open(opt.outf + '/loss_train.csv', 'w') as file:
        file.write('epoch,batchid,loss\n')
    with open(opt.outf + '/loss_test.csv', 'w') as file:
        file.write('epoch,batchid,loss\n')

# save the hyper parameters passed
with open(opt.outf + '/header.txt', 'w') as file:
    file.write(str(opt) + "\n")

with open(opt.outf + '/header.txt', 'w') as file:
    file.write(str(opt))
    file.write("seed: " + str(opt.manualseed) + '\n')
    with open(opt.outf + '/test_metric.csv', 'w') as file:
        file.write("epoch, passed,total \n")

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

        if opt.network == "DOPE":
            output_belief, output_affinities = net(data)
        else:
            output_belief, output_affinities = reshape_maps(net(data))  # shape of mapList is different from DOPE's

        # TODO: Remove debug prints for ouput-map shapes
        # print(f"output_belief len: {len(output_belief)}")
        # print(f"output_affinities len: {len(output_affinities)}")
        # print(f"output_belief[0] shape: {output_belief[0].shape}")
        # print(f"outAffinityList[0] shape: {output_affinities[0].shape}")

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
            s = '{}, {},{:.15f}\n'.format(epoch, batch_idx, loss.data)
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

    return loss.data # return loss


for epoch in range(last_epoch + 1, opt.epochs + 1):
    start = datetime.datetime.now()
    if not trainingdata is None:
        cur_epoch_loss = _runnetwork(epoch, trainingdata)
        if opt.LrSchedule:
            scheduler.step(cur_epoch_loss)

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