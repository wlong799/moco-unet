# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Import packages ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import numpy as np
print('numpy version: %s'%np.__version__)
import cv2 # opencv
import torch # pytorch
print('CUDA available: %d'%torch.cuda.is_available())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('torch version: %s'%torch.__version__)
from tqdm import tqdm # waitbar
import os # file path stuff
from glob import glob # listing files
import itertools
import pandas as pd
import random


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Network ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Define network structure - UNet
# Copy-paste & modify from https://github.com/milesial/Pytorch-UNet
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

# The UNet is defined modularly. 
# It is a series of downsampling layers defined by the module Down
# followed by upsampling layers defined by the module Up. The output is 
# a convolutional layer with an output channel for each landmark, defined by
# the module OutConv. 
# Each down and up layer is actually two convolutional layers with
# a ReLU nonlinearity and batch normalization, defined by the module
# DoubleConv.
# The Down module consists of a 2x2 max pool layer followed by the DoubleConv 
# module. 
# The Up module consists of an upsampling, either defined via bilinear 
# interpolation (bilinear=True), or a learned convolutional transpose, followed
# by a DoubleConv module.
# The Output layer is a single 2-D convolutional layer with no nonlinearity.
# The nonlinearity is incorporated into the network loss function. 

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

# copy-pasted and modified from unet_model.py

class UNet(nn.Module):
    def __init__(self, n_channels, n_landmarks, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_landmarks = n_landmarks
        self.bilinear = bilinear
        self.nchannels_inc = 128 #16 

        # define the layers

        # number of channels in the first layer
        nchannels_inc = self.nchannels_inc
        # increase the number of channels by a factor of 2 each layer
        nchannels_down1 = nchannels_inc*2
        nchannels_down2 = nchannels_down1*2
        nchannels_down3 = nchannels_down2*2
        # decrease the number of channels by a factor of 2 each layer
        nchannels_up1 = nchannels_down3//2
        nchannels_up2 = nchannels_up1//2
        nchannels_up3 = nchannels_up2//2

        if bilinear: 
            factor = 2
        else:
            factor = 1

        self.layer_inc = DoubleConv(n_channels, nchannels_inc)

        self.layer_down1 = Down(nchannels_inc, nchannels_down1)
        self.layer_down2 = Down(nchannels_down1, nchannels_down2)
        self.layer_down3 = Down(nchannels_down2, nchannels_down3//factor)

        self.layer_up1 = Up(nchannels_down3,nchannels_up1//factor,bilinear)
        self.layer_up2 = Up(nchannels_up1,nchannels_up2//factor,bilinear)
        self.layer_up3 = Up(nchannels_up2,nchannels_up3//factor,bilinear)

        self.layer_outc = OutConv(nchannels_up3//factor,self.n_landmarks)

    def forward(self, x, verbose=False):
        x1 = self.layer_inc(x)
        if verbose: print('inc: shape = '+str(x1.shape))
        x2 = self.layer_down1(x1)
        if verbose: print('down1: shape = '+str(x2.shape))
        x3 = self.layer_down2(x2)
        if verbose: print('down2: shape = '+str(x3.shape))
        x4 = self.layer_down3(x3)
        if verbose: print('down3: shape = '+str(x4.shape))
        x = self.layer_up1(x4, x3)
        if verbose: print('up1: shape = '+str(x.shape))        
        x = self.layer_up2(x, x2)
        if verbose: print('up2: shape = '+str(x.shape))
        x = self.layer_up3(x, x1)
        if verbose: print('up3: shape = '+str(x.shape))
        logits = self.layer_outc(x)
        if verbose: print('outc: shape = '+str(logits.shape))

        return logits

    def output(self,x,verbose=False):
        return torch.sigmoid(self.forward(x,verbose=verbose))

    def __str__(self):
        s = ''
        s += 'inc: '+str(self.layer_inc)+'\n'
        s += 'down1: '+str(self.layer_down1)+'\n'
        s += 'down2: '+str(self.layer_down2)+'\n'
        s += 'down3: '+str(self.layer_down3)+'\n'
        s += 'up1: '+str(self.layer_up1)+'\n'
        s += 'up2: '+str(self.layer_up2)+'\n'
        s += 'up3: '+str(self.layer_up3)+'\n'
        s += 'outc: '+str(self.layer_outc)+'\n'
        return s

    def __repr__(self):
        return str(self)
