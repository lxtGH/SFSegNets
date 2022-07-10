#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Implementation of PSPNet ResNet series.
# Author: Xiangtai(lxt@pku.edu.cn)

from torch import nn
from network import resnet_d as Resnet_Deep
from network.nn.mynn import initialize_weights, Norm2d, Upsample

from network.nn.operators import PSPModule
import time

class PSPNet(nn.Module):
    """
    Implement PSPNet model
    A: stride8
    B: stride16
    with skip connections
    """
    def __init__(self, num_classes, trunk='seresnext-50', criterion=None, variant="D"):
        super(PSPNet, self).__init__()
        self.criterion = criterion
        self.variant = variant

        if trunk == 'resnet-50-deep':
            resnet = Resnet_Deep.resnet50()
            resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        elif trunk == 'resnet-101-deep':
            resnet = Resnet_Deep.resnet101()
            resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        else:
            raise ValueError('Only support resnet50 and resnet101 for now')

        self.layer0 = resnet.layer0
        self.layer1, self.layer2, self.layer3, self.layer4 = \
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        if self.variant == 'D':
            for n, m in self.layer3.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
            for n, m in self.layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
        elif self.variant == 'D16':
            for n, m in self.layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
        else:
            print("Not using Dilation ")

        self.ppm = PSPModule(2048, 512, norm_layer=Norm2d)

        self.final = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1, bias=False))

        initialize_weights(self.ppm)
        initialize_weights(self.final)

    def forward(self, x, gts=None, cal_inference_time=False):

        start = time.time()
        x_size = x.size()  # 800
        x0 = self.layer0(x)  # 400
        x1 = self.layer1(x0)  # 400
        x2 = self.layer2(x1)  # 100
        x3 = self.layer3(x2)  # 100
        x4 = self.layer4(x3)  # 100
        xp = self.ppm(x4)
        dec1 = self.final(xp)
        main_out = Upsample(dec1, x_size[2:])

        end = time.time()
        if cal_inference_time:
            return end-start

        if self.training:
            return self.criterion(main_out, gts)

        return main_out


def PSPNet_v1_r101(num_classes, criterion):
    """
    ResNet-101 Based Network
    """
    return PSPNet(num_classes, trunk='resnet-101-deep', criterion=criterion, variant='D')

def PSPNet_v1_r50(num_classes, criterion):
    """
    ResNet-50 Based Network
    """
    return PSPNet(num_classes, trunk='resnet-50-deep', criterion=criterion, variant='D')
