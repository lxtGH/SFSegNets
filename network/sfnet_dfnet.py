#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Implementation of SFNet DFNet series.
# Author: Xiangtai Li(lxt@pku.edu.cn)
# Date: 2020/7/1

import torch.nn as nn

from network.dfnet import DFNetv1, DFNetv2
from network.sfnet_resnet import UperNetAlignHead
from network.nn.mynn import Norm2d, Upsample


class AlignNetDFnet(nn.Module):

    def __init__(self, num_classes, trunk='dfv1', criterion=None, variant='D',
                 skip='m1', skip_num=48, flow_conv_type="conv", fpn_dsn=False):
        super(AlignNetDFnet, self).__init__()
        self.criterion = criterion
        self.variant = variant
        self.skip = skip
        self.skip_num = skip_num
        self.fpn_dsn = fpn_dsn

        if trunk == 'dfv1':
            self.backbone = DFNetv1(pretrained=True, norm_layer=Norm2d)
        elif trunk == 'dfv2':
            self.backbone = DFNetv2(pretrained=True, norm_layer=Norm2d)
        else:
            raise ValueError("Not a valid network arch")

        self.head = UperNetAlignHead(inplane=512, num_class=num_classes, norm_layer=Norm2d,
                                     fpn_inplanes=[128, 256, 512], fpn_dim=64, conv3x3_type=flow_conv_type, fpn_dsn=fpn_dsn)

    def forward(self, x, gts=None):
        x_size = x.size()  # 800
        x2, x3, x4 = self.backbone(x)
        x = self.head([x2, x3, x4])
        main_out = Upsample(x[0], x_size[2:])
        if self.training:
            if not self.fpn_dsn:
                return self.criterion(main_out, gts)
            else:
                return self.criterion(x, gts)
        return main_out


def AlignedDFnetv1(num_classes,  criterion):
    return AlignNetDFnet(num_classes, trunk='dfv1', criterion=criterion)


def AlignedDFnetv2(num_classes,  criterion):
    return AlignNetDFnet(num_classes, trunk='dfv2', criterion=criterion)


def AlignedDFnetv1_FPNDSN(num_classes,  criterion):
    return AlignNetDFnet(num_classes, trunk='dfv1', criterion=criterion, fpn_dsn=True)


def AlignedDFnetv2_FPNDSN(num_classes,  criterion):
    return AlignNetDFnet(num_classes, trunk='dfv2', criterion=criterion, fpn_dsn=True)
