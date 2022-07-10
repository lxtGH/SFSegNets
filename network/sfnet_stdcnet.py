#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Implementation of SFNet STDCNet backbone.
# Author: Xiangtai(lxt@pku.edu.cn)
# Date: 2022/4/24

import torch.nn as nn

from network.stdcnet import STDCNet813, STDCNet1446
from network.sfnet_resnet import UperNetAlignHead, UperNetAlignHeadAddition, UperNetAlignHeadV2, UperNetHead
from network.nn.mynn import Norm2d, Upsample


class AlignNetSTDCnet(nn.Module):

    def __init__(self, num_classes, trunk='stdc1', criterion=None, flow_conv_type="conv",
                 head_type="v1", fpn_dsn=False, global_context="ppm", fa_type="spatial"):
        super(AlignNetSTDCnet, self).__init__()
        self.criterion = criterion
        self.fpn_dsn = fpn_dsn

        if trunk == 'stdc1':
            self.backbone = STDCNet813(norm_layer=Norm2d)
        elif trunk == 'stdc2':
            self.backbone = STDCNet1446(norm_layer=Norm2d)
        else:
            raise ValueError("Not a valid network arch")
        if head_type == "v2":
            self.head = UperNetAlignHeadV2(1024, num_class=num_classes, norm_layer=Norm2d, fa_type=fa_type,
                                fpn_inplanes=[64, 512], fpn_dim=128, fpn_dsn=fpn_dsn, global_context=global_context)
        elif head_type == "v2_add":
            self.head = UperNetAlignHeadAddition(1024, num_class=num_classes, norm_layer=Norm2d, fa_type=fa_type,
                                fpn_inplanes=[64, 512], fpn_dim=128, fpn_dsn=fpn_dsn, global_context=global_context)
        else:
            self.head = UperNetAlignHead(inplane=1024, num_class=num_classes, norm_layer=Norm2d,
                                fpn_inplanes=[64, 256, 512, 1024], fpn_dim=64, conv3x3_type=flow_conv_type,
                                         fpn_dsn=fpn_dsn, global_context=global_context)

    def forward(self, x, gts=None):
        x_size = x.size()  # 800
        feat2, feat4, feat8, feat16, feat32 = self.backbone(x)

        x = self.head([feat4, feat8, feat16, feat32])
        main_out = Upsample(x[0], x_size[2:])
        if self.training:
            if not self.fpn_dsn:
                return self.criterion(main_out, gts)
            else:
                return self.criterion(x, gts)
        return main_out


class STDCnetFPN(nn.Module):

    def __init__(self, num_classes, trunk='stdc1', criterion=None, flow_conv_type="conv",
                 head_type="v1", fpn_dsn=False, global_context="ppm", fa_type="spatial"):
        super(STDCnetFPN, self).__init__()
        self.criterion = criterion
        self.fpn_dsn = fpn_dsn

        if trunk == 'stdc1':
            self.backbone = STDCNet813(norm_layer=Norm2d)
        elif trunk == 'stdc2':
            self.backbone = STDCNet1446(norm_layer=Norm2d)
        else:
            raise ValueError("Not a valid network arch")

        self.head = UperNetHead(inplane=1024, num_class=num_classes, norm_layer=Norm2d,
                                fpn_inplanes=[64, 256, 512, 1024], fpn_dim=64, conv3x3_type=flow_conv_type,
                                         fpn_dsn=fpn_dsn, global_context=global_context)

    def forward(self, x, gts=None):
        x_size = x.size()  # 800
        feat2, feat4, feat8, feat16, feat32 = self.backbone(x)

        x = self.head([feat4, feat8, feat16, feat32])
        main_out = Upsample(x[0], x_size[2:])
        if self.training:
            if not self.fpn_dsn:
                return self.criterion(main_out, gts)
            else:
                return self.criterion(x, gts)
        return main_out


def STDC1(num_classes,  criterion):
    return STDCnetFPN(num_classes, trunk='stdc1', criterion=criterion)


def STDC2(num_classes,  criterion):
    return STDCnetFPN(num_classes, trunk='stdc2', criterion=criterion)


def AlignedSTDC1(num_classes,  criterion):
    return AlignNetSTDCnet(num_classes, trunk='stdc1', criterion=criterion)


def AlignedSTDC2(num_classes,  criterion):
    return AlignNetSTDCnet(num_classes, trunk='stdc2', criterion=criterion)


def AlignedSTDC2_SFV2(num_classes, criterion):
    """
    ResNet-18 Based Network wtih DSN supervision
    """
    return AlignNetSTDCnet(num_classes, trunk='stdc2', criterion=criterion, head_type="v2",
                        fpn_dsn=True)


def AlignedSTDC1_SFV2(num_classes, criterion):
    """
    ResNet-18 Based Network wtih DSN supervision
    """
    return AlignNetSTDCnet(num_classes, trunk='stdc1', criterion=criterion, head_type="v2",
                           fpn_dsn=True)


def AlignedSTDC2_SFV2add(num_classes, criterion):
    """
    ResNet-18 Based Network wtih DSN supervision
    """
    return AlignNetSTDCnet(num_classes, trunk='stdc2', criterion=criterion, head_type="v2_add",
                        fpn_dsn=True)


def AlignedSTDC1_SFV2add(num_classes, criterion):
    """
    ResNet-18 Based Network wtih DSN supervision
    """
    return AlignNetSTDCnet(num_classes, trunk='stdc1', criterion=criterion, head_type="v2",
                           fpn_dsn=True)


def AlignedSTDC1_SFV2_spatial_atten(num_classes, criterion):
    """
    ResNet-18 Based Network wtih DSN supervision
    """
    return AlignNetSTDCnet(num_classes, trunk='stdc1', criterion=criterion, head_type="v2",
                           fpn_dsn=True, fa_type="spatial_atten")

def AlignedSTDC2_SFV2_spatial_atten(num_classes, criterion):
    """
    ResNet-18 Based Network wtih DSN supervision
    """
    return AlignNetSTDCnet(num_classes, trunk='stdc2', criterion=criterion, head_type="v2",
                           fpn_dsn=True, fa_type="spatial_atten")


