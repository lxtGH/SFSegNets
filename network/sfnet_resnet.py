#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Implementation of SFNet ResNet series.
# Author: Xiangtai(lxt@pku.edu.cn)
# Date: 2022/5/20

import torch.nn as nn
import torch
from network.nn.operators import AlignedModule, PSPModule, AlignedModulev2, AlignedModulev2PoolingAtten, \
    SPSPModule

from network import resnet_d as Resnet_Deep
from network.nn.mynn import Norm2d, Upsample


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv3x3_bn_relu(in_planes, out_planes, stride=1, normal_layer=nn.BatchNorm2d):
    return nn.Sequential(
            conv3x3(in_planes, out_planes, stride),
            normal_layer(out_planes),
            nn.ReLU(inplace=True),
    )


class UperNetHead(nn.Module):

    def __init__(self, inplane, num_class, norm_layer=nn.BatchNorm2d, fpn_inplanes=[256, 512, 1024, 2048], fpn_dim=256,
                 conv3x3_type="conv", fpn_dsn=False, global_context="ppm"):
        super(UperNetHead, self).__init__()
        if global_context == "ppm":
            self.ppm = PSPModule(inplane, norm_layer=norm_layer, out_features=fpn_dim)

        self.fpn_dsn = fpn_dsn
        self.fpn_in = []
        for fpn_inplane in fpn_inplanes[:-1]:
            self.fpn_in.append(
                nn.Sequential(
                    nn.Conv2d(fpn_inplane, fpn_dim, 1),
                    norm_layer(fpn_dim),
                    nn.ReLU(inplace=False)
                )
            )
        self.fpn_in = nn.ModuleList(self.fpn_in)

        self.fpn_out = []
        self.dsn = []
        for i in range(len(fpn_inplanes) - 1):
            self.fpn_out.append(nn.Sequential(
                conv3x3_bn_relu(fpn_dim, fpn_dim, 1),
            ))

            if self.fpn_dsn:
                self.dsn.append(
                    nn.Sequential(
                        nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, stride=1, padding=1),
                        norm_layer(fpn_dim),
                        nn.ReLU(),
                        nn.Dropout2d(0.1),
                        nn.Conv2d(fpn_dim, num_class, kernel_size=1, stride=1, padding=0, bias=True)
                    )
                )

        self.fpn_out = nn.ModuleList(self.fpn_out)

        if self.fpn_dsn:
            self.dsn = nn.ModuleList(self.dsn)

        self.conv_last = nn.Sequential(
            conv3x3_bn_relu(len(fpn_inplanes) * fpn_dim, fpn_dim, 1),
            nn.Conv2d(fpn_dim, num_class, kernel_size=1)
        )

    def forward(self, conv_out):
        psp_out = self.ppm(conv_out[-1])

        f = psp_out
        fpn_feature_list = [psp_out]
        out = []
        for i in reversed(range(len(conv_out) - 1)):
            conv_x = conv_out[i]
            conv_x = self.fpn_in[i](conv_x)  # lateral branch
            size = conv_x.size()[2:]
            f = Upsample(f, size)
            # f = self.fpn_out_align[i]([conv_x, f])
            f = conv_x + f
            fpn_feature_list.append(self.fpn_out[i](f))
            if self.fpn_dsn:
                out.append(self.dsn[i](f))

        fpn_feature_list.reverse()  # [P2 - P5]
        output_size = fpn_feature_list[0].size()[2:]
        fusion_list = [fpn_feature_list[0]]

        for i in range(1, len(fpn_feature_list)):
            fusion_list.append(nn.functional.interpolate(
                fpn_feature_list[i],
                output_size,
                mode='bilinear', align_corners=True))

        fusion_out = torch.cat(fusion_list, 1)
        x = self.conv_last(fusion_out)

        return x, out


class UperNetAlignHead(nn.Module):

    def __init__(self, inplane, num_class, norm_layer=nn.BatchNorm2d, fpn_inplanes=[256, 512, 1024, 2048], fpn_dim=256,
                 conv3x3_type="conv", fpn_dsn=False, global_context="ppm"):
        super(UperNetAlignHead, self).__init__()
        if global_context == "ppm":
            self.ppm = PSPModule(inplane, norm_layer=norm_layer, out_features=fpn_dim)
        elif global_context == "sppm":
            self.ppm = SPSPModule(inplane, sizes=(1, 2, 4), norm_layer=norm_layer, out_features=fpn_dim)

        self.fpn_dsn = fpn_dsn
        self.fpn_in = []
        for fpn_inplane in fpn_inplanes[:-1]:
            self.fpn_in.append(
                nn.Sequential(
                    nn.Conv2d(fpn_inplane, fpn_dim, 1),
                    norm_layer(fpn_dim),
                    nn.ReLU(inplace=False)
                )
            )
        self.fpn_in = nn.ModuleList(self.fpn_in)

        self.fpn_out = []
        self.fpn_out_align =[]
        self.dsn = []
        for i in range(len(fpn_inplanes) - 1):
            self.fpn_out.append(nn.Sequential(
                conv3x3_bn_relu(fpn_dim, fpn_dim, 1),
            ))

            if conv3x3_type == 'conv':
                self.fpn_out_align.append(
                    AlignedModule(inplane=fpn_dim, outplane=fpn_dim//2)
                )

            if self.fpn_dsn:
                self.dsn.append(
                    nn.Sequential(
                        nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, stride=1, padding=1),
                        norm_layer(fpn_dim),
                        nn.ReLU(),
                        nn.Dropout2d(0.1),
                        nn.Conv2d(fpn_dim, num_class, kernel_size=1, stride=1, padding=0, bias=True)
                    )
                )

        self.fpn_out = nn.ModuleList(self.fpn_out)
        self.fpn_out_align = nn.ModuleList(self.fpn_out_align)

        if self.fpn_dsn:
            self.dsn = nn.ModuleList(self.dsn)

        self.conv_last = nn.Sequential(
            conv3x3_bn_relu(len(fpn_inplanes) * fpn_dim, fpn_dim, 1),
            nn.Conv2d(fpn_dim, num_class, kernel_size=1)
        )

    def forward(self, conv_out):
        psp_out = self.ppm(conv_out[-1])

        f = psp_out
        fpn_feature_list = [psp_out]
        out = []
        for i in reversed(range(len(conv_out) - 1)):
            conv_x = conv_out[i]
            conv_x = self.fpn_in[i](conv_x)  # lateral branch
            f = self.fpn_out_align[i]([conv_x, f])
            f = conv_x + f
            fpn_feature_list.append(self.fpn_out[i](f))
            if self.fpn_dsn:
                out.append(self.dsn[i](f))

        fpn_feature_list.reverse()  # [P2 - P5]
        output_size = fpn_feature_list[0].size()[2:]
        fusion_list = [fpn_feature_list[0]]

        for i in range(1, len(fpn_feature_list)):
            fusion_list.append(nn.functional.interpolate(
                fpn_feature_list[i],
                output_size,
                mode='bilinear', align_corners=True))

        fusion_out = torch.cat(fusion_list, 1)
        x = self.conv_last(fusion_out)

        return x, out


class UperNetAlignHeadV2(nn.Module):

    def __init__(self, inplane, num_class, norm_layer=nn.BatchNorm2d, fpn_inplanes=[256, 512, 1024, 2048], fpn_dim=256,
            fpn_dsn=False, fa_type="spatial", global_context="ppm"):
        super(UperNetAlignHeadV2, self).__init__()

        if global_context == "ppm":
            self.ppm = PSPModule(inplane, norm_layer=norm_layer, out_features=fpn_dim)
        elif global_context == "sppm":
            self.ppm = SPSPModule(inplane, sizes=(1, 2, 4), norm_layer=norm_layer, out_features=fpn_dim)

        self.fpn_dsn = fpn_dsn
        self.fpn_in = []
        for fpn_inplane in fpn_inplanes: # total 2 planes
            self.fpn_in.append(
                nn.Sequential(
                    nn.Conv2d(fpn_inplane, fpn_dim, 1),
                    norm_layer(fpn_dim),
                    nn.ReLU(inplace=False)
                )
            )
        self.fpn_in = nn.ModuleList(self.fpn_in)

        self.fpn_out_align =[]
        if fa_type == "spatial":
            self.flow_align_module = AlignedModulev2(inplane=fpn_dim, outplane=fpn_dim//2)
        elif fa_type == "spatial_atten":
            self.flow_align_module = AlignedModulev2PoolingAtten(inplane=fpn_dim, outplane=fpn_dim//2)

        self.fpn_out_align = nn.ModuleList(self.fpn_out_align)

        if self.fpn_dsn:
            self.dsn = nn.Sequential(
                        nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, stride=1, padding=1),
                        norm_layer(fpn_dim),
                        nn.ReLU(),
                        nn.Dropout2d(0.1),
                        nn.Conv2d(fpn_dim, num_class, kernel_size=1, stride=1, padding=0, bias=True)
                    )

        self.conv_last = nn.Sequential(
            conv3x3_bn_relu(fpn_dim*2, fpn_dim, 1),
            nn.Conv2d(fpn_dim, num_class, kernel_size=1)
        )

    def forward(self, conv_out):

        # p2, p3, p4, p5(ppm)
        psp_out = self.ppm(conv_out[-1])

        p2 = conv_out[0]
        p4 = conv_out[2]

        p2 = self.fpn_in[0](p2)
        p4 = self.fpn_in[1](p4)

        fusion_out = self.flow_align_module([p2, psp_out])
        output_size = fusion_out.size()[2:]

        p4 = nn.functional.interpolate(
               p4,
               output_size,
                mode='bilinear', align_corners=True)

        x = self.conv_last(torch.cat([fusion_out, p4], dim=1))

        return x, []



class AlignNetResNet(nn.Module):
    def __init__(self, num_classes, trunk='resnet-101', criterion=None, variant='D', head_type="v1",
                 skip='m1', skip_num=48, fpn_dsn=False, fa_type="spatial", global_context="ppm"):
        super(AlignNetResNet, self).__init__()
        self.criterion = criterion
        self.variant = variant
        self.skip = skip
        self.skip_num = skip_num
        self.fpn_dsn = fpn_dsn

        if trunk == trunk == 'resnet-50-deep':
            resnet = Resnet_Deep.resnet50()
        elif trunk == 'resnet-101-deep':
            resnet = Resnet_Deep.resnet101()
        elif trunk == 'resnet-18-deep':
            resnet = Resnet_Deep.resnet18()
        else:
            raise ValueError("Not a valid network arch")

        resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer0 = resnet.layer0
        self.layer1, self.layer2, self.layer3, self.layer4 = \
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        del resnet

        if self.variant == 'D':
            for n, m in self.layer3.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding = (2, 2), (2, 2)
                elif 'downsample.0' in n:
                    m.stride = (2, 2)
            for n, m in self.layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding = (4, 4), (4, 4)
                elif 'downsample.0' in n:
                    m.stride = (2, 2)
        else:
            print("Not using Dilation ")

        if trunk == 'resnet-18-deep':
            inplane_head = 512
            if head_type == "v2":
                self.head = UperNetAlignHeadV2(inplane_head, num_class=num_classes, norm_layer=Norm2d,
                                         fpn_inplanes=[64, 256, 512], fpn_dim=128, fpn_dsn=fpn_dsn, fa_type=fa_type,
                                               global_context = global_context
                                               )
            else:
                self.head = UperNetAlignHead(inplane_head, num_class=num_classes, norm_layer=Norm2d,
                                         fpn_inplanes=[64, 128, 256, 512], fpn_dim=128, fpn_dsn=fpn_dsn,
                                             global_context=global_context
                                             )
        else:
            inplane_head = 2048
            self.head = UperNetAlignHead(inplane_head, num_class=num_classes, norm_layer=Norm2d, fpn_dsn=fpn_dsn)

    def forward(self, x, gts=None):
        x_size = x.size()  # 800
        x0 = self.layer0(x)  # 400
        x1 = self.layer1(x0)  # 400
        x2 = self.layer2(x1)  # 100
        x3 = self.layer3(x2)  # 100
        x4 = self.layer4(x3)  # 100
        x = self.head([x1, x2, x3, x4])
        main_out = Upsample(x[0], x_size[2:])
        if self.training:
            if not self.fpn_dsn:
                return self.criterion(main_out, gts)
            return self.criterion(x, gts)
        return main_out


def DeepR101_SF_deeply(num_classes, criterion):
    """
    ResNet-50 Based Network
    """
    return AlignNetResNet(num_classes, trunk='resnet-101-deep', criterion=criterion, variant='D', skip='m1')


def DeepR50_SF_deeply(num_classes, criterion):
    """
    ResNet-50 Based Network
    """
    return AlignNetResNet(num_classes, trunk='resnet-50-deep', criterion=criterion, variant='D', skip='m1')


def DeepR18_SF_deeply(num_classes, criterion):
    """
    ResNet-18 Based Network
    """
    return AlignNetResNet(num_classes, trunk='resnet-18-deep', criterion=criterion, variant='D', skip='m1')


def DeepR18_SFV2_deeply_dsn_FA_Atten(num_classes, criterion):
    """
    ResNet-18 Based Network wtih DSN supervision
    """
    return AlignNetResNet(num_classes, trunk='resnet-18-deep', criterion=criterion, head_type="v2",
                          variant='D', skip='m1', fpn_dsn=True, fa_type="spatial_atten")


def DeepR101_SFV2_deeply_dsn_FA_Atten(num_classes, criterion):
    """
    ResNet-18 Based Network wtih DSN supervision
    """
    return AlignNetResNet(num_classes, trunk='resnet-101-deep', criterion=criterion, head_type="v2",
                          variant='D', skip='m1', fpn_dsn=True, fa_type="spatial_atten")
