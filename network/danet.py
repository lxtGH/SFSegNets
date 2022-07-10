#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Implementation of DANet ResNet series.
# Author: Xiangtai(lxt@pku.edu.cn)


from torch import nn
import torch
from network import resnet_d as Resnet_Deep
from network.nn.mynn import initialize_weights, Norm2d, Upsample

import time


class CAM_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out


class DABlock(nn.Module):
    def __init__(self, num_classes, in_planes=2048, norm_layer=nn.BatchNorm2d):
        super(DABlock, self).__init__()
        self.sa = PAM_Module(in_planes)
        self.sc = CAM_Module(in_planes)
        inner_planes = in_planes // 4
        self.conv5a = nn.Sequential(nn.Conv2d(in_planes, inner_planes, 3, padding=1, bias=False),
                                    norm_layer(inner_planes),
                                    nn.ReLU(inplace=True))

        self.conv5c = nn.Sequential(nn.Conv2d(in_planes, inner_planes, 3, padding=1, bias=False),
                                    norm_layer(inner_planes),
                                    nn.ReLU(inplace=True))

        self.sa = PAM_Module(inner_planes)
        self.sc = CAM_Module(inner_planes)
        self.conv51 = nn.Sequential(nn.Conv2d(inner_planes, inner_planes, 3, padding=1, bias=False),
                                    norm_layer(inner_planes),
                                    nn.ReLU(inplace=True))
        self.conv52 = nn.Sequential(nn.Conv2d(inner_planes, inner_planes, 3, padding=1, bias=False),
                                    norm_layer(inner_planes),
                                    nn.ReLU(inplace=True))

        self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(512, num_classes, 1))
        self.conv7 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(512, num_classes, 1))

        self.conv8 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(512, num_classes, 1))

    def forward(self, x, return_atten_map=False, hp=-1, wp=-1):
        feat1 = self.conv5a(x)
        if return_atten_map:
            return self.sa(feat1, return_atten_map, hp, wp)
        sa_feat = self.sa(feat1)
        sa_conv = self.conv51(sa_feat)
        sa_output = self.conv6(sa_conv)

        feat2 = self.conv5c(x)
        sc_feat = self.sc(feat2)
        sc_conv = self.conv52(sc_feat)
        sc_output = self.conv7(sc_conv)

        feat_sum = sa_conv + sc_conv

        sasc_output = self.conv8(feat_sum)

        output = [sasc_output]
        output.append(sa_output)
        output.append(sc_output)
        return sasc_output, sa_output, sc_output


class PAM_Module(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_dim, in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_dim, in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, return_atten_map=False, hp=-1, wp=-1):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        if return_atten_map:
            m_batchsize, C, height, width = x.size()
            position = hp * width + wp
            proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
            proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
            energy = torch.bmm(proj_query, proj_key)
            attention = self.softmax(energy)
            return attention[0, position, :].view(1, height, width)
        else:
            m_batchsize, C, height, width = x.size()
            proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
            proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
            energy = torch.bmm(proj_query, proj_key)
            attention = self.softmax(energy)
            proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

            out = torch.bmm(proj_value, attention.permute(0, 2, 1))
            out = out.view(m_batchsize, C, height, width)

            out = self.gamma*out + x
            return out


class DABlock_ours(nn.Module):
    def __init__(self, in_planes=2048, inner_planes=None, norm_layer=nn.BatchNorm2d):
        super(DABlock_ours, self).__init__()
        self.sa = PAM_Module(in_planes)
        self.sc = CAM_Module(in_planes)
        if inner_planes == None:
            inner_planes = in_planes // 4
        self.conv5a = nn.Sequential(nn.Conv2d(in_planes, inner_planes, 3, padding=1, bias=False),
                                    norm_layer(inner_planes),
                                    nn.ReLU(inplace=True))

        self.conv5c = nn.Sequential(nn.Conv2d(in_planes, inner_planes, 3, padding=1, bias=False),
                                    norm_layer(inner_planes),
                                    nn.ReLU(inplace=True))

        self.sa = PAM_Module(inner_planes)
        self.sc = CAM_Module(inner_planes)
        self.conv51 = nn.Sequential(nn.Conv2d(inner_planes, inner_planes, 3, padding=1, bias=False),
                                    norm_layer(inner_planes),
                                    nn.ReLU(inplace=True))
        self.conv52 = nn.Sequential(nn.Conv2d(inner_planes, inner_planes, 3, padding=1, bias=False),
                                    norm_layer(inner_planes),
                                    nn.ReLU(inplace=True))


    def forward(self, x, return_atten_map=False, hp=-1, wp=-1):
        feat1 = self.conv5a(x)
        if return_atten_map:
            return self.sa(feat1, return_atten_map, hp, wp)
        sa_feat = self.sa(feat1)
        sa_conv = self.conv51(sa_feat)

        feat2 = self.conv5c(x)
        sc_feat = self.sc(feat2)
        sc_conv = self.conv52(sc_feat)

        feat_sum = sa_conv + sc_conv

        return feat_sum


class DABlock_merge(nn.Module):
    def __init__(self,  in_planes=2048, out_planes=512, norm_layer=nn.BatchNorm2d):
        super(DABlock_merge, self).__init__()
        self.sa = PAM_Module(in_planes)
        self.sc = CAM_Module(in_planes)
        inner_planes = in_planes // 4
        self.conv5a = nn.Sequential(nn.Conv2d(in_planes, inner_planes, 3, padding=1, bias=False),
                                    norm_layer(inner_planes),
                                    nn.ReLU(inplace=True))

        self.conv5c = nn.Sequential(nn.Conv2d(in_planes, inner_planes, 3, padding=1, bias=False),
                                    norm_layer(inner_planes),
                                    nn.ReLU(inplace=True))

        self.sa = PAM_Module(inner_planes)
        self.sc = CAM_Module(inner_planes)
        self.conv51 = nn.Sequential(nn.Conv2d(inner_planes, inner_planes, 3, padding=1, bias=False),
                                    norm_layer(inner_planes),
                                    nn.ReLU(inplace=True))
        self.conv52 = nn.Sequential(nn.Conv2d(inner_planes, inner_planes, 3, padding=1, bias=False),
                                    norm_layer(inner_planes),
                                    nn.ReLU(inplace=True))

        self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(512, out_planes, 1))
        self.conv7 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(512, out_planes, 1))


    def forward(self, x, return_atten_map=False, hp=-1, wp=-1):
        feat1 = self.conv5a(x)
        if return_atten_map:
            return self.sa(feat1, return_atten_map, hp, wp)
        sa_feat = self.sa(feat1)
        sa_conv = self.conv51(sa_feat)
        sa_output = self.conv6(sa_conv)

        feat2 = self.conv5c(x)
        sc_feat = self.sc(feat2)
        sc_conv = self.conv52(sc_feat)
        sc_output = self.conv7(sc_conv)
        feat_sum = sa_output + sc_output

        return feat_sum


class DANet(nn.Module):
    """
    """
    def __init__(self, num_classes, trunk='seresnext-50', criterion=None, variant="D", norm_layer=nn.BatchNorm2d):
        super(DANet, self).__init__()
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

        self.da_head = DABlock_merge(2048, 512, norm_layer=norm_layer)
        self.final = nn.Conv2d(512, num_classes,kernel_size=1)
        initialize_weights(self.da_head)
        initialize_weights(self.final)

    def forward(self, x, gts=None, return_atten_map=False, hp=-1, wp=-1, cal_inference_time=False):

        start = time.time()
        x_size = x.size()  # 800
        x0 = self.layer0(x)  # 400
        x1 = self.layer1(x0)  # 400
        x2 = self.layer2(x1)  # 100
        x3 = self.layer3(x2)  # 100
        x4 = self.layer4(x3)  # 100
        if return_atten_map:
            return self.da_head(x4, return_atten_map, hp, wp)
        da_out = self.da_head(x4)
        da_out = self.final(da_out)
        main_out = Upsample(da_out, x_size[2:])

        end = time.time()
        if cal_inference_time:
            return end-start

        if self.training:
            return self.criterion(main_out, gts)

        return main_out


def DANet_r101(num_classes, criterion):
    """
    ResNet-101 Based Network
    """
    return DANet(num_classes, trunk='resnet-101-deep', criterion=criterion, variant='D',  norm_layer=Norm2d)


def DANet_r50(num_classes, criterion):
    """
    ResNet-101 Based Network
    """
    return DANet(num_classes, trunk='resnet-50-deep', criterion=criterion, variant='D', norm_layer=Norm2d)