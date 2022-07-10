# Author: Xiangtai Li
# Email: lixiangtai@sensetime.com
# Date: 2020/7/1
# Partial Order Pruning: for Best Speed/Accuracy Trade-off in Neural Architecture Search (CVPR2019)
from __future__ import print_function, division, absolute_import


import torch
import torch.nn as nn
import torch.nn.functional as F
from network.nn.operators import PSPModule, Aux_Module
import network.nn.mynn as mynn

model_urls = {
    'dfv1': './pretrained_models/df1_imagenet.pth',
    'dfv2': './pretrained_models/df2_imagenet.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class FusionNode(nn.Module):
    def __init__(self, inplane):
        super(FusionNode, self).__init__()
        self.fusion = conv3x3(inplane*2, inplane)

    def forward(self, x):
        x_h, x_l = x
        size = x_l.size()[2:]
        x_h = F.upsample(x_h, size, mode="bilinear", align_corners=True)
        res = self.fusion(torch.cat([x_h,x_l],dim=1))
        return res


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=nn.BatchNorm2d):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class dfnetv1(nn.Module):
    def __init__(self, num_classes=1000, norm_layer=mynn.Norm2d, stride=32):
        super(dfnetv1, self).__init__()
        self.inplanes = 64
        self.out_planes = 512
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=2, bias=False),
            norm_layer(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2, bias=False),
            norm_layer(64),
            nn.ReLU(inplace=True)
        )

        self.stage2 = self._make_layer(64, 3, stride=2, normal_layer=norm_layer)
        self.stage3 = self._make_layer(128, 3, stride=2, normal_layer=norm_layer)
        self.stage4 = self._make_layer(256, 3, stride=2, normal_layer=norm_layer)
        if stride == 32:
            self.stage5 = self._make_layer(512, 1, stride=1, normal_layer=norm_layer)
        elif stride ==64:
            self.stage5 = self._make_layer(512, 1, stride=2, normal_layer=norm_layer)
        else:
            raise ValueError("stride must be 32 or 64")

        self.avgpool = nn.AvgPool2d(7, stride=1)

    def _make_layer(self, planes, blocks, stride=1, normal_layer=nn.BatchNorm2d):
        downsample = None
        if stride != 1 or self.inplanes != planes * BasicBlock.expansion:

            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * BasicBlock.expansion,
                          kernel_size=1, stride=stride, bias=False),
                normal_layer(planes * BasicBlock.expansion),
            )

        layers = []
        layers.append(BasicBlock(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * BasicBlock.expansion
        for i in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stage1(x)  # 4x32
        x = self.stage2(x)  # 8x64
        x3 = self.stage3(x)  # 16x128
        x4 = self.stage4(x3)  # 32x256
        x5 = self.stage5(x4)  # 32x512

        return x3, x4, x5

    def get_outplanes(self):
        return self.out_planes


class dfnetv2(nn.Module):
    def __init__(self, num_classes=1000, norm_layer=mynn.Norm2d, stride=32):
        super(dfnetv2, self).__init__()
        self.inplanes = 64
        self.out_planes = 512
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=2, bias=False),
            norm_layer(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2, bias=False),
            norm_layer(64),
            nn.ReLU(inplace=True)
        )
        self.out_planes = 512
        self.stage2_1 = self._make_layer(64, 2, stride=2, norm_layer=norm_layer)
        self.stage2_2 = self._make_layer(128, 1, stride=1, norm_layer=norm_layer)
        self.stage3_1 = self._make_layer(128, 10, stride=2, norm_layer=norm_layer)
        self.stage3_2 = self._make_layer(256, 1, stride=1, norm_layer=norm_layer)
        self.stage4_1 = self._make_layer(256, 4, stride=2, norm_layer=norm_layer)
        if stride == 32:
            self.stage4_2 = self._make_layer(512, 2, stride=1, norm_layer=norm_layer)
        elif stride == 64:
            self.stage4_2 = self._make_layer(512, 2, stride=2, norm_layer=norm_layer)
        else:
            raise ValueError("stride must be 32 or 64")

        self.avgpool = nn.AvgPool2d(7, stride=1)

    def _make_layer(self, planes, blocks, stride=1,norm_layer=nn.BatchNorm2d):
        downsample = None
        if stride != 1 or self.inplanes != planes * BasicBlock.expansion:

            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * BasicBlock.expansion,
                          kernel_size=1, stride=stride, bias=False),
                norm_layer(planes * BasicBlock.expansion),
            )

        layers = []
        layers.append(BasicBlock(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * BasicBlock.expansion
        for i in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stage1(x)  # 4x32
        x = self.stage2_1(x)  # 8x64
        x3 = self.stage2_2(x)  # 8x64
        x4 = self.stage3_1(x3)  # 16x128
        x4 = self.stage3_2(x4)  # 16x128
        x5 = self.stage4_1(x4)  # 32x256
        x5 = self.stage4_2(x5)  # 32x256
        return x3, x4, x5

    def get_outplanes(self):
        return self.out_planes


class DFSegnet(nn.Module):
    def __init__(self, num_classes, trunk="dfv1", inner_planes=128, criterion=None):
        super(DFSegnet, self).__init__()
        in_planes = 512
        norm_layer = mynn.Norm2d
        if trunk == 'dfv1':
            self.backbone = DFNetv1(pretrained=True, norm_layer=norm_layer)
        elif trunk == 'dfv2':
            self.backbone = DFNetv2(pretrained=True, norm_layer=norm_layer)

        self.cc5 = nn.Conv2d(128, inner_planes, 1)
        self.cc4 = nn.Conv2d(256, inner_planes, 1)
        self.cc3 = nn.Conv2d(128, inner_planes, 1)

        self.ppm = PSPModule(in_planes, inner_planes, norm_layer=norm_layer)

        self.fn4 = FusionNode(inner_planes)
        self.fn3 = FusionNode(inner_planes)

        self.fc = Aux_Module(128, num_classes, norm_layer=norm_layer)

        self.criterion = criterion

    def forward(self, x, gts=None):
        size = x.size()[2:]
        if not self.training:
            x = F.upsample(x, size=(size[0]//2, size[1]//2),mode='bilinear',align_corners=True)
        fea = self.backbone(x)
        x3, x4, x5 = fea
        x5 = self.ppm(x5)
        x5 = self.cc5(x5)
        x4 = self.cc4(x4)
        f4 = self.fn4([x5, x4])
        x3 = self.cc3(x3)
        out = self.fn3([f4, x3])
        pred = self.fc(out)

        pred = F.upsample(pred, size=size, mode='bilinear', align_corners=True)

        if self.training:
            return self.criterion(pred, gts)
        return pred


def DFNetv1(pretrained=True, **kwargs):
    """
        Init model
    """
    model = dfnetv1(**kwargs)
    if pretrained:
        model.load_state_dict(torch.load(model_urls['dfv1'], map_location="cpu"), strict=False)
    return model


def DFNetv2(pretrained=True, **kwargs):
    """
        Init model
    """
    model = dfnetv2(**kwargs)
    if pretrained:
        model.load_state_dict(torch.load(model_urls['dfv2'], map_location="cpu"),strict=False)
    return model


def DFSegNetv1(num_classes,  criterion):
    return DFSegnet(num_classes, trunk='dfv1', criterion=criterion)


def DFSegNetv2(num_classes,  criterion):
    return DFSegnet(num_classes, trunk='dfv2', criterion=criterion)