

from torch import nn
import torch.nn.functional as F
from network import resnet_d as Resnet_Deep
from network.nn.operators import PSPModule, conv_bn_relu
from network.nn.mynn import Norm2d


class CascadeFeatureFusion(nn.Module):
    """CFF Unit"""
    def __init__(self, low_channels, high_channels, out_channels, nclass, norm_layer=nn.BatchNorm2d):
        super(CascadeFeatureFusion, self).__init__()
        self.conv_low = nn.Sequential(
            nn.Conv2d(low_channels, out_channels, 3, padding=2, dilation=2, bias=False),
            norm_layer(out_channels)
        )
        self.conv_high = nn.Sequential(
            nn.Conv2d(high_channels, out_channels, 1, bias=False),
            norm_layer(out_channels)
        )
        self.conv_low_cls = nn.Conv2d(out_channels, nclass, 1, bias=False)

    def forward(self, x_low, x_high):
        x_low = F.interpolate(x_low, size=x_high.size()[2:], mode='bilinear', align_corners=True)
        x_low = self.conv_low(x_low)
        x_high = self.conv_high(x_high)
        x = x_low + x_high
        x = F.relu(x, inplace=True)
        x_low_cls = self.conv_low_cls(x_low)

        return x, x_low_cls


class _ICHead(nn.Module):
    def __init__(self, nclass, norm_layer=nn.BatchNorm2d):
        super(_ICHead, self).__init__()
        self.cff_12 = CascadeFeatureFusion(128, 64, 128, nclass, norm_layer)
        self.cff_24 = CascadeFeatureFusion(256, 256, 128, nclass, norm_layer)

        self.conv_cls = nn.Conv2d(128, nclass, 1, bias=False)

    def forward(self, x_sub1, x_sub2, x_sub4):
        outputs = list()
        x_cff_24, x_24_cls = self.cff_24(x_sub4, x_sub2)
        outputs.append(x_24_cls)
        x_cff_12, x_12_cls = self.cff_12(x_cff_24, x_sub1)
        outputs.append(x_12_cls)

        up_x2 = F.interpolate(x_cff_12, scale_factor=2, mode='bilinear', align_corners=True)
        up_x2 = self.conv_cls(up_x2)
        outputs.append(up_x2)
        up_x8 = F.interpolate(up_x2, scale_factor=4, mode='bilinear', align_corners=True)
        # 1 -> 1/4 -> 1/8 -> 1/16

        return up_x8, outputs


class ICnet(nn.Module):
    """
    Implement the ICnet
    """
    def __init__(self, num_classes, trunk='resnet-50-ic', criterion=None, variant='D',
                 skip='m1', fpn_dsn=True, skip_num=48):
        super(ICnet, self).__init__()
        self.criterion = criterion
        self.variant = variant
        self.skip = skip
        self.skip_num = skip_num
        self.fpn_dsn = fpn_dsn
        assert trunk == 'resnet-50-ic'
        resnet = Resnet_Deep.resnet50()
        resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)

        self.layer0 = resnet.layer0
        self.layer1, self.layer2, self.layer3, self.layer4 = \
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        del resnet

        self.psp_head = PSPModule(2048, 512, norm_layer=Norm2d)

        self.conv_sub4 = conv_bn_relu(512, 256, 1, norm_layer=Norm2d)
        self.conv_sub2 = conv_bn_relu(512, 256, 1, norm_layer=Norm2d)

        self.conv_sub1 = nn.Sequential(
            conv_bn_relu(3, 32, 3, 2, norm_layer=Norm2d),
            conv_bn_relu(32, 32, 3, 2, norm_layer=Norm2d),
            conv_bn_relu(32, 64, 3, 2, norm_layer=Norm2d)
        )
        self.ic_head = _ICHead(num_classes, norm_layer=Norm2d)

    def forward(self, x, gts=None):
        size = x.size()[2:]
        # sub 1
        x_sub1_out = self.conv_sub1(x)

        # sub 2
        x_sub2 = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=True)

        x = self.layer0(x_sub2)

        x = self.layer1(x)
        x_sub2_out = self.layer2(x)

        # sub 4
        x_sub4 = F.interpolate(x_sub2_out, scale_factor=0.5, mode='bilinear', align_corners=True)

        x = self.layer3(x_sub4)
        x = self.layer4(x)
        x_sub4_out = self.psp_head(x)

        x_sub4_out = self.conv_sub4(x_sub4_out)
        x_sub2_out = self.conv_sub2(x_sub2_out)

        res = self.ic_head(x_sub1_out, x_sub2_out, x_sub4_out)

        if self.training:
            return self.criterion(res, gts)
        out = res[0]
        out = F.upsample(out, size=size, mode="bilinear", align_corners=True)
        return out


def ICNet_baseline(num_classes, criterion):
    """
    ResNet-50 Based Network FCN with low-level features
    """
    return ICnet(num_classes, trunk='resnet-50-ic', criterion=criterion, variant='D', skip='m1')
