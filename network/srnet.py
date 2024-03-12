# Implementation of SRNet:
#
import torch.nn as nn


from network import resnet_d as Resnet_Deep
from network.nn.mynn import initialize_weights, Norm2d, Upsample
from network.nn.operators import ModuleHead, Aux_Module, ChannelReasonModule


class SRNet(nn.Module):
    """
    Implement DeepLabV3 model
    A: stride8
    B: stride16
    with skip connections
    """

    def __init__(self, num_classes, trunk='seresnext-50', criterion=None,
                 variant='D'):
        super(SRNet, self).__init__()
        self.criterion = criterion
        self.variant = variant

        if trunk == 'resnet-50-deep':
            resnet = Resnet_Deep.resnet50()
            resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        elif trunk == 'resnet-101':
            resnet = Resnet_Deep.resnet101()
            resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        elif trunk == 'resnet-101-deep':
            resnet = Resnet_Deep.resnet101()
            resnet.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        else:
            raise ValueError("Not a valid network arch")

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

        self.head = ModuleHead(2048, 512, num_classes, module=ChannelReasonModule(512, 256))
        self.aux_layer = Aux_Module(1024, num_classes, norm_layer=Norm2d)

        initialize_weights(self.head)
        initialize_weights(self.aux_layer)

    def forward(self, x, gts=None):
        x_size = x.size()[2:]  # 800
        x0 = self.layer0(x)  # 400
        x1 = self.layer1(x0)  # 400
        x2 = self.layer2(x1)  # 100
        x3 = self.layer3(x2)  # 100
        x4 = self.layer4(x3)  # 100
        aux_out = self.aux_layer(x3)
        main_out = self.head(x4)

        aux_out = Upsample(aux_out, size=x_size)
        main_out = Upsample(main_out, size=x_size)

        if self.training:
            return self.criterion([main_out, aux_out], gts)

        return main_out


def SRNet_r50(num_classes, criterion):
    """
    ResNet-50 Based Network
    """
    return SRNet(num_classes, trunk='resnet-50-deep', criterion=criterion, variant='D')


def SRNet_r101(num_classes, criterion):
    """
    ResNet-50 Based Network
    """
    return SRNet(num_classes, trunk='resnet-101-deep', criterion=criterion, variant='D')
