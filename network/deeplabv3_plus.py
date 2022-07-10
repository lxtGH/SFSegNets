import torch
from torch import nn
from network import resnet_d as Resnet_Deep
from network.nn.mynn import initialize_weights, Norm2d, Upsample
from network.nn.operators import  _AtrousSpatialPyramidPoolingModule
import time


class DeepFCN(nn.Module):
    """
    Implement DeepLabV3 model
    A: stride8
    B: stride16
    with skip connections
    """

    def __init__(self, num_classes, trunk='seresnext-50', criterion=None, variant='D',
                 skip='m1', skip_num=48):
        super(DeepFCN, self).__init__()
        self.criterion = criterion
        self.variant = variant
        self.skip = skip
        self.skip_num = skip_num

        if trunk == 'resnet-50-deep':
            resnet = Resnet_Deep.resnet50()
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

        self.fcn_head = nn.Sequential(
            nn.Conv2d(2048, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True),
        )

        if self.skip == 'm1':
            self.bot_fine = nn.Conv2d(256, self.skip_num, kernel_size=1, bias=False)
        elif self.skip == 'm2':
            self.bot_fine = nn.Conv2d(512, self.skip_num, kernel_size=1, bias=False)
        else:
            raise Exception('Not a valid skip')


        self.final = nn.Sequential(
            nn.Conv2d(256 + self.skip_num, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1, bias=False))

        initialize_weights(self.fcn_head)
        initialize_weights(self.bot_fine)
        initialize_weights(self.final)

    def forward(self, x, gts=None):

        x_size = x.size()  # 800
        x0 = self.layer0(x)  # 400
        x1 = self.layer1(x0)  # 400
        x2 = self.layer2(x1)  # 100
        x3 = self.layer3(x2)  # 100
        x4 = self.layer4(x3)  # 100
        dec0_up = self.fcn_head(x4)

        if self.skip == 'm1':
            dec0_fine = self.bot_fine(x1)
            dec0_up = Upsample(dec0_up, x1.size()[2:])
        else:
            dec0_fine = self.bot_fine(x2)
            dec0_up = Upsample(dec0_up, x2.size()[2:])

        dec0 = [dec0_fine, dec0_up]
        dec0 = torch.cat(dec0, 1)
        dec1 = self.final(dec0)
        main_out = Upsample(dec1, x_size[2:])

        if self.training:
            return self.criterion(main_out, gts)

        return main_out


class DeepV3Plus(nn.Module):
    """
    Implement DeepLabV3 model
    A: stride8
    B: stride16
    with skip connections
    """

    def __init__(self, num_classes, trunk='seresnext-50', criterion=None, variant='D',
                 skip='m1', skip_num=48):
        super(DeepV3Plus, self).__init__()
        self.criterion = criterion
        self.variant = variant
        self.skip = skip
        self.skip_num = skip_num

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

        if trunk == 'resnet18':
            if self.variant == 'D':
                for n, m in self.layer3.named_modules():
                    if 'conv1' in n:
                        m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                    elif 'downsample.0' in n:
                        if trunk == 'resnext-101-32x8d' or trunk == 'resnext-50-32x4d':
                            m.kernel_size = (1, 1)
                        m.stride = (1, 1)
                for n, m in self.layer4.named_modules():
                    if 'conv1' in n:
                        m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
                    elif 'downsample.0' in n:
                        if trunk == 'resnext-101-32x8d' or trunk == 'resnext-50-32x4d':
                            m.kernel_size = (1, 1)
                        m.stride = (1, 1)
            elif self.variant == 'D16':
                for n, m in self.layer4.named_modules():
                    if 'conv1' in n:
                        m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                    elif 'downsample.0' in n:
                        if trunk == 'resnext-101-32x8d' or trunk == 'resnext-50-32x8d':
                            m.kernel_size = (1, 1)
                        m.stride = (1, 1)
            else:
                print('Not using dilation')
        else:
            if self.variant == 'D':
                for n, m in self.layer3.named_modules():
                    if 'conv2' in n:
                        m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                    elif 'downsample.0' in n:
                        if trunk == 'resnext-101-32x8d' or trunk == 'resnext-50-32x4d':
                            m.kernel_size = (1, 1)
                        m.stride = (1, 1)
                for n, m in self.layer4.named_modules():
                    if 'conv2' in n:
                        m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
                    elif 'downsample.0' in n:
                        if trunk == 'resnext-101-32x8d' or trunk == 'resnext-50-32x4d':
                            m.kernel_size = (1, 1)
                        m.stride = (1, 1)
            elif self.variant == 'D16':
                for n, m in self.layer4.named_modules():
                    if 'conv2' in n:
                        m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                    elif 'downsample.0' in n:
                        if trunk == 'resnext-101-32x8d' or trunk == 'resnext-50-32x8d':
                            m.kernel_size = (1, 1)
                        m.stride = (1, 1)
            else:
                print('Not using dilation')

        if self.variant == 'D':
            self.aspp = _AtrousSpatialPyramidPoolingModule(2048, 256, output_stride=8)
            if trunk == 'resnet18':
                self.aspp = _AtrousSpatialPyramidPoolingModule(512, 256, output_stride=8)
        elif self.variant == 'D16':
            self.aspp = _AtrousSpatialPyramidPoolingModule(2048, 256, output_stride=16)
            if trunk == 'resnet18':
                self.aspp = _AtrousSpatialPyramidPoolingModule(512, 256, output_stride=16)

        if self.skip == 'm1':
            self.bot_fine = nn.Conv2d(256, self.skip_num, kernel_size=1, bias=False)
            if trunk == 'resnet18':
                self.bot_fine = nn.Conv2d(64, self.skip_num, kernel_size=1, bias=False)
        elif self.skip == 'm2':
            self.bot_fine = nn.Conv2d(512, self.skip_num, kernel_size=1, bias=False)
            if trunk == 'resnet18':
                self.bot_fine = nn.Conv2d(128, self.skip_num, kernel_size=1, bias=False)
        else:
            raise Exception('Not a valid skip')

        self.bot_aspp = nn.Conv2d(1280, 256, kernel_size=1, bias=False)

        self.final = nn.Sequential(
            nn.Conv2d(256 + self.skip_num, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1, bias=False))

        initialize_weights(self.aspp)
        initialize_weights(self.bot_aspp)
        initialize_weights(self.bot_fine)
        initialize_weights(self.final)

    def forward(self, x, gts=None, cal_inference_time=False):

        start = time.time()
        x_size = x.size()  # 800
        x0 = self.layer0(x)  # 400
        x1 = self.layer1(x0)  # 400
        x2 = self.layer2(x1)  # 100
        x3 = self.layer3(x2)  # 100
        x4 = self.layer4(x3)  # 100
        xp = self.aspp(x4)

        dec0_up = self.bot_aspp(xp)
        if self.skip == 'm1':
            dec0_fine = self.bot_fine(x1)
            dec0_up = Upsample(dec0_up, x1.size()[2:])
        else:
            dec0_fine = self.bot_fine(x2)
            dec0_up = Upsample(dec0_up, x2.size()[2:])

        dec0 = [dec0_fine, dec0_up]
        dec0 = torch.cat(dec0, 1)
        dec1 = self.final(dec0)
        main_out = Upsample(dec1, x_size[2:])

        end = time.time()
        if cal_inference_time:
            return end-start

        if self.training:
            return self.criterion(main_out, gts)

        return main_out


def DeepR50V3PlusD_m1_deeply(num_classes, criterion):
    """
    ResNet-50 Based Network
    """
    return DeepV3Plus(num_classes, trunk='resnet-50-deep', criterion=criterion, variant='D', skip='m1')

def DeepR50V3PlusD16_m1_deeply(num_classes, criterion):
    """
    ResNet-50 Based Network with deep stem and stride=16
    """
    return DeepV3Plus(num_classes, trunk='resnet-50-deep', criterion=criterion, variant='D16', skip='m1')


def DeepR101V3PlusD_m1_deeply(num_classes, criterion):
    """
    ResNet-50 Based Network
    """
    return DeepV3Plus(num_classes, trunk='resnet-101-deep', criterion=criterion, variant='D', skip='m1')
