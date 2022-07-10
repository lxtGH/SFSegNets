import torch
import torch.nn as nn
from torch.nn import functional as F
from network.nn.mynn import Norm2d
from network import resnet_d as Resnet_Deep

class ConvBNReLU(nn.Module):

    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, norm_layer=nn.BatchNorm2d):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                out_chan,
                kernel_size = ks,
                stride = stride,
                padding = padding,
                bias = False)
        self.bn = norm_layer(out_chan)
        self.relu = nn.ReLU(inplace=True)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class UpSample(nn.Module):

    def __init__(self, n_chan, factor=2):
        super(UpSample, self).__init__()
        out_chan = n_chan * factor * factor
        self.proj = nn.Conv2d(n_chan, out_chan, 1, 1, 0)
        self.up = nn.PixelShuffle(factor)
        self.init_weight()

    def forward(self, x):
        feat = self.proj(x)
        feat = self.up(feat)
        return feat

    def init_weight(self):
        nn.init.xavier_normal_(self.proj.weight, gain=1.)


class BiSeNetOutput(nn.Module):

    def __init__(self, in_chan, mid_chan, n_classes, up_factor=32, *args, **kwargs):
        super(BiSeNetOutput, self).__init__()
        self.up_factor = up_factor
        out_chan = n_classes * up_factor * up_factor
        self.conv = ConvBNReLU(in_chan, mid_chan, ks=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(mid_chan, out_chan, kernel_size=1, bias=True)
        self.up = nn.PixelShuffle(up_factor)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        x = self.up(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class AttentionRefinementModule(nn.Module):
    def __init__(self, in_chan, out_chan, norm_layer=nn.BatchNorm2d):
        super(AttentionRefinementModule, self).__init__()
        self.conv = ConvBNReLU(in_chan, out_chan, ks=3, stride=1, padding=1)
        self.conv_atten = nn.Conv2d(out_chan, out_chan, kernel_size=1, bias=False)
        self.bn_atten = norm_layer(out_chan)
        self.sigmoid_atten = nn.Sigmoid()
        self.init_weight()

    def forward(self, x):
        feat = self.conv(x)
        atten = torch.mean(feat, dim=(2, 3), keepdim=True)
        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        atten = self.sigmoid_atten(atten)
        out = torch.mul(feat, atten)
        return out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class SpatialPath(nn.Module):
    def __init__(self, *args, **kwargs):
        super(SpatialPath, self).__init__()
        self.conv1 = ConvBNReLU(3, 64, ks=7, stride=2, padding=3)
        self.conv2 = ConvBNReLU(64, 64, ks=3, stride=2, padding=1)
        self.conv3 = ConvBNReLU(64, 64, ks=3, stride=2, padding=1)
        self.conv_out = ConvBNReLU(64, 128, ks=1, stride=1, padding=0)
        self.init_weight()

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.conv2(feat)
        feat = self.conv3(feat)
        feat = self.conv_out(feat)
        return feat

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class FeatureFusionModule(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(FeatureFusionModule, self).__init__()
        self.convblk = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(out_chan,
                out_chan//4,
                kernel_size = 1,
                stride = 1,
                padding = 0,
                bias = False)
        self.conv2 = nn.Conv2d(out_chan//4,
                out_chan,
                kernel_size = 1,
                stride = 1,
                padding = 0,
                bias = False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.init_weight()

    def forward(self, fsp, fcp):
        fcat = torch.cat([fsp, fcp], dim=1)
        feat = self.convblk(fcat)
        atten = torch.mean(feat, dim=(2, 3), keepdim=True)
        atten = self.conv1(atten)
        atten = self.relu(atten)
        atten = self.conv2(atten)
        atten = self.sigmoid(atten)
        feat_atten = torch.mul(feat, atten)
        feat_out = feat_atten + feat
        return feat_out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.modules.batchnorm._BatchNorm):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class BiSegNet(nn.Module):
    def __init__(self, num_classes=19, trunk="resnet-18-deep", output_aux=True,  variant='D', fpn_dsn=True, criterion=None):
        super(BiSegNet, self).__init__()
        norm_layer = Norm2d
        self.fpn_dsn = fpn_dsn
        self.variant= variant
        # spatial path
        self.sp = SpatialPath()
        # backbone
        if trunk == 'resnet-50-deep':
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

        # contex path
        self.arm16 = AttentionRefinementModule(256, 128, norm_layer=norm_layer)
        self.arm32 = AttentionRefinementModule(512, 128, norm_layer=norm_layer)
        self.conv_head32 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1, norm_layer=norm_layer)
        self.conv_head16 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1, norm_layer=norm_layer)
        self.conv_avg = ConvBNReLU(512, 128, ks=1, stride=1, padding=0, norm_layer=norm_layer)
        self.up32 = nn.Upsample(scale_factor=2)
        self.up16 = nn.Upsample(scale_factor=2)

        self.ffm = FeatureFusionModule(256, 256)
        self.conv_out = BiSeNetOutput(256, 256, num_classes, up_factor=8)
        self.output_aux = output_aux
        if self.output_aux:
            self.conv_out16 = BiSeNetOutput(128, 64, num_classes, up_factor=8)
            self.conv_out32 = BiSeNetOutput(128, 64, num_classes, up_factor=16)
        self.criterion = criterion

    def forward(self, x, gts=None):
        # featx: feat with output stride x

        x0 = self.layer0(x)  # 400
        x1 = self.layer1(x0)  # 400
        feat8 = self.layer2(x1)  # 100
        feat16 = self.layer3(feat8)  # 100
        feat32 = self.layer4(feat16)  # 100

        size = x.size()[2:]
        avg = torch.mean(feat32, dim=(2, 3), keepdim=True)
        avg = self.conv_avg(avg)

        feat32_arm = self.arm32(feat32)
        feat32_sum = feat32_arm + avg
        feat32_up = F.upsample(feat32_sum, size=feat16.size()[2:], mode="bilinear", align_corners=True)
        feat32_up = self.conv_head32(feat32_up)

        feat16_arm = self.arm16(feat16)
        feat16_sum = feat16_arm + feat32_up
        feat16_up = self.up16(feat16_sum)
        feat16_up = self.conv_head16(feat16_up)

        ## context path output: feat16_up, feat32_up
        feat_cp8, feat_cp16 = feat16_up, feat32_up

        # sp path
        feat_sp = self.sp(x)
        feat_cp8 = F.upsample(feat_cp8, size=feat_sp.size()[2:], mode="bilinear", align_corners=True)
        feat_fuse = self.ffm(feat_sp, feat_cp8)

        feat_out = self.conv_out(feat_fuse)
        feat_out = F.upsample(feat_out, size=size, mode='bilinear', align_corners=True)
        main_out = feat_out

        if self.training:
            if not self.fpn_dsn:
                return self.criterion(main_out, gts)
            else:
                feat_out16_pred = self.conv_out16(feat_cp8)
                feat_out32_pred = self.conv_out32(feat_cp16)
                feat_out16_pred = F.upsample(feat_out16_pred, size=size, mode='bilinear', align_corners=True)
                feat_out32_pred = F.upsample(feat_out32_pred, size=size, mode='bilinear', align_corners=True)
                results = (main_out, [feat_out16_pred, feat_out32_pred])
                return self.criterion(results, gts)

        return main_out


def BiSeg_r18(num_classes, criterion):
    """
    ResNet-50 Based Network
    """
    return BiSegNet(num_classes=num_classes, trunk='resnet-18-deep', criterion=criterion, fpn_dsn=True)