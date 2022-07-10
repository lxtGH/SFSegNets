
from torch.nn import functional as F
import torch
from torch import nn
from network.nn.mynn import Norm2d
from network.nn.operators import conv_sigmoid, BiSeNetOutput, BiSeNetOutput2
from network.nn.operators import PSPModule
from network.dfnet import DFNetv1, DFNetv2

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

class GatedConv(nn.Module):
    def __init__(self, inplane, outplane, kernel_size=3):
        super(GatedConv, self).__init__()
        self.feat_out1 = nn.Conv2d(inplane, inplane//2, 1)
        self.feat_out2 = nn.Conv2d(inplane, inplane//2, 1)
        self._gate_conv = nn.Sequential(
            Norm2d(inplane),
            nn.Conv2d(inplane, inplane//2, 1),
            nn.ReLU(),
            nn.Conv2d(inplane//2, 1, 1),
            Norm2d(1),
            nn.Sigmoid()
        )
        self.out = nn.Conv2d(outplane, outplane, 1)

    def forward(self, feats, x, base_idx=1):
        low_feature, h_feature = feats
        size = low_feature.size()[2:]

        feat1 = self.feat_out1(low_feature)
        feat2 = self.feat_out2(h_feature)
        feat2 = F.upsample(feat2, size=size, mode="bilinear", align_corners=True)

        alphas = self._gate_conv(torch.cat([feat1, feat2], dim=1))
        x = (x * (alphas + 1))
        x = self.out(x)
        return x

class SimpleGate(nn.Module):
    def __init__(self, inplane, outplane, kernel_size=3):
        super(SimpleGate, self).__init__()
        self.gate = conv_sigmoid(inplane, 2)

    def forward(self, feats, x, base_idx=1):
        size = feats[0].size()[2:]
        feature_origin = feats[base_idx]
        flow_gate = F.upsample(self.gate(feature_origin), size=size, mode="bilinear", align_corners=True)
        x = x*flow_gate
        return x


class AlignedModule(nn.Module):
    def __init__(self, inplane, outplane, kernel_size=3, gate='simple'):
        super(AlignedModule, self).__init__()

        self.down_h = nn.Conv2d(inplane, outplane, 1, bias=False)
        self.down_l = nn.Conv2d(inplane, outplane, 1, bias=False)
        self.flow_make = nn.Conv2d(outplane*2, 2, kernel_size=kernel_size, padding=1, bias=False)
        if gate == "simple":
            self.gate = SimpleGate(inplane, 2)
        elif gate == "GatedConv":
            self.gate = GatedConv(inplane, 2)
        elif gate == "":
            self.gate = None
        else:
            raise ValueError("no this type of gate")

    def forward(self, x, base_idx=1):
        low_feature, h_feature= x
        # h_feature_orign = h_feature
        feature_origin = x[base_idx]
        h, w = low_feature.size()[2:]
        size = (h, w)
        low_feature = self.down_l(low_feature)
        h_feature= self.down_h(h_feature)
        h_feature = F.upsample(h_feature, size=size, mode="bilinear", align_corners=True)
        flow = self.flow_make(torch.cat([h_feature, low_feature], 1))
        if self.gate:
            flow = self.gate(x, flow, base_idx)
        h_feature = self.flow_warp(feature_origin, flow, size=size)

        return h_feature

    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.size()

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        w = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        h = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((h.unsqueeze(2), w.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid)
        return output


class SpatialPath(nn.Module):
    def __init__(self, norm_layer=nn.BatchNorm2d):
        super(SpatialPath, self).__init__()
        self.conv1 = ConvBNReLU(3, 64, ks=7, stride=2, padding=3, norm_layer=norm_layer)
        self.conv2 = ConvBNReLU(64, 64, ks=3, stride=2, padding=1, norm_layer=norm_layer)
        self.conv3 = ConvBNReLU(64, 64, ks=3, stride=2, padding=1, norm_layer=norm_layer)
        self.conv_out = ConvBNReLU(64, 128, ks=1, stride=1, padding=0, norm_layer=norm_layer)
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


class BiAlignNet(nn.Module):
    def __init__(self, backbone='dfv2', gate="", aux=False, edgemap=False, num_classes=19, sync_bn=True, criterion=None):
        super(BiAlignNet, self).__init__()
        norm_layer = Norm2d if sync_bn else nn.BatchNorm2d
        # backbone
        if backbone == 'dfv1':
            self.backbone = DFNetv1(pretrained=True, norm_layer=norm_layer)
        elif backbone == 'dfv2':
            self.backbone = DFNetv2(pretrained=True, norm_layer=norm_layer)
        else:
            raise ValueError("Not a valid network arch")

        self.edgemap = edgemap

        self.ppm = PSPModule(512, 128, norm_layer=norm_layer)

        # spatial path
        self.sp = SpatialPath(norm_layer=norm_layer)

        if self.edgemap:
            self.spatial_out = BiSeNetOutput(128, 64, 1, up_factor=8, norm_layer=norm_layer)
            self.sigmoid_spatial = nn.Sigmoid()

        #fuse
        self.spatial2Context = AlignedModule(128, 128//2, gate=gate)
        self.context2Spatial = AlignedModule(128, 128//2, gate=gate)

        self.conv_out = BiSeNetOutput(256, 256, num_classes, up_factor=8, norm_layer=norm_layer)

        self.aux = aux
        if self.aux:
            self.cxt16_aux = BiSeNetOutput2(256, 256, num_classes, up_factor=16, norm_layer=norm_layer)
            self.cxt8_aux = BiSeNetOutput2(128, 256, num_classes, up_factor=16, norm_layer=norm_layer)

        self.criterion = criterion

    def forward(self, x, gts=None):

        feat8, feat16, feat32 = self.backbone(x)

        size = x.size()
        aspp = self.ppm(feat32)
        cxt_out = aspp


        # sp path
        feat_sp = self.sp(x)

        cxt_fuse = self.spatial2Context([feat_sp, cxt_out])
        spatial_fuse = self.context2Spatial([feat_sp, cxt_out], 0)

        feat_fuse = torch.cat([cxt_fuse, spatial_fuse], 1)

        final_out = self.conv_out(feat_fuse)
        final_out = F.upsample(final_out, size=size[2:], mode='bilinear', align_corners=True)

        if self.training:
            return self.criterion(final_out, gts)
        else:
            return final_out


def BiAlignNetDFNetV2SimpGateNoLoss(num_classes,  criterion):
    return BiAlignNet(num_classes=num_classes, backbone='dfv2', gate='simple', edgemap=False, criterion=criterion)

def BiAlignNetDFNetV2SimpGateDeepLoss(num_classes,  criterion):
    return BiAlignNet(num_classes=num_classes, backbone='dfv2', gate='simple', aux=True, criterion=criterion)

def BiAlignNetDFNetV1SimpGateDeepLoss(num_classes,  criterion):
    return BiAlignNet(num_classes=num_classes, backbone='dfv1', gate='simple', aux=True, criterion=criterion)


