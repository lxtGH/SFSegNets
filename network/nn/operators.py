"""
Custom Norm operators to enable sync BN, regular BN and for weight initialization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from network.nn.mynn import Upsample, Norm2d

class ChannelReasonModule(nn.Module):
    """
    Spatial CGNL block with dot production kernel for image classfication.
    """
    def __init__(self, inplanes, planes, groups=None, node_num=32):
        if groups == None:
            groups = planes
        self.groups = groups
        super(ChannelReasonModule, self).__init__()
        # conv theta
        self.t = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        # conv phi
        self.p = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        # conv g
        self.g = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        # conv z
        self.z = nn.Conv2d(planes, inplanes, kernel_size=1, stride=1,
                                                  groups=self.groups, bias=False)
        self.gn = nn.GroupNorm(num_groups=self.groups, num_channels=inplanes)

        self.node_num = node_num
        self.node_fea = planes // node_num

        #  Adjacency Matrix: A_g
        self.conv_adj = nn.Conv1d(self.node_num, self.node_num, kernel_size=1, bias=False)
        self.bn_adj = nn.BatchNorm1d(self.node_num)

        #  State Update Function: W_g
        self.conv_wg = nn.Conv1d(self.node_fea, self.node_fea, kernel_size=1, bias=False)
        self.bn_wg = nn.BatchNorm1d(self.node_fea)
        self.relu = nn.ReLU()

    def kernel(self, p, g, b, c, h, w):
        """The linear kernel (dot production).
        Args:
            p: output of conv phi
            g: output of conv g
            b: batch size
            c: channels number
            h: height of featuremaps
            w: width of featuremaps
        """
        p = p.view(b, 1, c * h * w)
        g = g.view(b, c * h * w, 1)

        att = torch.bmm(p, g)

        return att

    def forward(self, x):
        residual = x
        t = self.t(x)
        p = self.p(x)
        g = self.g(x)
        b, c, h, w = t.size()

        assert self.groups and self.groups > 1
        _c = int(c / self.groups)

        ps = torch.split(p, split_size_or_sections=_c, dim=1)
        gs = torch.split(g, split_size_or_sections=_c, dim=1)

        _t_sequences = []

        for i in range(self.groups):
            _x = self.kernel(ps[i], gs[i],
                             b, _c, h, w)
            _t_sequences.append(_x)

        x = torch.cat(_t_sequences, dim=1)

        z_idt = torch.split(x, split_size_or_sections=self.node_num, dim=1)
        res = []
        for i in z_idt:
            res.append(i)
        z_idt = torch.cat(res, dim=2)

        z = self.conv_adj(z_idt)
        z = self.bn_adj(z)
        z = self.relu(z)

        # Laplacian smoothing: (I - A_g)Z => Z - A_gZ
        z += z_idt

        z = z.transpose(1, 2).contiguous()
        z = self.conv_wg(z)
        z = self.bn_wg(z)
        z = self.relu(z)
        z = z.transpose(1, 2).contiguous()
        c, n, f = z.size()
        z = z.reshape(c, -1, 1).unsqueeze(2)
        x = z * t
        x = self.z(x)

        x = self.gn(x) + residual

        return x


class SRHead(nn.Module):
    """
        Squeeze-Reasoning Head
    """
    def __init__(self, in_dim, middle_dim=256, node=32):
        super(SRHead, self).__init__()
        self.down = nn.Sequential(nn.Conv2d(in_dim, middle_dim, kernel_size=1, bias=False),
                          Norm2d(middle_dim), nn.ReLU(inplace=True))
        self.sr = ChannelReasonModule(middle_dim, middle_dim, node_num=node)

    def forward(self, x):
        x_down = self.down(x)
        x_sr = self.sr(x_down)

        return x_sr

class _AtrousSpatialPyramidPoolingModule(nn.Module):
    """
    operations performed:
      1x1 x depth
      3x3 x depth dilation 6
      3x3 x depth dilation 12
      3x3 x depth dilation 18
      image pooling
      concatenate all together
      Final 1x1 conv
    """

    def __init__(self, in_dim, reduction_dim=256, output_stride=16, rates=(6, 12, 18)):
        super(_AtrousSpatialPyramidPoolingModule, self).__init__()

        if output_stride == 8:
            rates = [2 * r for r in rates]
        elif output_stride == 16:
            pass
        else:
            raise 'output stride of {} not supported'.format(output_stride)

        self.features = []
        # 1x1
        self.features.append(
            nn.Sequential(nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                          Norm2d(reduction_dim), nn.ReLU(inplace=True)))
        # other rates
        for r in rates:
            self.features.append(nn.Sequential(
                nn.Conv2d(in_dim, reduction_dim, kernel_size=3,
                          dilation=r, padding=r, bias=False),
                Norm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = torch.nn.ModuleList(self.features)

        # img level features
        self.img_pooling = nn.AdaptiveAvgPool2d(1)
        self.img_conv = nn.Sequential(
            nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
            Norm2d(reduction_dim), nn.ReLU(inplace=True))

    def forward(self, x):
        x_size = x.size()

        img_features = self.img_pooling(x)
        img_features = self.img_conv(img_features)
        img_features = Upsample(img_features, x_size[2:])
        out = img_features

        for f in self.features:
            y = f(x)
            out = torch.cat((out, y), 1)
        return out


def conv_bn_relu(in_channels, out_channels, kernel_size=1, stride=1, norm_layer=nn.BatchNorm2d):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, bias=False),
        norm_layer(out_channels),
        nn.ReLU(inplace=True)
    )


def conv_sigmoid(in_channels, out_channels, kernel_size=1, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size//2),
        nn.Sigmoid()
    )


class DenseBlock(nn.Sequential):
    def __init__(self, input_num, num1, num2, dilation_rate, drop_out, bn_start=True, norm_layer=nn.BatchNorm2d):
        super(DenseBlock, self).__init__()
        if bn_start:
            self.add_module('norm1', norm_layer(input_num)),

        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(in_channels=input_num, out_channels=num1, kernel_size=1)),

        self.add_module('norm2', norm_layer(num1)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(in_channels=num1, out_channels=num2, kernel_size=3,
                                            dilation=dilation_rate, padding=dilation_rate)),
        self.drop_rate = drop_out

    def forward(self, _input):
        feature = super(DenseBlock, self).forward(_input)
        if self.drop_rate > 0:
            feature = F.dropout2d(feature, p=self.drop_rate, training=self.training)
        return feature


class PSPModule(nn.Module):
    """
    Reference:
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """
    def __init__(self, features, out_features=512, sizes=(1, 2, 3, 6), norm_layer=nn.BatchNorm2d):
        super(PSPModule, self).__init__()

        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, out_features, size, norm_layer) for size in sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features+len(sizes)*out_features, out_features, kernel_size=1, padding=0, dilation=1, bias=False),
            norm_layer(out_features),
            nn.ReLU(),
            nn.Dropout2d(0.1)
            )

    def _make_stage(self, features, out_features, size, norm_layer):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, out_features, kernel_size=1, bias=False)
        bn = norm_layer(out_features)
        return nn.Sequential(prior, conv, bn)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear', align_corners=True) for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return bottle


class Aux_Module(nn.Module):
    def __init__(self, in_planes, num_classes=19, norm_layer=nn.BatchNorm2d):
        super(Aux_Module, self).__init__()

        self.aux = nn.Sequential(
                nn.Conv2d(in_planes, 256, kernel_size=3, stride=1, padding=1),
                norm_layer(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1),
                nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0, bias=True))

    def forward(self, x):
        res = self.aux(x)
        return res


class AlignedModule(nn.Module):

    def __init__(self, inplane, outplane, kernel_size=3):
        super(AlignedModule, self).__init__()
        self.down_h = nn.Conv2d(inplane, outplane, 1, bias=False)
        self.down_l = nn.Conv2d(inplane, outplane, 1, bias=False)
        self.flow_make = nn.Conv2d(outplane*2, 2, kernel_size=kernel_size, padding=1, bias=False)

    def forward(self, x):
        low_feature, h_feature = x
        h_feature_orign = h_feature
        h, w = low_feature.size()[2:]
        size = (h, w)
        low_feature = self.down_l(low_feature)
        h_feature= self.down_h(h_feature)
        h_feature = F.upsample(h_feature, size=size, mode="bilinear", align_corners=True)
        flow = self.flow_make(torch.cat([h_feature, low_feature], 1))
        h_feature = self.flow_warp(h_feature_orign, flow, size=size)

        return h_feature

    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.size()
        # n, c, h, w
        # n, 2, h, w

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        h = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w)
        w = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((w.unsqueeze(2), h.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        grid = grid + flow.permute(0, 2, 3, 1) / norm

        output = F.grid_sample(input, grid)
        return output


class ModuleHead(nn.Module):
    """
        CC-Net-like module head
    """
    def __init__(self, inplanes, interplanes, outplanes, num_classes, norm_layer=nn.BatchNorm2d, module=None):
        super(ModuleHead, self).__init__()
        self.conva = nn.Sequential(nn.Conv2d(inplanes, interplanes, 3, padding=1, bias=False),
                                   norm_layer(interplanes),
                                   nn.ReLU(interplanes))
        self.module = module
        self.convb = nn.Sequential(nn.Conv2d(interplanes, interplanes, 3, padding=1, bias=False),
                                   norm_layer(interplanes),
                                   nn.ReLU(interplanes))

        self.bottleneck = nn.Sequential(
            nn.Conv2d(inplanes + interplanes, outplanes, kernel_size=3, padding=1, dilation=1, bias=False),
            norm_layer(outplanes),
            nn.ReLU(outplanes),
            nn.Conv2d(outplanes, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, x):
        output = self.conva(x)
        output = self.module(output)
        output = self.convb(output)
        output = self.bottleneck(torch.cat([x, output], 1))
        return output