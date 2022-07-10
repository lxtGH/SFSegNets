
import time

import torch
from torch import nn
from torch.nn import functional as F
from network.nn.mynn import initialize_weights, Norm2d, Upsample
from network import resnet_d as Resnet_Deep


def label_to_onehot(gt, num_classes, ignore_index=-1):
    '''
    gt: ground truth with size (N, H, W)
    num_classes: the number of classes of different label
    '''
    N, H, W = gt.size()
    x = gt
    x[x == ignore_index] = num_classes
    # convert label into onehot format
    onehot = torch.zeros(N, x.size(1), x.size(2), num_classes + 1).cuda()
    onehot = onehot.scatter_(-1, x.unsqueeze(-1), 1)

    return onehot.permute(0, 3, 1, 2)


class SpatialGather_Module(nn.Module):
    """
        Aggregate the context features according to the initial predicted probability distribution.
        Employ the soft-weighted method to aggregate the context.
    """
    def __init__(self, cls_num=0, scale=1, use_gt=False):
        super(SpatialGather_Module, self).__init__()
        self.cls_num = cls_num
        self.scale = scale
        self.use_gt = use_gt
        self.relu = nn.ReLU(inplace=True)

    def forward(self, feats, probs, gt_probs=None):
        if self.use_gt and gt_probs is not None:
            gt_probs = label_to_onehot(gt_probs.squeeze(1).type(torch.cuda.LongTensor), probs.size(1))
            batch_size, c, h, w = gt_probs.size(0), gt_probs.size(1), gt_probs.size(2), gt_probs.size(3)
            gt_probs = gt_probs.view(batch_size, c, -1)
            feats = feats.view(batch_size, feats.size(1), -1)
            feats = feats.permute(0, 2, 1) # batch x hw x c
            gt_probs = F.normalize(gt_probs, p=1, dim=2)# batch x k x hw
            ocr_context = torch.matmul(gt_probs, feats).permute(0, 2, 1).unsqueeze(3)# batch x k x c
            return ocr_context
        else:
            batch_size, c, h, w = probs.size(0), probs.size(1), probs.size(2), probs.size(3)
            probs = probs.view(batch_size, c, -1)
            feats = feats.view(batch_size, feats.size(1), -1)
            feats = feats.permute(0, 2, 1) # batch x hw x c
            probs = F.softmax(self.scale * probs, dim=2)# batch x k x hw
            ocr_context = torch.matmul(probs, feats).permute(0, 2, 1).unsqueeze(3)# batch x k x c
            return ocr_context


class SpatialOCR_Module(nn.Module):
    """
    Implementation of the OCR module:
    We aggregate the global object representation to update the representation for each pixel.

    use_gt=True: whether use the ground-truth label to compute the ideal object contextual representations.
    use_bg=True: use the ground-truth label to compute the ideal background context to augment the representations.
    use_oc=True: use object context or not.
    """
    def __init__(self,
                 in_channels,
                 key_channels,
                 out_channels,
                 scale=1,
                 dropout=0.1,
                 use_gt=False,
                 use_bg=False,
                 use_oc=True,
                 fetch_attention=False,
                 normal_layer=nn.BatchNorm2d):
        super(SpatialOCR_Module, self).__init__()
        self.use_gt = use_gt
        self.use_bg = use_bg
        self.use_oc = use_oc
        self.fetch_attention = fetch_attention
        self.object_context_block = ObjectAttentionBlock2D(in_channels,
                                                           key_channels,
                                                           scale,
                                                           use_gt,
                                                           use_bg,
                                                           fetch_attention,
                                                           normal_layer)
        if self.use_bg:
            if self.use_oc:
                _in_channels = 3 * in_channels
            else:
                _in_channels = 2 * in_channels
        else:
            _in_channels = 2 * in_channels

        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(_in_channels, out_channels, kernel_size=1, padding=0),
            normal_layer(out_channels),
            nn.Dropout2d(dropout)
        )

    def forward(self, feats, proxy_feats, gt_label=None):
        if self.use_gt and gt_label is not None:
            if self.use_bg:
                context, bg_context = self.object_context_block(feats, proxy_feats, gt_label)
            else:
                context = self.object_context_block(feats, proxy_feats, gt_label)
        else:
            if self.fetch_attention:
                context, sim_map = self.object_context_block(feats, proxy_feats)
            else:
                context = self.object_context_block(feats, proxy_feats)

        if self.use_bg:
            if self.use_oc:
                output = self.conv_bn_dropout(torch.cat([context, bg_context, feats], 1))
            else:
                output = self.conv_bn_dropout(torch.cat([bg_context, feats], 1))
        else:
            output = self.conv_bn_dropout(torch.cat([context, feats], 1))

        if self.fetch_attention:
            return output, sim_map
        else:
            return output


class _ObjectAttentionBlock(nn.Module):
    '''
    The basic implementation for object context block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        scale             : choose the scale to downsample the input feature maps (save memory cost)
        use_gt            : whether use the ground truth label map to compute the similarity map
        fetch_attention   : whether return the estimated similarity map
        bn_type           : specify the bn type
    Return:
        N X C X H X W
    '''
    def __init__(self,
                 in_channels,
                 key_channels,
                 scale=1,
                 use_gt=False,
                 use_bg=False,
                 fetch_attention=False,
                 norm_layer=nn.BatchNorm2d):
        super(_ObjectAttentionBlock, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.use_gt = use_gt
        self.use_bg = use_bg
        self.fetch_attention = fetch_attention
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_pixel = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0),
            norm_layer(self.key_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0),
            norm_layer(self.key_channels),
            nn.ReLU(),
        )
        self.f_object = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0),
            norm_layer(self.key_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0),
            norm_layer(self.key_channels),
            nn.ReLU(),
        )
        self.f_down = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0),
            norm_layer(self.key_channels),
            nn.ReLU(),
        )
        self.f_up = nn.Sequential(
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.in_channels,
                kernel_size=1, stride=1, padding=0),
            norm_layer(self.in_channels),
            nn.ReLU(),
        )

    def forward(self, x, proxy, gt_label=None):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        if self.scale > 1:
            x = self.pool(x)

        query = self.f_pixel(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)
        key = self.f_object(proxy).view(batch_size, self.key_channels, -1)
        value = self.f_down(proxy).view(batch_size, self.key_channels, -1)
        value = value.permute(0, 2, 1)

        if self.use_gt and gt_label is not None:
            gt_label = label_to_onehot(gt_label.squeeze(1).type(torch.cuda.LongTensor), proxy.size(2)-1)
            sim_map = gt_label[:, :, :, :].permute(0, 2, 3, 1).view(batch_size, h*w, -1)
            if self.use_bg:
                bg_sim_map = 1.0 - sim_map
                bg_sim_map = F.normalize(bg_sim_map, p=1, dim=-1)
            sim_map = F.normalize(sim_map, p=1, dim=-1)
        else:
            sim_map = torch.matmul(query, key)
            sim_map = (self.key_channels**-.5) * sim_map
            sim_map = F.softmax(sim_map, dim=-1)

        # add bg context ...
        context = torch.matmul(sim_map, value) # hw x k x k x c
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.key_channels, *x.size()[2:])
        context = self.f_up(context)
        if self.scale > 1:
            context = F.interpolate(input=context, size=(h, w), mode='bilinear', align_corners=True)

        if self.use_bg:
            bg_context = torch.matmul(bg_sim_map, value)
            bg_context = bg_context.permute(0, 2, 1).contiguous()
            bg_context = bg_context.view(batch_size, self.key_channels, *x.size()[2:])
            bg_context = self.f_up(bg_context)
            bg_context = F.interpolate(input=bg_context, size=(h, w), mode='bilinear', align_corners=True)
            return context, bg_context
        else:
            if self.fetch_attention:
                return context, sim_map
            else:
                return context


class ObjectAttentionBlock2D(_ObjectAttentionBlock):
    def __init__(self,
                 in_channels,
                 key_channels,
                 scale=1,
                 use_gt=False,
                 use_bg=False,
                 fetch_attention=False,
                 norm_layer=nn.BatchNorm2d):
        super(ObjectAttentionBlock2D, self).__init__(in_channels,
                                                     key_channels,
                                                     scale,
                                                     use_gt,
                                                     use_bg,
                                                     fetch_attention,
                                                     norm_layer=norm_layer)


class OCRnet(nn.Module):
    """
    Implement Encoding model
    A: stride8
    B: stride16
    with skip connections
    """
    def __init__(self, num_classes, trunk='resnet-50-deep', criterion=None, variant="D", fpn_dsn=True):
        super(OCRnet, self).__init__()
        self.criterion = criterion
        self.variant = variant
        self.num_classes = num_classes
        self.fpn_dsn = fpn_dsn

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

        in_channels = [1024, 2048]

        self.conv_3x3 = nn.Sequential(
            nn.Conv2d(in_channels[1], 512, kernel_size=3, stride=1, padding=1),
            Norm2d(512),
        )

        self.spatial_context_head = SpatialGather_Module(self.num_classes)
        self.spatial_ocr_head = SpatialOCR_Module(in_channels=512,
                                                  key_channels=256,
                                                  out_channels=512,
                                                  scale=1,
                                                  dropout=0.05,
                                                  normal_layer=Norm2d)

        self.head = nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)

        self.dsn_head = nn.Sequential(
            nn.Conv2d(in_channels[0], 512, kernel_size=3, stride=1, padding=1),
            Norm2d(512),
            nn.Dropout2d(0.05),
            nn.Conv2d(512, self.num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        )

        initialize_weights(self.head)

    def forward(self, x, gts=None, cal_inference_time=False):

        start = time.time()
        x_size = x.size()  # 800
        x0 = self.layer0(x)  # 400
        x1 = self.layer1(x0)  # 400
        x2 = self.layer2(x1)  # 100
        x3 = self.layer3(x2)  # 100
        x_dsn = self.dsn_head(x3)
        x4 = self.layer4(x3)  # 100
        x = self.conv_3x3(x4)
        context = self.spatial_context_head(x, x_dsn)
        x = self.spatial_ocr_head(x, context)
        out = self.head(x)

        out = Upsample(out, x_size[2:])

        end = time.time()

        results = (out, [x_dsn])
        main_out = results[0]

        if cal_inference_time:
            return end-start

        if self.training:
            if not self.fpn_dsn:
                return self.criterion(main_out, gts)
            else:
                return self.criterion(results, gts)

        return main_out


def OCRnet_r50(num_classes, criterion, variant='D'):
    """
    ResNet-50 Based Network
    """
    return OCRnet(num_classes, trunk='resnet-50-deep', criterion=criterion, variant=variant, fpn_dsn=True)