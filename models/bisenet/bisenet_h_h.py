# code from https://github.com/hydxqing/BiSeNet-pytorch-chapter5/blob/master/model/build_BiSeNet.py
# code from https://github.com/PaddlePaddle/PaddleSeg/blob/release/2.8/paddleseg/models/bisenetv1.py

from torch import nn, cat
from config import cfg


class ConvBnRelu(nn.Module):
    def __init__(self, in_planes, out_planes, ksize, stride, pad, dilation=1,
                 groups=1, has_bn=True, norm_layer=nn.BatchNorm2d, bn_eps=1e-5,
                 has_relu=True, inplace=True, has_bias=False):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=ksize,
                              stride=stride, padding=pad,
                              dilation=dilation, groups=groups, bias=has_bias)
        self.has_bn = has_bn
        if self.has_bn:
            self.bn = norm_layer(out_planes, eps=bn_eps).cuda()
        self.has_relu = has_relu
        if self.has_relu:
            self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):

        self.conv = self.conv.cuda()  # Convert the convolution layer to CUDA format
        x = self.conv(x)
        if self.has_bn:
            # Convert the batch normalization layer to CUDA format

            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)

        return x


class FeatureFusion(nn.Module):
    def __init__(self, in_planes, out_planes, reduction=1, norm_layer=nn.BatchNorm2d):
        super(FeatureFusion, self).__init__()
        self.conv_1x1 = ConvBnRelu(in_planes, out_planes, 1, 1, 0,
                                   has_bn=True, norm_layer=norm_layer,
                                   has_relu=True, has_bias=False)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBnRelu(out_planes, out_planes // reduction, 1, 1, 0,
                       has_bn=False, norm_layer=norm_layer,
                       has_relu=True, has_bias=False),
            ConvBnRelu(out_planes // reduction, out_planes, 1, 1, 0,
                       has_bn=False, norm_layer=norm_layer,
                       has_relu=False, has_bias=False),
            ConvBnRelu(out_planes, out_planes // reduction, 1, 1, 0,
                       has_bn=False, norm_layer=norm_layer,
                       has_relu=True, has_bias=False),
            ConvBnRelu(out_planes // reduction, out_planes, 1, 1, 0,
                       has_bn=False, norm_layer=norm_layer,
                       has_relu=False, has_bias=False),
            ConvBnRelu(out_planes, out_planes // reduction, 1, 1, 0,
                       has_bn=False, norm_layer=norm_layer,
                       has_relu=True, has_bias=False),
            ConvBnRelu(out_planes // reduction, out_planes, 1, 1, 0,
                       has_bn=False, norm_layer=norm_layer,
                       has_relu=False, has_bias=False),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        fm = cat([x1, x2], dim=1)
        fm = self.conv_1x1(fm)
        fm_se = self.channel_attention(fm)
        output = fm + fm * fm_se
        return output


class BiSeNetHead(nn.Module):
    def __init__(self, in_planes, out_planes, scale, is_aux=False, norm_layer=nn.BatchNorm2d):
        super(BiSeNetHead, self).__init__()
        if is_aux:
            self.conv_3x3 = ConvBnRelu(in_planes, 128, 3, 1, 1, has_bn=True, norm_layer=norm_layer, has_relu=True,
                                       has_bias=False)
        else:
            self.conv_3x3 = ConvBnRelu(in_planes, 64, 3, 1, 1, has_bn=True, norm_layer=norm_layer, has_relu=True,
                                       has_bias=False)
        # self.dropout = nn.Dropout(0.1)
        if is_aux:
            self.conv_1x1 = nn.Conv2d(
                128, out_planes, kernel_size=1, stride=1, padding=0)
        else:
            self.conv_1x1 = nn.Conv2d(
                64, out_planes, kernel_size=1, stride=1, padding=0)
        self.scale = scale

    def forward(self, x):
        fm = self.conv_3x3(x)
        # fm = self.dropout(fm)
        self.conv_1x1 = self.conv_1x1.cuda()
        output = self.conv_1x1(fm)
        if self.scale > 1:
            upsample = nn.Upsample(scale_factor=self.scale, mode='bilinear', align_corners=True)
            output = upsample(output)
        return output


class AttentionRefinement(nn.Module):
    def __init__(self, in_planes, out_planes,
                 norm_layer=nn.BatchNorm2d):
        super(AttentionRefinement, self).__init__()
        self.conv_3x3 = ConvBnRelu(in_planes, out_planes, 3, 1, 1,
                                   has_bn=True, norm_layer=norm_layer,
                                   has_relu=True, has_bias=False)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBnRelu(out_planes, out_planes, 1, 1, 0,
                       has_bn=True, norm_layer=norm_layer,
                       has_relu=False, has_bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        fm = self.conv_3x3(x)
        fm_se = self.channel_attention(fm)
        fm = fm * fm_se

        return fm


class SpatialPath(nn.Module):
    def __init__(self, in_planes, out_planes, norm_layer=nn.BatchNorm2d):
        super(SpatialPath, self).__init__()
        inner_channel = 64
        self.conv_7x7 = ConvBnRelu(in_planes, inner_channel, 7, 2, 3,
                                   has_bn=True, norm_layer=norm_layer,
                                   has_relu=True, has_bias=False)
        self.conv_3x3_1 = ConvBnRelu(inner_channel, inner_channel, 3, 2, 1,
                                     has_bn=True, norm_layer=norm_layer,
                                     has_relu=True, has_bias=False)
        self.conv_3x3_2 = ConvBnRelu(inner_channel, inner_channel, 3, 2, 1,
                                     has_bn=True, norm_layer=norm_layer,
                                     has_relu=True, has_bias=False)
        self.conv_1x1 = ConvBnRelu(inner_channel, out_planes, 1, 4, 0,
                                   has_bn=True, norm_layer=norm_layer,
                                   has_relu=True, has_bias=False)

    def forward(self, x):
        x = self.conv_7x7(x)
        x = self.conv_3x3_1(x)
        x = self.conv_3x3_2(x)
        output = self.conv_1x1(x)

        return output


class SeparableConvBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, has_relu=True,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBnRelu, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                               bias=False)
        self.point_wise_cbr = ConvBnRelu(in_channels, out_channels, 1, 1, 0, has_bn=True, norm_layer=norm_layer,
                                         has_relu=has_relu, has_bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.point_wise_cbr(x)
        return x


class Block(nn.Module):
    expansion = 4

    def __init__(self, in_channels, mid_out_channels, has_proj, stride, dilation=1, norm_layer=nn.BatchNorm2d):
        super(Block, self).__init__()
        self.has_proj = has_proj

        if has_proj:
            self.proj = SeparableConvBnRelu(in_channels, mid_out_channels * self.expansion, 3, stride, 1,
                                            has_relu=False, norm_layer=norm_layer)

        self.residual_branch = nn.Sequential(
            SeparableConvBnRelu(in_channels, mid_out_channels, 3, stride, dilation, dilation, has_relu=True,
                                norm_layer=norm_layer),
            SeparableConvBnRelu(mid_out_channels, mid_out_channels,
                                3, 1, 1, has_relu=True, norm_layer=norm_layer),
            SeparableConvBnRelu(mid_out_channels, mid_out_channels * self.expansion, 3, 1, 1, has_relu=False,
                                norm_layer=norm_layer))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        shortcut = x
        if self.has_proj:
            shortcut = self.proj(x)

        residual = self.residual_branch(x)
        output = self.relu(shortcut + residual)

        return output


class Xception(nn.Module):
    def __init__(self, block, layers, channels, norm_layer=nn.BatchNorm2d):
        super(Xception, self).__init__()

        self.in_channels = 8
        self.conv1 = ConvBnRelu(3, self.in_channels, 3, 2, 1, has_bn=True, norm_layer=norm_layer, has_relu=True,
                                has_bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, norm_layer, layers[0], channels[0], stride=2)
        self.layer2 = self._make_layer(block, norm_layer, layers[1], channels[1], stride=2)
        self.layer3 = self._make_layer(block, norm_layer, layers[2], channels[2], stride=2)
        self.layer4 = self._make_layer(block, norm_layer, layers[3], channels[3], stride=2)
        self.layer5 = self._make_layer(block, norm_layer, layers[4], channels[4], stride=2)

    def _make_layer(self, block, norm_layer, blocks, mid_out_channels, stride=1):
        layers = []
        has_proj = True if stride > 1 else False
        layers.append(block(self.in_channels, mid_out_channels, has_proj, stride=stride, norm_layer=norm_layer))
        self.in_channels = mid_out_channels * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_channels, mid_out_channels, has_proj=False, stride=1, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        blocks = []
        x = self.layer1(x)
        blocks.append(x)
        x = self.layer2(x)
        blocks.append(x)
        x = self.layer3(x)
        blocks.append(x)
        x = self.layer4(x)
        blocks.append(x)
        x = self.layer5(x)
        blocks.append(x)

        return blocks


def load_xception39():
    model = Xception(Block, [4, 8, 4, 8, 16], [16, 32, 64, 128, 256])

    return model


class BiSeNet_h_h(nn.Module):
    def __init__(self, num_classes=cfg.DATA.NUM_CLASSES, norm_layer=nn.BatchNorm2d):
        super(BiSeNet_h_h, self).__init__()

        self.context_path = load_xception39()
        self.spatial_path = SpatialPath(3, 256, norm_layer).cuda()
        conv_c = 256
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBnRelu(1024, conv_c, 1, 1, 0, has_bn=True, has_relu=True, has_bias=False, norm_layer=norm_layer)
        )
        self.arms = [AttentionRefinement(1024, conv_c, norm_layer), AttentionRefinement(512, conv_c, norm_layer)]
        self.refines = [ConvBnRelu(conv_c, conv_c, 3, 1, 1, has_bn=True, norm_layer=norm_layer, has_relu=True,
                                   has_bias=False),
                        ConvBnRelu(conv_c, conv_c, 3, 1, 1, has_bn=True, norm_layer=norm_layer, has_relu=True,
                                   has_bias=False)]
        self.heads = [BiSeNetHead(conv_c, num_classes, 16, True, norm_layer),
                      BiSeNetHead(conv_c, num_classes, 8, True, norm_layer),
                      BiSeNetHead(conv_c * 2, num_classes, 8, False, norm_layer)]
        self.ffm = FeatureFusion(conv_c * 2, conv_c * 2, 1, norm_layer)

    def forward(self, x):
        spatial_out = self.spatial_path(x)
        context_blocks = self.context_path(x)
        context_blocks.reverse()

        global_context = self.global_context(context_blocks[0])
        upsample = nn.Upsample(size=context_blocks[0].size()[2:], mode='bilinear', align_corners=True)
        global_context = upsample(global_context)
        last_fm = global_context
        pred_out = []

        for i, (fm, arm, refine) in enumerate(zip(context_blocks[:2], self.arms, self.refines)):
            fm = arm(fm)
            fm += last_fm
            upsample = nn.Upsample(size=context_blocks[i + 1].size()[2:], mode='bilinear', align_corners=True)
            last_fm = upsample(fm)
            last_fm = refine(last_fm)
            pred_out.append(last_fm)
        context_out = last_fm

        concate_fm = self.ffm(spatial_out, context_out)
        pred_out.append(concate_fm)

        output = []
        if self.training:
            for i, head in enumerate(self.heads):
                out = head(pred_out[i])
                output.append(out)
        else:
            out = self.heads[-1](pred_out[-1])
            output.append(out)
        target_size = (224, 448)
        upsampled_tensor = nn.functional.interpolate(output[-1], size=target_size, mode='bilinear', align_corners=False)

        return upsampled_tensor  # this surely must be wrong
