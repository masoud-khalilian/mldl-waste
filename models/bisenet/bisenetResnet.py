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
            self.conv_1x1 = nn.Conv2d(128, out_planes, kernel_size=1, stride=1, padding=0)
        else:
            self.conv_1x1 = nn.Conv2d(64, out_planes, kernel_size=1, stride=1, padding=0)
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
        self.conv_3x3 = ConvBnRelu(in_planes, out_planes, 3, 1, 1, has_bn=True, norm_layer=norm_layer, has_relu=True,
                                   has_bias=False)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBnRelu(out_planes, out_planes, 1, 1, 0, has_bn=True, norm_layer=norm_layer, has_relu=False,
                       has_bias=False),
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
        self.conv_1x1 = ConvBnRelu(inner_channel, out_planes, 1, 1, 0,
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
            SeparableConvBnRelu(mid_out_channels, mid_out_channels, 3, 1, 1, has_relu=True, norm_layer=norm_layer),
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


# Basic building block for ResNet
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


# ResNet-18 architecture
class ResNet18(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000):
        super(ResNet18, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        # self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        blocks = []
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        blocks.append(x)
        x = self.layer2(x)
        blocks.append(x)
        x = self.layer3(x)
        blocks.append(x)
        # x = self.layer4(x)
        # blocks.append(x)
        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)
        return blocks


# Create ResNet-18 model
def resnet18(num_classes=1000):
    return ResNet18(BasicBlock, [2, 2, 2, 2], num_classes)


class BiSeNetResnet(nn.Module):
    def __init__(self, num_classes=cfg.DATA.NUM_CLASSES, norm_layer=nn.BatchNorm2d):
        super(BiSeNetResnet, self).__init__()

        self.context_path = resnet18(num_classes=num_classes)
        self.spatial_path = SpatialPath(3, 128, norm_layer).cuda()
        conv_c = 128
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBnRelu(256, conv_c, 1, 1, 0, has_bn=True, has_relu=True, has_bias=False, norm_layer=norm_layer))
        self.arms = [AttentionRefinement(256, conv_c, norm_layer), AttentionRefinement(128, conv_c, norm_layer)]
        self.refines = [
            ConvBnRelu(conv_c, conv_c, 3, 1, 1, has_bn=True, norm_layer=norm_layer, has_relu=True, has_bias=False),
            ConvBnRelu(conv_c, conv_c, 3, 1, 1, has_bn=True, norm_layer=norm_layer, has_relu=True, has_bias=False)]
        self.heads = [BiSeNetHead(conv_c, num_classes, 16, True, norm_layer),
                      BiSeNetHead(conv_c, num_classes, 8, True, norm_layer),
                      BiSeNetHead(conv_c * 2, num_classes, 8, False, norm_layer)]
        self.ffm = FeatureFusion(conv_c * 2, conv_c * 2, 1, norm_layer)

    def forward(self, x):
        # print("we reached the forward of bisenet resnet18")
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
            upsample = nn.Upsample(
                size=context_blocks[i + 1].size()[2:], mode='bilinear', align_corners=True)
            last_fm = upsample(fm)
            last_fm = refine(last_fm)
            pred_out.append(last_fm)
        context_out = last_fm

        # Resize context_out to match the size of spatial_out
        desired_size = spatial_out.shape[2:]  # Get the spatial dimensions from spatial_out
        resized_context_out = nn.functional.interpolate(context_out, size=desired_size, mode='bilinear',
                                                        align_corners=False)

        concate_fm = self.ffm(spatial_out, resized_context_out)
        pred_out.append(concate_fm)

        output = []
        if self.training:
            for i, head in enumerate(self.heads):
                out = head(pred_out[i])
                output.append(out)
        else:
            out = self.heads[-1](pred_out[-1])
            output.append(out)
        return output[-1]  # this surely must be wrong
