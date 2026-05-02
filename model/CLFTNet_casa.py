from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from einops import rearrange


class ConvBlock(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(ConvBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size = 3, stride = 1, padding = 1, bias = True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace = True),
            nn.Conv2d(out_ch, out_ch, kernel_size = 3, stride = 1, padding = 1, bias = True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace = True))

    def forward(self, x):
        x = self.conv(x)
        return x


class UpConv(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(UpConv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor = 2, mode = 'bilinear'),
            nn.Conv2d(in_ch, out_ch, kernel_size = 3, stride = 1, padding = 1, bias = True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace = True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


# self.active = torch.nn.Sigmoid()
def _upsample_like(src, tar):
    src = F.interpolate(src, size = tar.shape[2:], mode = 'bilinear')
    return src


def conv_relu_bn(in_channel, out_channel, dirate):
    return nn.Sequential(
        nn.Conv2d(in_channels = in_channel, out_channels = out_channel, kernel_size = 3, stride = 1, padding = dirate,
                  dilation = dirate),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace = True)
    )


class DconvBlock(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(DconvBlock, self).__init__()
        self.conv1 = conv_relu_bn(in_ch, out_ch, 1)
        self.dconv1 = conv_relu_bn(out_ch, out_ch // 2, 2)
        self.dconv2 = conv_relu_bn(out_ch // 2, out_ch // 2, 4)
        self.dconv3 = conv_relu_bn(out_ch, out_ch, 2)
        self.conv2 = conv_relu_bn(out_ch * 2, out_ch, 1)

    def forward(self, x):
        x1 = self.conv1(x)
        dx1 = self.dconv1(x1)
        dx2 = self.dconv2(dx1)
        dx3 = self.dconv3(torch.cat((dx1, dx2), dim = 1))

        out = self.conv2(torch.cat((x1, dx3), dim = 1))
        return out


class Attention(nn.Module):
    def __init__(self, in_dim, in_feature, out_feature):
        super(Attention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels = in_dim, out_channels = 1, kernel_size = 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim, out_channels = 1, kernel_size = 1)
        self.query_line = nn.Linear(in_features = in_feature, out_features = out_feature)
        self.key_line = nn.Linear(in_features = in_feature, out_features = out_feature)
        self.s_conv = nn.Conv2d(in_channels = 1, out_channels = in_dim, kernel_size = 1)
        self.softmax = nn.Softmax(dim = -1)

    def forward(self, x):
        q = rearrange(self.query_line(rearrange(self.query_conv(x), 'b 1 h w -> b (h w)')), 'b h -> b h 1')
        k = rearrange(self.key_line(rearrange(self.key_conv(x), 'b 1 h w -> b (h w)')), 'b h -> b 1 h')
        att = rearrange(torch.matmul(q, k), 'b h w -> b 1 h w')
        att = self.softmax(self.s_conv(att))
        return att


class Conv(nn.Module):
    def __init__(self, in_dim):
        super(Conv, self).__init__()
        self.convs = nn.ModuleList([conv_relu_bn(in_dim, in_dim, 1) for _ in range(3)])

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        return x


class DConv(nn.Module):
    def __init__(self, in_dim):
        super(DConv, self).__init__()
        dilation = [2, 4, 2]
        self.dconvs = nn.ModuleList([conv_relu_bn(in_dim, in_dim, dirate) for dirate in dilation])

    def forward(self, x):
        for dconv in self.dconvs:
            x = dconv(x)
        return x


class ConvAttention(nn.Module):
    def __init__(self, in_dim, in_feature, out_feature):
        super(ConvAttention, self).__init__()
        self.conv = Conv(in_dim)
        self.dconv = DConv(in_dim)
        self.att = Attention(in_dim, in_feature, out_feature)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        q = self.conv(x)
        k = self.dconv(x)
        v = q + k
        att = self.att(x)
        out = torch.matmul(att, v)
        return self.gamma * out + v + x


class FeedForward(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FeedForward, self).__init__()
        self.conv = conv_relu_bn(in_dim, out_dim, 1)
        # self.x_conv = nn.Conv2d(in_dim, out_dim, kernel_size=1)
        self.x_conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size = 1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace = True)
        )

    def forward(self, x):
        out = self.conv(x)
        x = self.x_conv(x)
        return x + out


class ConvTransformer(nn.Module):
    def __init__(self, in_dim, out_dim, in_feature, out_feature):
        super(ConvTransformer, self).__init__()
        self.attention = ConvAttention(in_dim, in_feature, out_feature)
        self.feedforward = FeedForward(in_dim, out_dim)

    def forward(self, x):
        x = self.attention(x)
        out = self.feedforward(x)
        return out


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio = 16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias = False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias = False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size = 7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding = padding, bias = False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim = 1, keepdim = True)
        max_out, _ = torch.max(x, dim = 1, keepdim = True)
        x = torch.cat([avg_out, max_out], dim = 1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio = 16, kernel_size = 7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x


class CLFTNetCaSa(nn.Module):
    def __init__(self, in_ch = 3, out_ch = 1, dim = 64, ori_h = 144, extra_fc = False, e_factor = [2, 4, 8, 16]):
        super(CLFTNetCaSa, self).__init__()
        filters = [dim, dim * e_factor[0], dim * e_factor[1], dim * e_factor[2], dim * e_factor[3]]
        features = [ori_h // 2, ori_h // 4, ori_h // 8, ori_h // 16]

        self.maxpools = nn.ModuleList([nn.MaxPool2d(kernel_size = 2, stride = 2) for _ in range(4)])
        self.Conv1 = ConvBlock(in_ch = in_ch, out_ch = filters[0])
        self.cbam1 = CBAM(filters[0])

        self.Convtans2 = ConvTransformer(filters[0], filters[1], pow(features[0], 2), features[0])
        self.cbam2 = CBAM(filters[1])

        self.Convtans3 = ConvTransformer(filters[1], filters[2], pow(features[1], 2), features[1])
        self.cbam3 = CBAM(filters[2])

        self.Convtans4 = ConvTransformer(filters[2], filters[3], pow(features[2], 2), features[2])
        self.cbam4 = CBAM(filters[3])

        self.Conv5 = DconvBlock(in_ch = filters[3], out_ch = filters[4])
        self.cbam5 = CBAM(filters[4])

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(filters[4], 512),
            nn.ReLU(inplace = True),
            nn.Dropout(0.5),
            nn.Linear(512, 256) if extra_fc else None,
            nn.Linear(256, out_ch) if extra_fc else nn.Linear(512, out_ch)
        )

    def forward(self, x):
        e1 = self.Conv1(x)
        e1 = self.cbam1(e1)

        e2 = self.maxpools[0](e1)
        e2 = self.Convtans2(e2)
        e2 = self.cbam2(e2)

        e3 = self.maxpools[1](e2)
        e3 = self.Convtans3(e3)
        e3 = self.cbam3(e3)

        e4 = self.maxpools[2](e3)
        e4 = self.Convtans4(e4)
        e4 = self.cbam4(e4)

        e5 = self.maxpools[3](e4)
        e5 = self.Conv5(e5)
        e5 = self.cbam5(e5)

        x = self.avgpool(e5)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    x = torch.randn(8, 3, 144, 144)
    model = CLFTNetCaSa(ori_h = 144)
    print(model(x))
