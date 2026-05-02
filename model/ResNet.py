import torch
import torch.nn as nn


class Conv2dBN(nn.Module):
    def __init__(self, in_chans, out_chans, kernel_size, stride=1, padding=1, bias=False):
        super(Conv2dBN, self).__init__()
        self.conv = nn.Conv2d(in_chans, out_chans, kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_chans)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_chans, out_chans, kernel_size, stride=1, with_conv_shortcut=False):
        super(ConvBlock, self).__init__()
        self.conv1 = Conv2dBN(in_chans, out_chans, kernel_size, stride=stride, padding=1)
        self.conv2 = Conv2dBN(out_chans, out_chans, kernel_size, padding=1)
        self.with_conv_shortcut = with_conv_shortcut
        if self.with_conv_shortcut:
            self.shortcut = Conv2dBN(in_chans, out_chans, kernel_size, stride=stride, padding=1)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.with_conv_shortcut:
            residual = self.shortcut(x)
        out = out + residual
        return out


class ResNet(nn.Module):
    def __init__(self, in_chans, use_extra_layers=False):
        super(ResNet, self).__init__()
        self.conv1 = Conv2dBN(in_chans, 64, 3, stride=1)
        self.maxpool = nn.MaxPool2d(2, 2)

        self.conv_block1 = ConvBlock(64, 64, 3)
        self.conv_block2 = ConvBlock(64, 64, 3)

        self.conv_block3 = ConvBlock(64, 128, 3, stride=2, with_conv_shortcut=True)
        self.conv_block4 = ConvBlock(128, 128, 3)

        self.conv_block5 = ConvBlock(128, 256, 3, stride=2, with_conv_shortcut=True)
        self.conv_block6 = ConvBlock(256, 256, 3)

        self.conv_block7 = ConvBlock(256, 512, 3, stride=2, with_conv_shortcut=True)
        self.conv_block8 = ConvBlock(512, 512, 3)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.use_extra_layers = use_extra_layers
        if use_extra_layers:
            self.fc_1 = nn.Linear(512, 512)
            self.fc = nn.Linear(512, 1)
        else:
            self.fc = nn.Linear(512, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.conv_block1(x)
        x = self.conv_block2(x)

        x = self.conv_block3(x)
        x = self.conv_block4(x)

        x = self.conv_block5(x)
        x = self.conv_block6(x)

        x = self.conv_block7(x)
        x = self.conv_block8(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        if self.use_extra_layers:
            x = self.fc_1(x)

        x = self.fc(x)

        return x  # Logits: [bs, num_classes]


if __name__ == '__main__':
    x = torch.randn(8, 3, 144, 144)
    model = ResNet(in_chans=3, use_extra_layers=False)
    print(model, model(x))
