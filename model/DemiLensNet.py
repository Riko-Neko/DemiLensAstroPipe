from __future__ import print_function, division

import math
import os

import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from einops import rearrange
from matplotlib import pyplot as plt
from torchinfo import summary


class ConvBlock(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(ConvBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

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
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


# self.active = torch.nn.Sigmoid()
def _upsample_like(src, tar):
    src = F.interpolate(src, size=tar.shape[2:], mode='bilinear')
    return src


def conv_relu_bn(in_channel, out_channel, dirate):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=dirate,
                  dilation=dirate),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace=True)
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
        dx3 = self.dconv3(torch.cat((dx1, dx2), dim=1))

        out = self.conv2(torch.cat((x1, dx3), dim=1))
        return out


class Attention(nn.Module):
    def __init__(self, in_dim, in_feature, out_feature):
        super(Attention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=1, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=1, kernel_size=1)
        self.query_line = nn.Linear(in_features=in_feature, out_features=out_feature)
        self.key_line = nn.Linear(in_features=in_feature, out_features=out_feature)
        self.s_conv = nn.Conv2d(in_channels=1, out_channels=in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

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


class DeformableAttention(nn.Module):
    def __init__(self, stride=1):
        super(DeformableAttention, self).__init__()

        self.conv = nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()
        self.upsample = nn.Upsample(scale_factor=2)
        self.downavg = nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1)
        self.downmax = nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1)

        self.d_conv = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=stride)
        self.d_conv1 = nn.Conv2d(1, 1, kernel_size=3, padding=1, stride=stride)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = self.downavg(avg_out)
        max_out = self.downmax(max_out)

        d_avg_out = torch.sigmoid(self.d_conv(avg_out))
        d_max_out = torch.sigmoid(self.d_conv1(max_out))
        out = torch.cat([d_avg_out * max_out, d_max_out * avg_out], dim=1)
        out = self.conv(out)

        # mask = self.sigmoid(out)
        da_mask = self.sigmoid(self.upsample(out))
        return da_mask


class ConvAttention(nn.Module):
    def __init__(self, in_dim, in_feature, out_feature, use_DIA=True, vis=[False, None]):
        super(ConvAttention, self).__init__()
        self.conv = Conv(in_dim)
        self.dconv = DConv(in_dim)
        self.att = Attention(in_dim, in_feature, out_feature)
        self.gamma = nn.Parameter(torch.zeros(1))

        self._visualize = vis[0] if use_DIA else False
        self.use_DIA = use_DIA
        if use_DIA:
            self.datt = DeformableAttention(stride=1)
        if self._visualize:
            self.visualizer = ConvTransformerVisualizer(
                output_dir=os.path.join("./result/vis/clftdia_vis_output", f"{vis[1]}"))
            self._visualized = False
            self._visualize_threshold = 10

    def forward(self, x):
        q = self.conv(x)
        k = self.dconv(x)
        v = q + k
        att = self.att(x)
        x_mask = self.datt(x) if self.use_DIA else None
        v_att = torch.matmul(att, v)
        out = (self.gamma * v_att + v) * x_mask + x if self.use_DIA else self.gamma * v_att + v + x

        if self._visualize and not self._visualized:
            if self._visualize_threshold == 0:
                self._visualized = True
                print("[\033[97mModel Info\033[0m] You are applying visualization of CLFT-DIA...")
                self.visualizer.visualize_sample(x, att, v_att, x_mask, out)
            else:
                self._visualize_threshold -= 1

        return out


class FeedForward(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FeedForward, self).__init__()
        self.conv = conv_relu_bn(in_dim, out_dim, 1)
        # self.x_conv = nn.Conv2d(in_dim, out_dim, kernel_size=1)
        self.x_conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.conv(x)
        x = self.x_conv(x)
        return x + out


class ConvTransformer(nn.Module):
    def __init__(self, in_dim, out_dim, in_feature, out_feature, use_DIA=True, vis=[False, None]):
        super(ConvTransformer, self).__init__()
        self.attention = ConvAttention(in_dim, in_feature, out_feature, use_DIA, vis)
        self.feedforward = FeedForward(in_dim, out_dim)

    def forward(self, x):
        x = self.attention(x)
        out = self.feedforward(x)
        return out


class FrequencyChannelAttention(nn.Module):
    """Frequency Channel Attention as in FcaNet: Extract frequency information using 2D DCT basis functions and generate channel weights."""

    def __init__(self, channels, height, width, reduction=32, groups=2):
        super().__init__()
        assert channels % groups == 0, "Number of channels should be divisible by groups"
        self.groups = groups
        self.group_channels = channels // groups
        self.height = height
        self.width = width
        # Select the frequency indices (u, v) for each group, which can be customized
        # For example, the first group uses low frequency (0,0), and the last group uses high frequency (height-1, width-1)
        freqs = [(0, 0), (height - 1, width - 1)]
        if groups <= len(freqs):
            self.freqs = freqs[:groups]
        else:
            # If the number of groups is more than the preset frequencies, they can be cycled or other frequencies can be customized
            self.freqs = self._generate_freqs()
        # Construct the DCT basis weight matrix, shape: (groups, H, W)
        bases = []
        for (u, v) in self.freqs:
            # Generate basis function matrix with height H and width W
            pi = math.pi
            grid_h = torch.arange(self.height).unsqueeze(1)  # H x 1
            grid_w = torch.arange(self.width).unsqueeze(0)  # 1 x W
            basis_u = torch.cos(pi * u * (2 * grid_h + 1) / (2 * self.height))  # H x 1
            basis_v = torch.cos(pi * v * (2 * grid_w + 1) / (2 * self.width))  # 1 x W
            basis = basis_u @ basis_v  # Outer product, get H x W
            bases.append(basis)
        # bases is a list of (H x W) tensor
        self.register_buffer('dct_bases', torch.stack(bases, dim=0))  # (groups, H, W)

        # Two-layer FC for channel attention
        hidden = max(channels // reduction, 8)
        self.fc1 = nn.Linear(channels, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, channels, bias=True)

    def _generate_freqs(self):
        freqs = []
        for i in range(self.groups):
            u = (i * self.height) // self.groups
            v = (i * self.width) // self.groups
            freqs.append((u, v))
        return freqs

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        # 1) DCT Compression: Compute the spectrum component for each group of channels
        x_groups = torch.chunk(x, self.groups, dim=1)  # Each group (B, C//g, H, W)
        freq_list = []
        for i, xg in enumerate(x_groups):
            # Dot product of DCT basis function with group features and sum
            # xg: (B, Cg, H, W), basis: (H, W) -> Expand to (B, Cg, H, W) for broadcasting
            basis = self.dct_bases[i]  # H x W
            # Sum weighted by channel
            # (B, Cg, H, W) * (H, W) -> (B, Cg, H, W), sum over H and W
            coeff = (xg * basis.unsqueeze(0).unsqueeze(0)).sum(dim=(2, 3))  # (B, Cg)
            freq_list.append(coeff)
        # cat -> (B, C)
        freq_feat = torch.cat(freq_list, dim=1)  # (B, C)
        # 2) Generate attention with two-layer FC
        attn = F.relu(self.fc1(freq_feat))  # (B, hidden)
        attn = torch.sigmoid(self.fc2(attn))  # (B, C)
        # 3) Apply channel weights
        attn = attn.view(B, C, 1, 1)  # (B, C, 1, 1)
        out = x * attn  # (B, C, H, W)

        return out


class FcaAfnoBlock(nn.Module):
    """A plug-and-play block combining Frequency Channel Attention from FcaNet and FFT-based frequency global filtering."""

    def __init__(self, channels, height, width, reduction=16, groups=2, init_std=0.005, vis=[False, None]):
        super().__init__()
        self.height = height
        self.width = width
        # Frequency Channel Attention 子模块（假定在外部定义）
        self.fca = FrequencyChannelAttention(channels, height, width, reduction, groups)

        # 复数频域滤波器参数 (learnable real & imag)
        # shape: (1, C, H, Wf) where Wf = width//2 + 1 for rfft2
        Wf = width // 2 + 1
        real_init = torch.normal(mean=1.0, std=init_std, size=(1, channels, height, Wf))
        imag_init = torch.normal(mean=0.0, std=init_std, size=(1, channels, height, Wf))
        self.freq_filter_real = nn.Parameter(real_init)
        self.freq_filter_imag = nn.Parameter(imag_init)

        # per-channel multiplicative scale (broadcast over spatial+freq dims)
        # shape (1, C, 1, 1) so it will broadcast to (1,C,H,Wf)
        self.scale = nn.Parameter(torch.ones(1, channels, 1, 1))

        # per-channel gating param: gate = sigmoid(alpha) in (0,1)
        # initialize alpha to a negative value so gate starts small (prefer keeping original x early)
        self.alpha = nn.Parameter(torch.full((1, channels, 1, 1), fill_value=-3.0))

        # keep visualization plumbing if you had it
        self._visualize = vis[0]
        if self._visualize:
            self.visualizer = FrequencyVisualizer(
                output_dir=os.path.join("./result/vis/fa_vis_output", f"{vis[1]}"))
            self._visualized = False
            self._visualize_threshold = 0

    def forward(self, x):
        # x: (B, C, H, W)
        x_attn = self.fca(x)  # (B, C, H, W)

        # FFT to complex spectrum (B, C, H, Wf)
        Xf = torch.fft.rfft2(x_attn, dim=(-2, -1))

        # build complex learned weight (allow both amplitude & phase to change)
        weight = torch.complex(self.freq_filter_real, self.freq_filter_imag)  # (1, C, H, Wf)

        # apply per-channel scale (broadcasting to H and Wf dims)
        # scale shape (1,C,1,1) -> broadcast to (1,C,H,Wf)
        scaled_weight = weight * self.scale

        # apply filter in freq domain
        Xf = scaled_weight * Xf

        # inverse FFT back to real spatial domain
        x_ifft = torch.fft.irfft2(Xf, s=(self.height, self.width), dim=(-2, -1))  # (B,C,H,W)

        # per-channel gate: decide how much of x_ifft to add/use
        gate = torch.sigmoid(self.alpha)  # shape (1, C, 1, 1)
        # broadcast gate to (B,C,H,W) when multiplying
        out = (1.0 - gate) * x + gate * x_ifft

        # visualization (unchanged behavior)
        if self._visualize and not getattr(self, "_visualized", False):
            if self._visualize_threshold == 0:
                self._visualized = True
                print("[\033[97mModel Info\033[0m] You are applying visualization of FA...")
                self.visualizer.visualize_sample(x, x_attn, x_ifft, Xf, out)
            else:
                self._visualize_threshold -= 1

        return out


class DemiLensNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, dim=64, ori_h=144, extra_fc=False, e_factor=[2, 4, 8, 16],
                 ablated=[False, False], visualize=False):
        super(DemiLensNet, self).__init__()
        self.visualize = visualize
        self.use_FA = ablated[0]
        self.use_DIA = ablated[1]
        if visualize:
            self._visualize_threshold = 10
            self._visualized = False

        filters = [dim, dim * e_factor[0], dim * e_factor[1], dim * e_factor[2], dim * e_factor[3]]
        features = [ori_h // 2, ori_h // 4, ori_h // 8, ori_h // 16]

        self.maxpools = nn.ModuleList([nn.MaxPool2d(kernel_size=2, stride=2) for _ in range(4)])
        # self.fa_layer1 = FrequencyAwareModule(in_channels = in_ch, freq_size = 96, k = 4, hidden_dim = 192, vis = [self.visualize, 'fa1'])
        self.Conv1 = ConvBlock(in_ch=in_ch, out_ch=filters[0])
        self.fa_layer1 = FcaAfnoBlock(channels=filters[0], reduction=filters[0] // 8, height=ori_h, width=ori_h,
                                      groups=filters[0] // 4, vis=[self.visualize, 'fa1'])
        self.Convtans2 = ConvTransformer(filters[0], filters[1], pow(features[0], 2), features[0], use_DIA=self.use_DIA,
                                         vis=[self.visualize, 'ct1'])
        self.Convtans3 = ConvTransformer(filters[1], filters[2], pow(features[1], 2), features[1], use_DIA=self.use_DIA,
                                         vis=[self.visualize, 'ct2'])
        self.Convtans4 = ConvTransformer(filters[2], filters[3], pow(features[2], 2), features[2], use_DIA=self.use_DIA,
                                         vis=[self.visualize, 'ct3'])
        self.Conv5 = DconvBlock(in_ch=filters[3], out_ch=filters[4])
        # self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        if extra_fc:
            self.fc = nn.Sequential(
                nn.Linear(filters[4] * 4, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(512, 256),
                nn.Linear(256, out_ch)
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(filters[4] * 4, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(512, out_ch)
            )

    def forward(self, x):
        if self.visualize and not self._visualized:
            if self._visualize_threshold == 0:
                self._visualized = True
                OrgBandsVisualizer(x, output_dir=os.path.join("./result/vis/bands_vis_output"))
                print("[\033[97mModel Info\033[0m] You are applying visualization of ORG BANDS...")
            else:
                self._visualize_threshold -= 1

        e1 = self.Conv1(x)
        f1 = self.fa_layer1(e1) if self.use_FA == True else e1

        e2 = self.maxpools[0](f1)
        e2 = self.Convtans2(e2)

        e3 = self.maxpools[1](e2)
        e3 = self.Convtans3(e3)

        e4 = self.maxpools[2](e3)
        e4 = self.Convtans4(e4)

        e5 = self.maxpools[3](e4)
        e5 = self.Conv5(e5)

        x = self.avgpool(e5)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


"""
Key Feature Visualization
"""


def OrgBandsVisualizer(tensor, output_dir="vis_output"):
    assert tensor.ndim == 4 and tensor.size(1) == 3, "Input tensor should be (B, 3, H, W)"
    relative_output_dir = output_dir
    abs_output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), relative_output_dir)
    os.makedirs(abs_output_dir, exist_ok=True)
    output_dir = abs_output_dir

    r_band = tensor[0, 0].cpu().numpy()
    g_band = tensor[0, 1].cpu().numpy()
    i_band = tensor[0, 2].cpu().numpy()

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(r_band, cmap='gray')
    ax.set_title('r band')
    ax.axis('off')
    plt.tight_layout()
    save_path = os.path.join(output_dir, "r_band.png")
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(g_band, cmap='gray')
    ax.set_title('g band')
    ax.axis('off')
    plt.tight_layout()
    save_path = os.path.join(output_dir, "g_band.png")
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(i_band, cmap='gray')
    ax.set_title('i band')
    ax.axis('off')
    plt.tight_layout()
    save_path = os.path.join(output_dir, "i_band.png")
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


class ConvTransformerVisualizer:
    def __init__(self, output_dir="vis_output"):
        relative_output_dir = output_dir
        abs_output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), relative_output_dir)
        os.makedirs(abs_output_dir, exist_ok=True)
        self.output_dir = abs_output_dir
        self._prepare_dirs()

    def _prepare_dirs(self):
        for subfolder in ['x', 'attn', 'v_attn', 'mask', 'out']:
            path = os.path.join(self.output_dir, subfolder)
            os.makedirs(path, exist_ok=True)

    def visualize_sample(self, x: torch.Tensor, attn: torch.Tensor, x_attn: torch.Tensor, mask: torch.Tensor,
                         out: torch.Tensor):
        self._plot_tensor(x, subdir='x', label='x')
        self._plot_tensor(attn, subdir='attn', label='attn')
        self._plot_tensor(x_attn, subdir='v_attn', label='v_attn')
        self._plot_tensor(mask, subdir='mask', label='mask')
        self._plot_tensor(out, subdir='out', label='out')

    def _plot_tensor(self, tensor: torch.Tensor, subdir: str, label: str):
        if tensor.dim() != 4:
            print(f"Warning: Tensor '{label}' should be 4D (B, C, H, W), got shape {tensor.shape}")
            return

        tensor = tensor.detach().cpu()
        B, C, H, W = tensor.shape

        for i in range(C):
            img = tensor[0, i].numpy()
            img = self._normalize(img)
            filename = f"{label}_channel_{i}.png"
            self._save_image(img, os.path.join(subdir, filename), title=f"{label} - Channel {i}")

    def _normalize(self, img):
        min_val, max_val = img.min(), img.max()
        if max_val - min_val == 0:
            return img  # avoid division by zero
        return (img - min_val) / (max_val - min_val)

    def _save_image(self, img, relative_path, title=""):
        path = os.path.join(self.output_dir, relative_path)
        plt.figure()
        plt.imshow(img, cmap='viridis')
        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(path)
        plt.close()


class FrequencyVisualizer:
    def __init__(self, output_dir="vis_output"):
        relative_output_dir = output_dir
        abs_output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), relative_output_dir)
        os.makedirs(abs_output_dir, exist_ok=True)
        self.output_dir = abs_output_dir
        for sub in ['x', 'x_attn', 'x_ifft', 'Xf', 'out']:
            os.makedirs(os.path.join(self.output_dir, sub), exist_ok=True)

    def visualize_sample(self, x: torch.Tensor, x_attn: torch.Tensor, x_ifft: torch.Tensor, Xf: torch.Tensor,
                         out: torch.Tensor):
        # x: (B, C, H, W)
        # fused_feat: (B, hidden_dim)
        # out: (B, C, H, W)
        self._plot_tensor(x, subdir='x', label='x')
        self._plot_tensor(x_attn, subdir='x_attn', label='x_attn')
        self._plot_tensor(x_ifft, subdir='x_ifft', label='x_ifft')
        self._plot_complex_vector(Xf, subdir='Xf', label='Xf')
        self._plot_tensor(out, subdir='out', label='out')

    def _plot_tensor(self, tensor: torch.Tensor, subdir: str, label: str):
        tensor = tensor.detach().cpu()
        if tensor.dim() != 4:
            print(f"Warning: Tensor '{label}' expected 4D, got {tensor.shape}")
            return
        B, C, H, W = tensor.shape
        # Only visualize first batch
        for i in range(C):
            img = tensor[0, i].numpy()
            img = self._normalize(img)
            fname = os.path.join(self.output_dir, subdir, f"{label}_ch{i}.png")
            plt.figure()
            plt.imshow(img, cmap='viridis')
            plt.title(f"{label} channel {i}")
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(fname)
            plt.close()

    def _plot_complex_vector(self, vec: torch.Tensor, subdir: str, label: str):
        vec = vec.detach().cpu()

        if torch.is_complex(vec):
            mag = torch.abs(vec)
            phase = torch.angle(vec)

            mag_dir = os.path.join(self.output_dir, subdir, "magnitude")
            phase_dir = os.path.join(self.output_dir, subdir, "phase")
            os.makedirs(mag_dir, exist_ok=True)
            os.makedirs(phase_dir, exist_ok=True)

            self._plot_complex_vector(mag, os.path.join(subdir, "magnitude"), f"{label}_mag")
            self._plot_complex_vector(phase, os.path.join(subdir, "phase"), f"{label}_phase")
            return

        if vec.dim() == 4:
            channel_dir = os.path.join(self.output_dir, subdir)
            os.makedirs(channel_dir, exist_ok=True)

            B, C, H, W = vec.shape
            for c in range(min(C, 3)):
                channel_data = vec[0, c].numpy()
                fname = os.path.join(channel_dir, f"{label}_ch{c}.png")

                plt.figure(figsize=(6, 8))
                plt.imshow(channel_data, cmap='viridis')
                plt.colorbar()
                plt.title(f"{label} Channel {c} (HxW: {H}x{W})")
                plt.tight_layout()
                plt.savefig(fname)
                plt.close()
            return

        if vec.dim() == 2:
            B, D = vec.shape
            bar = vec[0].numpy()
            fname = os.path.join(self.output_dir, subdir, f"{label}.png")

            plt.figure(figsize=(12, 6))
            plt.bar(range(D), bar)
            plt.title(f"{label} (D={D})")
            plt.tight_layout()
            plt.savefig(fname)
            plt.close()
            return

        print(f"Warning: Cannot visualize '{label}' with shape {vec.shape}")

    def _normalize(self, img):
        min_val, max_val = img.min(), img.max()
        if max_val - min_val == 0:
            return img  # avoid division by zero
        return (img - min_val) / (max_val - min_val)


def generate_wave_image(H, W):
    x = torch.arange(W).float().unsqueeze(0).repeat(H, 1)
    y = torch.arange(H).float().unsqueeze(1).repeat(1, W)

    red = torch.sin(x / 10)  # 水平波
    green = torch.sin(y / 10)  # 垂直波
    blue = torch.sin((x + y) / 15)  # 对角波

    img = torch.stack([red, green, blue], dim=0)
    return img


if __name__ == '__main__':
    # x = torch.randn(8, 3, 96, 96)
    x = torch.stack([generate_wave_image(144, 144) for _ in range(8)], dim=0)  # shape (8, 3, 144, 144)
    model = DemiLensNet(ori_h=144, dim=32, extra_fc=False, e_factor=[2, 6, 12, 20], ablated=[True, True],
                        visualize=False)
    print(model)
    summary(model, input_size=(1, 3, 144, 144))
    print(model(x))
