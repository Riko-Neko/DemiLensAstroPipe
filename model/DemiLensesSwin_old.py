import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features = None, out_features = None, act_layer = nn.GELU, drop = 0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DyT(nn.Module):
    def __init__(self, dim, init_alpha = 0.5):
        super(DyT, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1) * init_alpha)
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        return self.gamma * x + self.beta


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        pretrained_window_size (tuple[int]): The height and width of the window in pre-training.
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias = True, attn_drop = 0., proj_drop = 0.,
                 pretrained_window_size = [0, 0]):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad = True)

        # mlp to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(nn.Linear(2, 512, bias = True),
                                     nn.ReLU(inplace = True),
                                     nn.Linear(512, num_heads, bias = False))

        # get relative_coords_table
        relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype = torch.float32)
        relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype = torch.float32)
        relative_coords_table = torch.stack(
            torch.meshgrid([relative_coords_h,
                            relative_coords_w])).permute(1, 2, 0).contiguous().unsqueeze(0)  # 1, 2*Wh-1, 2*Ww-1, 2
        if pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :, 0] /= (pretrained_window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (pretrained_window_size[1] - 1)
        else:
            relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)
        relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            torch.abs(relative_coords_table) + 1.0) / np.log2(8)

        self.register_buffer("relative_coords_table", relative_coords_table)

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias = False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim = -1)

    def forward(self, x, mask = None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad = False), self.v_bias))
        qkv = F.linear(input = x, weight = self.qkv.weight, bias = qkv_bias)
        qkv = qkv.reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # cosine attention
        attn = (F.normalize(q, dim = -1) @ F.normalize(k, dim = -1).transpose(-2, -1))
        logit_scale = torch.clamp(self.logit_scale.to(x.device),
                                  max = torch.log(torch.tensor(1. / 0.01, device = x.device))).exp()
        attn = attn * logit_scale

        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, ' \
               f'pretrained_window_size={self.pretrained_window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class DemiLensesSwinBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        pretrained_window_size (int): Window size in pre-training.
    """

    def __init__(self, dim, input_resolution, num_heads, window_size = 7, shift_size = 0,
                 mlp_ratio = 4., qkv_bias = True, drop = 0., attn_drop = 0., drop_path = 0.,
                 act_layer = nn.GELU, norm_layer = nn.LayerNorm, pretrained_window_size = 0):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size = to_2tuple(self.window_size), num_heads = num_heads,
            qkv_bias = qkv_bias, attn_drop = attn_drop, proj_drop = drop,
            pretrained_window_size = to_2tuple(pretrained_window_size))

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features = dim, hidden_features = mlp_hidden_dim, act_layer = act_layer, drop = drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts = (-self.shift_size, -self.shift_size), dims = (1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask = self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts = (self.shift_size, self.shift_size), dims = (1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(self.norm1(x))

        # FFN
        x = x + self.drop_path(self.norm2(self.mlp(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer = nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias = False)
        self.norm = norm_layer(2 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.reduction(x)
        x = self.norm(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        flops += H * W * self.dim // 2
        return flops


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, H, W, reduction = 16):  # 添加H和W参数
        super().__init__()
        self.H = H
        self.W = W
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, 1, bias = False)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, 1, bias = False)
        self.relu = nn.ReLU(inplace = True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, L, C = x.shape
        x = x.view(B, self.H, self.W, C).permute(0, 3, 1, 2)  # [B, C, H, W]

        avg_pool = self.avg_pool(x)  # [B, C, 1, 1]
        max_pool = self.max_pool(x)  # [B, C, 1, 1]

        avg_out = self.fc2(self.relu(self.fc1(avg_pool)))
        max_out = self.fc2(self.relu(self.fc1(max_pool)))

        out = avg_out + max_out
        out = self.sigmoid(out)

        x = x * out
        x = x.permute(0, 2, 3, 1).view(B, L, C)
        return x


class SpatialAttention(nn.Module):
    def __init__(self, in_channels, H, W):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size = 3, padding = 1, bias = False)
        self.H = H
        self.W = W

    def forward(self, x):
        B, L, C = x.shape
        x = x.view(B, self.H, self.W, C).permute(0, 3, 1, 2)

        avg_pool = x.mean(dim = 1, keepdim = True)
        max_pool = x.max(dim = 1, keepdim = True).values
        concat = torch.cat([avg_pool, max_pool], dim = 1)
        attend = self.conv(concat).sigmoid()
        x = x * attend

        x = x.permute(0, 2, 3, 1).view(B, L, C)
        return x


class FrequencySpatialAttention(nn.Module):
    def __init__(self, in_channels, H, W):
        super().__init__()
        self.H = H
        self.W = W
        self.dct_conv = nn.Conv2d(in_channels * 2, in_channels * 2, 1)  # 处理实部和虚部
        self.idct_conv = nn.Conv2d(in_channels * 2, in_channels, 1)  # 输出还原

        self.freq_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels * 2, in_channels // 8, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 8, in_channels * 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, L, C = x.shape
        x = x.view(B, self.H, self.W, C).permute(0, 3, 1, 2)  # [B, C, H, W]

        x_dct = torch.fft.rfft2(x, norm = 'ortho')
        x_dct = torch.cat([x_dct.real, x_dct.imag], dim = 1)  # [B, 2C, H, W//2+1]

        x_dct = self.dct_conv(x_dct)
        freq_weights = self.freq_att(x_dct)
        x_dct = x_dct * freq_weights

        x_idct = torch.complex(x_dct[:, :C], x_dct[:, C:])
        x_out = torch.fft.irfft2(x_idct, s = (self.H, self.W), norm = 'ortho')
        x_out = self.idct_conv(x_out)  # [B, C, H, W]

        return x_out.permute(0, 2, 3, 1).view(B, L, C)


class EnhancedSpatialAttention(nn.Module):
    def __init__(self, in_channels, num_heads = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads

        self.to_qkv = nn.Linear(in_channels, 3 * num_heads * self.head_dim)
        self.to_out = nn.Linear(num_heads * self.head_dim, in_channels)

        self.spatial_gate = nn.Sequential(
            nn.Linear(in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, L, C = x.shape

        qkv = self.to_qkv(x).chunk(3, dim = -1)  # 3 * [B, L, num_heads*D]
        q, k, v = [t.view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
                   for t in qkv]  # [B,
        # H, L, D]

        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = attn.softmax(dim = -1)
        out = torch.matmul(attn, v)  # [B, H, L, D]

        out = out.permute(0, 2, 1, 3).reshape(B, L, -1)  # [B, L, H*D]
        out = self.to_out(out)  # [B, L, C]

        spatial_weights = self.spatial_gate(x)  # [B, L, 1]
        return out * spatial_weights + x


class EnhancedSpatialAttention2D(nn.Module):
    def __init__(self, in_channels, H, W, num_heads = 4):
        super().__init__()
        self.H, self.W = H, W
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads

        self.to_qkv = nn.Conv2d(in_channels, 3 * num_heads * self.head_dim, 1)

        self.to_out = nn.Conv2d(num_heads * self.head_dim, in_channels, 1)

        self.spatial_gate = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size = 3, padding = 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, L, C = x.shape
        x_2d = x.view(B, self.H, self.W, C).permute(0, 3, 1, 2)  # [B, C, H, W]

        qkv = self.to_qkv(x_2d).chunk(3, dim = 1)  # 3 * [B, num_heads * head_dim, H, W]
        q, k, v = [t.view(B, self.num_heads, self.head_dim, self.H, self.W)
                   for t in qkv]  # [B, num_heads, head_dim, H, W]

        # Compute attention scores per position
        attn = torch.einsum("bhdij,bhdij->bhij", q, k)  # [B, num_heads, H, W]
        attn = attn / math.sqrt(self.head_dim)
        attn = torch.softmax(attn, dim = 2)

        # Apply attention to v
        out = torch.einsum("bhij,bhdij->bhdij", attn, v)  # [B, num_heads, head_dim, H, W]

        out = out.reshape(B, self.num_heads * self.head_dim, self.H, self.W)  # Fix applied here
        out = self.to_out(out)  # [B, C, H, W]

        gate = self.spatial_gate(x_2d)  # [B, 1, H, W]
        out = out * gate + x_2d

        return out.permute(0, 2, 3, 1).view(B, L, C)


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (Any, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        add_spatial_attention (bool): If True, add spatial attention layer after each block. Default: False.
        add_channel_attention (bool): If True, add channel attention layer after each block. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size, mlp_ratio = 4., qkv_bias = True, drop = 0.,
                 attn_drop = 0., drop_path = 0., norm_layer = DyT, downsample = None, add_spatial_attention = False,
                 add_channel_attention = False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.attenuation_layer = nn.GELU()

        # build blocks
        self.blocks = nn.ModuleList([
            DemiLensesSwinBlock(dim = dim, input_resolution = input_resolution,
                                num_heads = num_heads, window_size = window_size,
                                shift_size = 0 if (i % 2 == 0) else window_size // 2,
                                mlp_ratio = mlp_ratio,
                                qkv_bias = qkv_bias,
                                drop = drop, attn_drop = attn_drop,
                                drop_path = drop_path[i] if isinstance(drop_path, list) else drop_path,
                                norm_layer = norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim = dim, norm_layer = norm_layer)
        else:
            self.downsample = None

        # add spatial attention layer
        if add_spatial_attention:
            H, W = self.input_resolution
            self.sa = SpatialAttention(dim, H, W)
        else:
            self.sa = None

        # add channel attention layer
        if add_channel_attention:
            H, W = input_resolution
            self.ca = ChannelAttention(dim, H, W, reduction = 16)
        else:
            self.ca = None

        # add cooperation attention layer
        if add_channel_attention and add_spatial_attention:
            H, W = input_resolution
            self.coa = CoAttention(dim, H, W)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)

        if self.sa is not None and self.ca is not None:
            x = x + self.coa(x)
            x = x + self.sa(self.ca(x))
        elif self.sa is not None:
            x = x + self.sa(x)
        elif self.ca is not None:
            x = x + self.ca(x)

        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class PatchEmbedConv(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size = 224, patch_size = 4, in_chans = 3, embed_dim = 96, norm_layer = None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.conv_init = nn.Conv2d(in_chans, in_chans, kernel_size = 3, stride = 1, padding = 1)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size = patch_size, stride = patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.conv_init(x)
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class _DemiLensesSwin(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        add_spatial_attention (bool): If True, add spatial attention layer after each block. Default: False.
        add_channel_attention (bool): If True, add channel attention layer after each block. Default: False.
    """

    def __init__(self, img_size = 96, patch_size = 4, in_chans = 3, num_classes = 1, embed_dim = 64,
                 depths = [2, 2, 2, 2], num_heads = [3, 6, 12, 24], window_size = 6, mlp_ratio = 4., qkv_bias = True,
                 drop_rate = 0., attn_drop_rate = 0., drop_path_rate = 0.1, ape = False, patch_norm = True,
                 add_spatial_attention = False, add_channel_attention = False, **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.norm_layer = DyT
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbedConv(
            img_size = img_size, patch_size = patch_size, in_chans = in_chans, embed_dim = embed_dim,
            norm_layer = nn.LayerNorm if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std = .02)

        self.pos_drop = nn.Dropout(p = drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            add_sa = True if (i_layer < self.num_layers - 1) else False
            add_ca = True if (i_layer < self.num_layers - 1) else False
            layer = BasicLayer(dim = int(embed_dim * 2 ** i_layer),
                               input_resolution = (patches_resolution[0] // (2 ** i_layer),
                                                   patches_resolution[1] // (2 ** i_layer)),
                               depth = depths[i_layer],
                               num_heads = num_heads[i_layer],
                               window_size = window_size,
                               mlp_ratio = self.mlp_ratio,
                               qkv_bias = qkv_bias,
                               drop = drop_rate,
                               attn_drop = attn_drop_rate,
                               drop_path = dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer = self.norm_layer,
                               downsample = PatchMerging if (i_layer < self.num_layers - 1) else None,
                               add_spatial_attention = add_sa if add_spatial_attention else False,
                               add_channel_attention = add_ca if add_channel_attention else False)
            self.layers.append(layer)

        self.norm = nn.LayerNorm(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std = .02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"cpb_mlp", "logit_scale", 'relative_position_bias_table'}

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.num_classes
        return flops


class BeforeDownsampleBasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage, with before and after downsampling.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (Any, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        add_spatial_attention (bool): If True, add spatial attention layer after each block. Default: True.
        add_channel_attention (bool): If True, add channel attention layer after each block. Default: True.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size, mlp_ratio = 4., qkv_bias = True, drop = 0.,
                 attn_drop = 0., drop_path = 0., norm_layer = nn.LayerNorm, downsample = None,
                 add_spatial_attention = True, add_channel_attention = True):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.attenuation_layer = nn.GELU()
        self.blocks = nn.ModuleList([
            DemiLensesSwinBlock(dim = dim, input_resolution = input_resolution,
                                num_heads = num_heads, window_size = window_size,
                                shift_size = 0 if (i % 2 == 0) else window_size // 2,
                                mlp_ratio = mlp_ratio,
                                qkv_bias = qkv_bias, drop = drop, attn_drop = attn_drop,
                                drop_path = drop_path[i] if isinstance(drop_path, list) else drop_path,
                                norm_layer = norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim = dim, norm_layer = norm_layer)
        else:
            self.downsample = None

        # add spatial attention layer
        if add_spatial_attention:
            H, W = self.input_resolution
            self.sa = SpatialAttention(dim, H, W)
        else:
            self.sa = None

        # add channel attention layer
        if add_channel_attention:
            H, W = input_resolution
            self.ca = ChannelAttention(dim, H, W, reduction = 16)
        else:
            self.ca = None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)

        # Apply channel attention and spatial attention
        if self.sa is not None and self.ca is not None:
            x = x + self.sa(self.ca(x))
        elif self.sa is not None:
            x = self.sa(x)
        elif self.ca is not None:
            x = self.ca(x)

        # Set the output without downsampling for final feature fusion
        before_downsample = x
        if self.downsample is not None:
            x = self.downsample(x)
        return before_downsample, x


class DemiLensesSwin(nn.Module):
    r""" A Specialized Swin Transformer with attention-based feature fusion for image classification.
        This model is a modified version of Swin Transformer with attention-based feature fusion and extra modules.
        The trivial improvement work is based on the excellent work of the Swin Transformer.
        They deserve more credit for their contributions.
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        D (int): Dimension of the feature fusion space. Default: 768
        num_attention_heads (int): Number of attention heads for self-attention. Default: 8
        add_spatial_attention (bool): If True, add spatial attention layer after each block. Default: True.
        add_channel_attention (bool): If True, add channel attention layer after each block. Default: True.
    """

    def __init__(self, img_size = 96, patch_size = 4, in_chans = 3, num_classes = 1, embed_dim = 64,
                 depths = [2, 2, 6, 2], num_heads = [3, 6, 12, 24], window_size = 6, mlp_ratio = 4., qkv_bias = True,
                 drop_rate = 0., attn_drop_rate = 0., drop_path_rate = 0., norm_layer = DyT, ape = False,
                 patch_norm = True, D = 768, num_attention_heads = 8, add_spatial_attention = True,
                 add_channel_attention = True, **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.patch_embed = PatchEmbedConv(
            img_size = img_size, patch_size = patch_size, in_chans = in_chans, embed_dim = embed_dim,
            norm_layer = nn.LayerNorm if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        self.pos_drop = nn.Dropout(p = drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            add_sa = True if (i_layer < self.num_layers - 1) else False
            add_ca = True if (i_layer < self.num_layers - 1) else False
            layer = BeforeDownsampleBasicLayer(dim = int(self.embed_dim * 2 ** i_layer),
                                               input_resolution = (patches_resolution[0] // (2 ** i_layer),
                                                                   patches_resolution[1] // (2 ** i_layer)),
                                               depth = depths[i_layer],
                                               num_heads = num_heads[i_layer],
                                               window_size = window_size,
                                               mlp_ratio = mlp_ratio,
                                               qkv_bias = qkv_bias,
                                               drop = drop_rate,
                                               attn_drop = attn_drop_rate,
                                               drop_path = dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                               norm_layer = norm_layer,
                                               downsample = PatchMerging if (
                                                       i_layer < self.num_layers - 1) else None,
                                               add_spatial_attention = add_sa if add_spatial_attention else False,
                                               add_channel_attention = add_ca if add_channel_attention else False)
            self.layers.append(layer)
        total_channels = sum([self.embed_dim * 2 ** i for i in range(self.num_layers)])
        self.head = nn.Linear(total_channels, num_classes)
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std = .02)
        # New additions for attention-based feature fusion
        self.D = D
        self.num_stages = self.num_layers
        self.linear_proj = nn.ModuleList(
            [nn.Linear(C_i, D) for C_i in [self.embed_dim * 2 ** i for i in range(self.num_stages)]])
        self.cls_token = nn.Parameter(torch.zeros(1, D))
        trunc_normal_(self.cls_token, std = .02)
        self.self_attention = nn.MultiheadAttention(D, num_attention_heads)
        self.debug = False

    def forward(self, x):
        B_input = x.size(0)
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        features = []
        for layer in self.layers:
            before_downsample, _ = layer(x)
            features.append(before_downsample)
            x = layer(x)[1] if layer.downsample else before_downsample

        # Feature pooling and projection
        fused_features = []
        for i, feat in enumerate(features):
            H_i = self.patches_resolution[0] // (2 ** i)
            W_i = self.patches_resolution[1] // (2 ** i)
            feat = feat.transpose(1, 2).reshape(B_input, -1, H_i, W_i)
            pooled_feat = torch.nn.AdaptiveAvgPool2d(1)(feat).flatten(1)
            fused_features.append(pooled_feat)

        proj_f_i = [self.linear_proj[i](feat) for i, feat in enumerate(fused_features)]

        # Create sequence with cls_token
        cls_token_batch = self.cls_token.repeat(B_input, 1)  # (16, 768)
        proj_f_i_stack = torch.stack(proj_f_i, dim = 1)  # (16, 4, 768)
        seq = torch.cat([cls_token_batch.unsqueeze(1), proj_f_i_stack], dim = 1)  # (16, 5, 768)

        # Adjust seq for MultiheadAttention
        seq = seq.permute(1, 0, 2)  # (5, 16, 768)

        # Self-attention
        output, attention_weights = self.self_attention(seq, seq, seq, need_weights = True,
                                                        average_attn_weights = False)

        # Debug output
        if self.debug:
            print(f"Input batch size: {B_input}")
            print(f"cls_token_batch shape: {cls_token_batch.shape}")
            print(f"proj_f_i[0] shape: {proj_f_i[0].shape}")
            print(f"seq shape: {seq.shape}")
            print(f"attention_weights shape: {attention_weights.shape}")

        # Extract attention weights
        weights = []
        for i in range(self.num_stages):
            weight_i = attention_weights[:, :, 0, i + 1].mean(dim = 1)  # (16,)
            weights.append(weight_i)

        # Weighted fusion
        weighted_f_i = [weight[:, None] * feat for weight, feat in zip(weights, fused_features)]
        fused_feat = torch.cat(weighted_f_i, dim = 1)

        # Final output
        x = self.head(fused_feat)
        return x
