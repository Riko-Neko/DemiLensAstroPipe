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


class QuantumLinear(nn.Module):
    """Quantum Linear Layer with complex weights"""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.real = nn.Linear(in_dim, out_dim)
        self.imag = nn.Linear(in_dim, out_dim)
        nn.init.orthogonal_(self.real.weight)  # 正交初始化保持酉性
        nn.init.orthogonal_(self.imag.weight)

    def forward(self, x):
        return self.real(x), self.imag(x)


class DensityMatrix(nn.Module):
    """修正后的密度矩阵计算模块"""

    def __init__(self):
        super().__init__()

    def forward(self, psi_real, psi_imag):
        """
        输入维度：
        psi_real: [B, num_heads, N, head_dim]
        psi_imag: [B, num_heads, N, head_dim]
        """
        # 计算实部密度矩阵 [B, nh, N, hd, hd]
        rho_real = torch.einsum('bhin,bhjm->bhnij', psi_real, psi_real) + \
                   torch.einsum('bhin,bhjm->bhnij', psi_imag, psi_imag)

        # 计算虚部密度矩阵 [B, nh, N, hd, hd]
        rho_imag = torch.einsum('bhin,bhjm->bhnij', psi_real, psi_imag) - \
                   torch.einsum('bhin,bhjm->bhnij', psi_imag, psi_real)

        return rho_real, rho_imag


class QuantumAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias = True, attn_drop = 0., proj_drop = 0., qk_scale = None):
        super().__init__()
        self.dim = dim
        self.window_size = to_2tuple(window_size)
        self.num_window_tokens = self.window_size[0] * self.window_size[1]
        self.num_heads = num_heads

        # 关键修改：严格校验维度划分
        assert (dim // 2) % num_heads == 0, f"dim//2 ({dim // 2}) 必须能被 num_heads ({num_heads}) 整除"
        self.qk_head_dim = (dim // 2) // num_heads
        self.v_head_dim = dim // num_heads

        self.scale = qk_scale or self.qk_head_dim ** -0.5

        # 量子投影层（确保输出维度为dim//2）
        self.q_proj = QuantumLinear(dim, dim // 2)
        self.k_proj = QuantumLinear(dim, dim // 2)
        self.v_proj = nn.Linear(dim, dim)

        self.density = DensityMatrix()

        # 确保酉矩阵参数与qk_head_dim一致
        self.H_params = nn.Parameter(
            torch.randn(num_heads, self.qk_head_dim, self.qk_head_dim, 2)
        )
        nn.init.orthogonal_(self.H_params[..., 0])
        nn.init.orthogonal_(self.H_params[..., 1])

        # 位置编码与qk_head_dim对齐
        self.pos_real = nn.Parameter(torch.randn(self.num_window_tokens, self.qk_head_dim))
        self.pos_imag = nn.Parameter(torch.randn(self.num_window_tokens, self.qk_head_dim))

        # 输出层
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)

    def apply_unitary(self, rho_real, rho_imag):
        B, nh, N, hd, _ = rho_real.shape  # hd应等于self.qk_head_dim

        # 构造酉矩阵（确保维度正确）
        H_real = self.H_params[..., 0] - self.H_params[..., 0].transpose(-1, -2)
        H_imag = self.H_params[..., 1] - self.H_params[..., 1].transpose(-1, -2)
        H = torch.complex(H_real, H_imag)
        U = torch.matrix_exp(1j * H)  # [nh, hd, hd]

        # 扩展维度以匹配输入
        U_expanded = U.unsqueeze(0).unsqueeze(2)  # [1, nh, 1, hd, hd]
        U_expanded = U_expanded.expand(B, -1, N, -1, -1)  # [B, nh, N, hd, hd]

        # 应用酉变换（确保维度一致）
        rho = torch.complex(rho_real, rho_imag)
        rho_transformed = U_expanded @ rho @ U_expanded.conj().transpose(-1, -2)

        return rho_transformed.real, rho_transformed.imag

    def forward(self, x, mask = None):
        B_, N, C = x.shape

        # 投影并重塑维度
        q_real, q_imag = self.q_proj(x)  # 各 [B_, N, C//2]
        k_real, k_imag = self.k_proj(x)
        v = self.v_proj(x)  # [B_, N, C]

        # 重塑为 [B_, num_heads, N, head_dim]
        q_real = q_real.view(B_, N, self.num_heads, self.qk_head_dim).permute(0, 2, 1, 3)
        q_imag = q_imag.view(B_, N, self.num_heads, self.qk_head_dim).permute(0, 2, 1, 3)
        k_real = k_real.view(B_, N, self.num_heads, self.qk_head_dim).permute(0, 2, 1, 3)
        k_imag = k_imag.view(B_, N, self.num_heads, self.qk_head_dim).permute(0, 2, 1, 3)
        v = v.view(B_, N, self.num_heads, self.v_head_dim).permute(0, 2, 1, 3)

        # 计算密度矩阵
        rho_q_real, rho_q_imag = self.density(q_real, q_imag)
        rho_k_real, rho_k_imag = self.density(k_real, k_imag)

        # 酉演化
        rho_q_real, rho_q_imag = self.apply_unitary(rho_q_real, rho_q_imag)
        rho_k_real, rho_k_imag = self.apply_unitary(rho_k_real, rho_k_imag)

        # 计算迹项
        trace_term = torch.einsum('bhnii,bhmii->bhnm', rho_q_real, rho_k_real) + \
                     torch.einsum('bhnii,bhmii->bhnm', rho_q_imag, rho_k_imag)

        # 量子位置编码
        pos_real = F.normalize(self.pos_real, dim = -1)
        pos_imag = F.normalize(self.pos_imag, dim = -1)
        pos_sim = torch.einsum('ni,mi->nm', pos_real, pos_real) + \
                  torch.einsum('ni,mi->nm', pos_imag, pos_imag) - \
                  torch.einsum('ni,mi->nm', pos_real, pos_imag)

        # 合成注意力
        attn = trace_term * self.scale + pos_sim.unsqueeze(0).unsqueeze(0)

        # 掩码处理
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = F.softmax(attn, dim = -1)
        attn = self.attn_drop(attn)

        # 值聚合
        x = torch.einsum('bhnm,bhmi->bhni', attn, v)
        x = x.permute(0, 2, 1, 3).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class QuantumSwinTransformerBlock(nn.Module):
    r""" Demi Lenses Swin Transformer Block.

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
        """"
        self.attn = WindowAttention(
            dim, window_size = to_2tuple(self.window_size), num_heads = num_heads,
            qkv_bias = qkv_bias, attn_drop = attn_drop, proj_drop = drop,
            pretrained_window_size = to_2tuple(pretrained_window_size))
        """
        self.attn = QuantumAttention(
            dim, window_size = to_2tuple(self.window_size), num_heads = num_heads,
            qkv_bias = qkv_bias, attn_drop = attn_drop, proj_drop = drop)

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


class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size = 3, padding = 1, bias = False)

    def forward(self, x):
        avg_pool = x.mean(dim = 1, keepdim = True)
        max_pool = x.max(dim = 1, keepdim = True).values
        concat = torch.cat([avg_pool, max_pool], dim = 1)
        attend = self.conv(concat).sigmoid()
        x = x * attend
        return x


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


class SwinSpatialAttention(nn.Module):
    def __init__(self, in_channels, H, W):
        super().__init__()
        self.sa = SpatialAttention(in_channels)
        self.H = H
        self.W = W

    def forward(self, x):
        B, L, C = x.size()
        x = x.view(B, self.H, self.W, C).permute(0, 3, 1, 2)
        x = self.sa(x)
        x = x.permute(0, 2, 3, 1).view(B, L, C)
        return x


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
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        pretrained_window_size (int): Local window size in pretraining.
        add_spatial_attention (bool): If True, add spatial attention layer after each block. Default: False.
        add_channel_attention (bool): If True, add channel attention layer after each block. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio = 4., qkv_bias = True, drop = 0., attn_drop = 0.,
                 drop_path = 0., norm_layer = nn.LayerNorm, downsample = None, use_checkpoint = False,
                 pretrained_window_size = 0, add_spatial_attention = False, add_channel_attention = False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            QuantumSwinTransformerBlock(dim = dim, input_resolution = input_resolution,
                                        num_heads = num_heads, window_size = window_size,
                                        shift_size = 0 if (i % 2 == 0) else window_size // 2,
                                        mlp_ratio = mlp_ratio,
                                        qkv_bias = qkv_bias,
                                        drop = drop, attn_drop = attn_drop,
                                        drop_path = drop_path[i] if isinstance(drop_path, list) else drop_path,
                                        norm_layer = norm_layer,
                                        pretrained_window_size = pretrained_window_size)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim = dim, norm_layer = norm_layer)
        else:
            self.downsample = None

        # add spatial attention layer
        if add_spatial_attention:
            H, W = self.input_resolution
            self.sa = SwinSpatialAttention(dim, H, W)
        else:
            self.sa = None

        self.saw = nn.Parameter(torch.ones(1))

        # add channel attention layer
        if add_channel_attention:
            H, W = input_resolution
            self.ca = ChannelAttention(dim, H, W, reduction = 16)
        else:
            self.ca = None

        self.caw = nn.Parameter(torch.ones(1))

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)

        if self.sa is not None and self.ca is not None:
            sa_out = self.sa(x)
            ca_out = self.ca(x)
            x = self.saw * sa_out + self.caw * ca_out
        elif self.sa is not None:
            x = self.sa(x)
        elif self.ca is not None:
            x = self.ca(x)

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

    def _init_respostnorm(self):
        for blk in self.blocks:
            nn.init.constant_(blk.norm1.bias, 0)
            nn.init.constant_(blk.norm1.weight, 0)
            nn.init.constant_(blk.norm2.bias, 0)
            nn.init.constant_(blk.norm2.weight, 0)


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


class QuantumSwinTransformer(nn.Module):
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
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        pretrained_window_sizes (tuple(int)): Pretrained window sizes of each layer.
        add_spatial_attention (bool): If True, add spatial attention layer after each block. Default: False.
    """

    def __init__(self, img_size = 224, patch_size = 4, in_chans = 3, num_classes = 1000,
                 embed_dim = 96, depths = [2, 2, 6, 2], num_heads = [3, 6, 12, 24],
                 window_size = 7, mlp_ratio = 4., qkv_bias = True,
                 drop_rate = 0., attn_drop_rate = 0., drop_path_rate = 0.1,
                 norm_layer = nn.LayerNorm, ape = False, patch_norm = True,
                 use_checkpoint = False, pretrained_window_sizes = [0, 0, 0, 0],
                 add_spatial_attention = True, **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbedConv(
            img_size = img_size, patch_size = patch_size, in_chans = in_chans, embed_dim = embed_dim,
            norm_layer = norm_layer if self.patch_norm else None)
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
                               drop = drop_rate, attn_drop = attn_drop_rate,
                               drop_path = dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer = norm_layer,
                               downsample = PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint = use_checkpoint,
                               pretrained_window_size = pretrained_window_sizes[i_layer],
                               add_spatial_attention = add_sa,
                               add_channel_attention = add_ca)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)
        for bly in self.layers:
            bly._init_respostnorm()

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
