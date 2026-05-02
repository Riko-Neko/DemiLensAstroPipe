import torch
import torch.nn as nn
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


# Mlp class (unchanged)
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


# DyT class (unchanged)
class DyT(nn.Module):
    def __init__(self, dim, init_alpha = 0.5):
        super(DyT, self).__init__()
        self.alpha = nn.Parameter(torch.ones(1) * init_alpha)
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        return self.gamma * x + self.beta


# conv_relu_bn helper function (unchanged)
def conv_relu_bn(in_channel, out_channel, dirate):
    return nn.Sequential(
        nn.Conv2d(in_channels = in_channel, out_channels = out_channel, kernel_size = 3, stride = 1, padding = dirate,
                  dilation = dirate),
        nn.BatchNorm2d(out_channel),
        nn.ReLU(inplace = True)
    )


# Conv class (unchanged)
class Conv(nn.Module):
    def __init__(self, in_dim):
        super(Conv, self).__init__()
        self.convs = nn.ModuleList([conv_relu_bn(in_dim, in_dim, 1) for _ in range(3)])

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        return x


# DConv class (unchanged)
class DConv(nn.Module):
    def __init__(self, in_dim):
        super(DConv, self).__init__()
        dilation = [2, 4, 2]
        self.dconvs = nn.ModuleList([conv_relu_bn(in_dim, in_dim, dirate) for dirate in dilation])

    def forward(self, x):
        for dconv in self.dconvs:
            x = dconv(x)
        return x


# Window partitioning functions (unchanged)
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


# Corrected Attention class
class Attention(nn.Module):
    def __init__(self, in_dim, in_feature, out_feature, window_size):
        super(Attention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels = in_dim, out_channels = 1, kernel_size = 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim, out_channels = 1, kernel_size = 1)
        self.query_line = nn.Linear(in_features = in_feature, out_features = out_feature)
        self.key_line = nn.Linear(in_features = in_feature, out_features = out_feature)
        self.s_conv = nn.Conv2d(in_channels = 1, out_channels = in_dim, kernel_size = 1)
        self.softmax = nn.Softmax(dim = -1)

        # Relative position bias for rows and columns
        self.relative_position_bias_row = nn.Parameter(torch.zeros(2 * window_size - 1))
        self.relative_position_bias_col = nn.Parameter(torch.zeros(2 * window_size - 1))
        self.window_size = window_size

    def forward(self, x, mask = None):
        B, C, H, W = x.shape
        assert H == W == self.window_size, "Input size must match window size"

        # Calculate query and key
        q = rearrange(self.query_line(rearrange(self.query_conv(x), 'b 1 h w -> b (h w)')), 'b h -> b h 1')  # (B, H, 1)
        k = rearrange(self.key_line(rearrange(self.key_conv(x), 'b 1 h w -> b (h w)')), 'b h -> b 1 h')  # (B, 1, H)
        att = rearrange(torch.matmul(q, k), 'b h w -> b 1 h w')  # (B, 1, H, H)

        # Compute relative position bias
        relative_position_bias = self.compute_relative_position_bias(H)

        # Add relative position bias to attention
        att = att + relative_position_bias

        # Apply mask if provided
        if mask is not None:
            att = att + mask

        # Apply s_conv and softmax
        att = self.s_conv(att)  # (B, C, H, H)
        att = self.softmax(att)
        return att

    def compute_relative_position_bias(self, window_size):
        # Compute relative position indices
        relative_position_index = torch.arange(window_size).view(-1, 1) - torch.arange(window_size).view(1, -1)
        relative_position_index_row = relative_position_index + window_size - 1
        relative_position_index_col = relative_position_index.T + window_size - 1

        # Get biases
        bias_row = self.relative_position_bias_row[relative_position_index_row]
        bias_col = self.relative_position_bias_col[relative_position_index_col]

        # Combine row and column biases
        relative_position_bias = bias_row + bias_col
        return relative_position_bias.unsqueeze(0).unsqueeze(0)  # (1, 1, H, H)


# ConvAttention class (unchanged except for using corrected Attention)
class ConvAttention(nn.Module):
    def __init__(self, in_dim, in_feature, out_feature, window_size):
        super(ConvAttention, self).__init__()
        self.conv = Conv(in_dim)
        self.dconv = DConv(in_dim)
        self.att = Attention(in_dim, in_feature, out_feature, window_size)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, mask = None):
        q = self.conv(x)
        k = self.dconv(x)
        v = q + k
        att = self.att(x, mask = mask)
        out = torch.matmul(att, v)
        return self.gamma * out + v + x


# WindowConvAttention class (unchanged except for using corrected ConvAttention)
class WindowConvAttention(nn.Module):
    def __init__(self, dim, window_size):
        super(WindowConvAttention, self).__init__()
        self.window_size = window_size
        self.conv_attention = ConvAttention(
            in_dim = dim,
            in_feature = window_size * window_size,
            out_feature = window_size,
            window_size = window_size
        )

    def forward(self, x, mask = None):
        B_, N, C = x.shape
        window_size = self.window_size
        assert N == window_size * window_size, "Input size mismatch"

        x = x.view(B_, window_size, window_size, C).permute(0, 3, 1, 2)
        if mask is not None:
            batch_size = B_ // mask.shape[0]  # calculate batch_size
            # extend mask to match the shape of windows
            mask = mask.repeat(batch_size, 1, 1, 1)  # reshape to (B_, 1, window_size, window_size)

        x = self.conv_attention(x, mask = mask)
        x = x.permute(0, 2, 3, 1).contiguous().view(B_, N, C)
        return x


# CLFTSwinBlock class (unchanged except for using corrected WindowConvAttention)
class CLFTSwinBlock(nn.Module):
    r""" CLFT Swin Transformer Block.

        Args:
            dim (int): Number of input channels.
            input_resolution (tuple[int]): Input resulotion.
            num_heads (int): Number of attention heads.
            window_size (int): Window size.
            shift_size (int): Shift size for SW-MSA.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            drop (float, optional): Dropout rate. Default: 0.0
            drop_path (float, optional): Stochastic depth rate. Default: 0.0
            act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
            norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        """

    def __init__(self, dim, input_resolution, num_heads, window_size = 7, shift_size = 0, mlp_ratio = 4., drop = 0.,
                 drop_path = 0., act_layer = nn.GELU, norm_layer = nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowConvAttention(dim, window_size = window_size)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features = dim, hidden_features = mlp_hidden_dim, act_layer = act_layer, drop = drop)

        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
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
            mask_windows = window_partition(img_mask, self.window_size)  # (nW, window_size, window_size, 1)
            mask_windows = mask_windows.squeeze(-1)  # (nW, window_size, window_size)

            attn_mask = (mask_windows[:, None, :, :] != mask_windows[:, :, None,
                                                        :]).float()  # (nW, window_size, window_size, window_size)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
            attn_mask = attn_mask[:, 0, :, :].unsqueeze(
                1)  # take the first head and reshape to (nW, 1, window_size, window_size)
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = x.view(B, H, W, C)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts = (-self.shift_size, -self.shift_size), dims = (1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        attn_windows = self.attn(x_windows, mask = self.attn_mask)

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts = (self.shift_size, self.shift_size), dims = (1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        x = shortcut + self.drop_path(self.norm1(x))
        x = x + self.drop_path(self.norm2(self.mlp(x)))
        return x


# PatchMerging class (unchanged)
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
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)

        x = self.reduction(x)
        x = self.norm(x)
        return x


# BasicLayer class (unchanged)
class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

        Args:
            dim (int): Number of input channels.
            input_resolution (tuple[int]): Input resolution.
            depth (int): Number of blocks.
            num_heads (int): Number of attention heads.
            window_size (int): Local window size.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            drop (float, optional): Dropout rate. Default: 0.0
            drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
            norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
            downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size, mlp_ratio = 4., drop = 0., drop_path = 0.,
                 norm_layer = nn.LayerNorm, downsample = None):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        self.blocks = nn.ModuleList([
            CLFTSwinBlock(dim = dim, input_resolution = input_resolution,
                          num_heads = num_heads, window_size = window_size,
                          shift_size = 0 if (i % 2 == 0) else window_size // 2,
                          mlp_ratio = mlp_ratio, drop = drop,
                          drop_path = drop_path[i] if isinstance(drop_path, list) else drop_path,
                          norm_layer = norm_layer)
            for i in range(depth)])

        self.downsample = downsample(input_resolution, dim = dim, norm_layer = norm_layer) if downsample else None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


# PatchEmbed class (unchanged)
class PatchEmbed(nn.Module):
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

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size = patch_size, stride = patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else None

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x


# CLFTSwinTransformer class (unchanged)
class CLFTSwinTransformer(nn.Module):
    r""" Swin Transformer with CLFT Attention.
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
            drop_rate (float): Dropout rate. Default: 0
            drop_path_rate (float): Stochastic depth rate. Default: 0.1
            ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
            patch_norm (bool): If True, add normalization after patch embedding. Default: True
        """

    def __init__(self, img_size = 144, patch_size = 4, in_chans = 3, num_classes = 1, embed_dim = 96,
                 depths = [2, 2, 6, 2], num_heads = [3, 6, 12, 24], window_size = 6, mlp_ratio = 4., drop_rate = 0.,
                 drop_path_rate = 0.1, ape = False, patch_norm = True, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        self.patch_embed = PatchEmbed(
            img_size = img_size, patch_size = patch_size, in_chans = in_chans, embed_dim = embed_dim,
            norm_layer = nn.LayerNorm if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std = .02)

        self.pos_drop = nn.Dropout(p = drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim = int(embed_dim * 2 ** i_layer),
                               input_resolution = (patches_resolution[0] // (2 ** i_layer),
                                                   patches_resolution[1] // (2 ** i_layer)),
                               depth = depths[i_layer],
                               num_heads = num_heads[i_layer],
                               window_size = window_size,
                               mlp_ratio = self.mlp_ratio,
                               drop = drop_rate,
                               drop_path = dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer = DyT,
                               downsample = PatchMerging if (i_layer < self.num_layers - 1) else None)
            self.layers.append(layer)

        self.norm = nn.LayerNorm(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std = .02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        x = self.avgpool(x.transpose(1, 2))
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x
