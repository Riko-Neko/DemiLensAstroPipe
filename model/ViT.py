import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    """
    Splits image into patches and projects to embedding dimension.
    """
    def __init__(self, img_size=96, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        assert img_size % patch_size == 0, "Image size must be divisible by patch size"
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):  # x: [B, C, H, W]
        x = self.proj(x)  # [B, embed_dim, H/patch, W/patch]
        x = x.flatten(2)  # [B, embed_dim, num_patches]
        x = x.transpose(1, 2)  # [B, num_patches, embed_dim]
        return x

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):  # x: [B, N, embed_dim]
        x2 = self.norm1(x)
        attn_out, _ = self.attn(x2, x2, x2)
        x = x + self.dropout(attn_out)
        x2 = self.norm2(x)
        mlp_out = self.mlp(x2)
        x = x + self.dropout(mlp_out)
        return x

class ViT(nn.Module):
    """
    Vision Transformer for binary classification.
    Input: x of shape [B, 3, 96, 96]
    Output: logits of shape [B, 1], suitable for BCEWithLogitsLoss.
    """
    def __init__(
        self,
        img_size=96,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        dropout=0.0,
        attn_dropout=0.0,
    ):
        super().__init__()
        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches

        # Class token + positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_dropout = nn.Dropout(p=dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, mlp_ratio, dropout=attn_dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Classification head: single logit
        self.head = nn.Linear(embed_dim, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)  # [B, num_patches, embed_dim]
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B,1,embed_dim]
        x = torch.cat((cls_tokens, x), dim=1)  # [B,1+num_patches,embed_dim]
        x = x + self.pos_embed
        x = self.pos_dropout(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        cls_out = x[:, 0]  # [B, embed_dim]
        logits = self.head(cls_out)  # [B,1]
        return logits

# Sanity check
if __name__ == "__main__":
    model = ViT()
    dummy = torch.randn(4, 3, 96, 96)
    out = model(dummy)
    print(out.shape)  # Expected [4,1]
