import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, RandomCrop, RandomHorizontalFlip, Compose
import torch.nn as nn


# Vision Transformer Components
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=192):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: [B, C, H, W]
        x = self.proj(x)  # [B, embed_dim, H/patch_size, W/patch_size]
        x = x.flatten(2)  # [B, embed_dim, n_patches]
        x = x.transpose(1, 2)  # [B, n_patches, embed_dim]
        return x

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x


class TransformerBlockModified(nn.Module):
    def __init__(self, embed_dim, k_dim, v_dim, seq_len, mlp_ratio=4, dropout=0.1, mask=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = ModifiedSelectiveAttentionModule(
            embed_dim, k_dim, v_dim, seq_len, mask
        )
        self.norm2 = nn.LayerNorm(embed_dim)

        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = MLP(embed_dim, mlp_hidden_dim, embed_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output = self.attn(self.norm1(x))
        x = x + self.dropout(attn_output)
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x

class ViTWithModifiedSelectiveAttention(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, num_classes=10,
                 embed_dim=192, depth=9, k_dim=64, v_dim=64, mlp_ratio=4,
                 dropout=0.1, pool='cls'):
        super().__init__()

        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = (img_size // patch_size) ** 2
        self.seq_len = num_patches + 1  # For class token

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.seq_len, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.blocks = nn.ModuleList([
            TransformerBlockModified(
                embed_dim, k_dim, v_dim,
                seq_len=self.seq_len,  
                mlp_ratio=mlp_ratio,
                dropout=dropout
            ) for _ in range(depth)
        ])

    
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self.pool = pool

    def forward(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        x = x[:, 0] if self.pool == 'cls' else x.mean(dim=1)
        return self.head(x)