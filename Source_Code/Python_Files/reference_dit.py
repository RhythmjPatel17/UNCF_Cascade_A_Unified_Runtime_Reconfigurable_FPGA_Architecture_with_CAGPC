import torch
import torch.nn as nn
from .attention import DecoupledSpatialTemporalAttention

class MLP(nn.Module):

    def __init__(self, dim, mlp_ratio=4):
        super().__init__()
        hidden = dim * mlp_ratio
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

class DiTBlock(nn.Module):

    def __init__(self, dim, n_heads, n_tok, win_w, topk_k, layer_id, refresh_interval, mlp_ratio=4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = DecoupledSpatialTemporalAttention(dim, n_heads, n_tok, win_w, topk_k, layer_id, refresh_interval)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class ReferenceDiT(nn.Module):

    def __init__(self, n_tok=2304, hid_d=384, n_heads=6, win_w=4, topk_k=32, depth=12, mlp_ratio=4, refresh_interval=1, patch_in_dim=12):
        super().__init__()
        self.n_tok = n_tok
        self.patch_embed = nn.Linear(patch_in_dim, hid_d)
        self.blocks = nn.ModuleList([DiTBlock(hid_d, n_heads, n_tok, win_w, topk_k, layer_id=l, refresh_interval=refresh_interval, mlp_ratio=mlp_ratio) for l in range(depth)])
        self.norm_out = nn.LayerNorm(hid_d)
        self.head = nn.Linear(hid_d, patch_in_dim)

    def forward(self, patches):
        x = self.patch_embed(patches)
        for block in self.blocks:
            x = block(x)
        x = self.norm_out(x)
        return self.head(x)

    def set_refresh_interval(self, r):
        for block in self.blocks:
            block.attn.refresh_interval = max(1, r)
