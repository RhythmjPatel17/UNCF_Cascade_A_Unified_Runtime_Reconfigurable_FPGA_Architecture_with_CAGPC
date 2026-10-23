import torch
import torch.nn as nn
import torch.nn.functional as F

def gather_tokens(x, idx):
    B, H = x.shape[0], x.shape[1]
    b_ix = torch.arange(B, device=x.device).view(B, 1, 1, 1)
    h_ix = torch.arange(H, device=x.device).view(1, H, 1, 1)
    return x[b_ix, h_ix, idx]


class DecoupledSpatialTemporalAttention(nn.Module):

    def __init__(self, dim, n_heads, n_tok, win_w, topk_k, layer_id, refresh_interval=1, qkv_bias=True):
        super().__init__()
        assert dim % n_heads == 0
        assert n_tok % win_w == 0
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.n_tok = n_tok
        self.win_w = win_w
        self.topk_k = topk_k
        self.layer_id = layer_id
        self.refresh_interval = max(1, refresh_interval)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.register_buffer('_cached_idx', None, persistent=False)
        self._current_step = 0

    def set_step(self, step_idx):
        self._current_step = step_idx

    def is_refresh_step(self):
        return self._current_step % self.refresh_interval == 0

    def reset_cache(self):
        self._cached_idx = None

    def _spatial(self, q, k, v):
        B, H, N, Dh = q.shape
        W = self.win_w
        qw = q.view(B, H, N // W, W, Dh)
        kw = k.view(B, H, N // W, W, Dh)
        vw = v.view(B, H, N // W, W, Dh)
        scores = torch.einsum('bhwqd,bhwkd->bhwqk', qw, kw) / Dh ** 0.5
        attn = F.softmax(scores, dim=-1)
        out = torch.einsum('bhwqk,bhwkd->bhwqd', attn, vw)
        return out.reshape(B, H, N, Dh)

    def _temporal_refresh(self, q, k, v):
        B, H, N, Dh = q.shape
        scores = torch.einsum('bhqd,bhkd->bhqk', q, k) / Dh ** 0.5
        topk_val, topk_idx = torch.topk(scores, self.topk_k, dim=-1)
        attn = F.softmax(topk_val, dim=-1)
        v_gathered = gather_tokens(v, topk_idx)
        out = torch.einsum('bhqk,bhqkd->bhqd', attn, v_gathered)
        self._cached_idx = topk_idx.detach()
        return out

    def _temporal_reuse(self, q, k, v):
        assert self._cached_idx is not None, f'layer {self.layer_id}: reuse step called before any refresh step has populated the candidate cache'
        B, H, N, Dh = q.shape
        idx = self._cached_idx
        if idx.shape[0] != B:
            idx = idx.expand(B, -1, -1, -1)
        k_gathered = gather_tokens(k, idx)
        v_gathered = gather_tokens(v, idx)
        scores = torch.einsum('bhqd,bhqkd->bhqk', q, k_gathered) / Dh ** 0.5
        attn = F.softmax(scores, dim=-1)
        out = torch.einsum('bhqk,bhqkd->bhqd', attn, v_gathered)
        return out

    def forward(self, x):
        B, N, C = x.shape
        assert N == self.n_tok, f'expected n_tok={self.n_tok}, got {N}'
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = (qkv[0], qkv[1], qkv[2])
        spatial_out = self._spatial(q, k, v)
        if self.is_refresh_step():
            temporal_out = self._temporal_refresh(q, k, v)
        else:
            temporal_out = self._temporal_reuse(q, k, v)
        out = spatial_out + temporal_out
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.proj(out)

class TridentStepScheduler:

    def __init__(self, model):
        self.layers = [m for m in model.modules() if isinstance(m, DecoupledSpatialTemporalAttention)]

    def set_step(self, step_idx):
        for layer in self.layers:
            layer.set_step(step_idx)

    def reset(self):
        for layer in self.layers:
            layer.reset_cache()
            layer.set_step(0)
