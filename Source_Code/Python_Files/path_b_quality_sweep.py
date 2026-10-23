import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, json, time, math, os, sys, glob
from PIL import Image
torch.manual_seed(0)
np.random.seed(0)
DEPTH = 12
HIDDEN = 384
HEADS = 6
PATCH = 2
MLP_RATIO = 4
IMG = 8
CH = 3
L = (IMG // PATCH) ** 2
WIN_W = 4
TOPK_K = 4
T_STEPS = 50
N_STEPS = 400
BATCH = 32
REFRESH_INTERVALS = [1, 2, 4, 8, 16]
EXP_DIR = '/home/claude/experiments'
os.makedirs(EXP_DIR, exist_ok=True)
CIFAR_ROOT = '/home/claude/cifar10_repo/CIFAR-10-images-master/train'

def load_cifar_tensor(max_per_class=600):
    classes = sorted(os.listdir(CIFAR_ROOT))
    imgs = []
    for c in classes:
        files = sorted(glob.glob(os.path.join(CIFAR_ROOT, c, '*.jpg')))[:max_per_class]
        for fp in files:
            im = Image.open(fp).convert('RGB').resize((IMG, IMG), Image.BILINEAR)
            arr = np.asarray(im, dtype=np.float32) / 255.0
            imgs.append(arr)
    arr = np.stack(imgs, axis=0)
    t = torch.from_numpy(arr).permute(0, 3, 1, 2)
    t = t * 2 - 1
    return t
print('Loading real CIFAR-10 images (resized to 8x8)...')
t0 = time.time()
CIFAR_TENSOR = load_cifar_tensor(max_per_class=600)
print(f'Loaded {CIFAR_TENSOR.shape[0]} real CIFAR-10 images in {time.time() - t0:.1f}s')

def make_batch_cifar(n):
    idx = np.random.randint(0, CIFAR_TENSOR.shape[0], size=n)
    return CIFAR_TENSOR[idx].clone()
betas = torch.linspace(0.0001, 0.02, T_STEPS)
alphas = 1.0 - betas
alpha_bar = torch.cumprod(alphas, dim=0)

def q_sample(x0, t, noise):
    ab = alpha_bar[t].view(-1, 1, 1, 1)
    return torch.sqrt(ab) * x0 + torch.sqrt(1 - ab) * noise

def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half).float() / half)
    args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

class TridentCascadeAttention(nn.Module):

    def __init__(self, dim, n_heads, n_tok, win_w, topk_k, layer_id):
        super().__init__()
        assert dim % n_heads == 0 and n_tok % win_w == 0
        self.n_heads, self.head_dim = (n_heads, dim // n_heads)
        self.n_tok, self.win_w, self.topk_k = (n_tok, win_w, topk_k)
        self.layer_id = layer_id
        self.refresh_interval = 1
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self._cached_idx = None
        self._step = 0

    def set_step(self, step_idx):
        self._step = step_idx

    def is_refresh_step(self):
        return self._step % max(1, self.refresh_interval) == 0

    def reset_cache(self):
        self._cached_idx = None

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = (qkv[0], qkv[1], qkv[2])
        Dh = self.head_dim
        W = self.win_w
        qw = q.view(B, self.n_heads, N // W, W, Dh)
        kw = k.view(B, self.n_heads, N // W, W, Dh)
        vw = v.view(B, self.n_heads, N // W, W, Dh)
        sscore = torch.einsum('bhwqd,bhwkd->bhwqk', qw, kw) / Dh ** 0.5
        sattn = F.softmax(sscore, dim=-1)
        sout = torch.einsum('bhwqk,bhwkd->bhwqd', sattn, vw).reshape(B, self.n_heads, N, Dh)
        if self.is_refresh_step() or self._cached_idx is None:
            tscore = torch.einsum('bhqd,bhkd->bhqk', q, k) / Dh ** 0.5
            tval, tidx = torch.topk(tscore, self.topk_k, dim=-1)
            tattn = F.softmax(tval, dim=-1)
            vg = torch.gather(v.unsqueeze(2).expand(B, self.n_heads, N, N, Dh), 3, tidx.unsqueeze(-1).expand(B, self.n_heads, N, self.topk_k, Dh))
            tout = torch.einsum('bhqk,bhqkd->bhqd', tattn, vg)
            self._cached_idx = tidx.detach()
        else:
            idx = self._cached_idx
            kg = torch.gather(k.unsqueeze(2).expand(B, self.n_heads, N, N, Dh), 3, idx.unsqueeze(-1).expand(B, self.n_heads, N, self.topk_k, Dh))
            vg = torch.gather(v.unsqueeze(2).expand(B, self.n_heads, N, N, Dh), 3, idx.unsqueeze(-1).expand(B, self.n_heads, N, self.topk_k, Dh))
            tscore = torch.einsum('bhqd,bhqkd->bhqk', q, kg) / Dh ** 0.5
            tattn = F.softmax(tscore, dim=-1)
            tout = torch.einsum('bhqk,bhqkd->bhqd', tattn, vg)
        out = (sout + tout).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)

class TridentBlock(nn.Module):

    def __init__(self, d, heads, n_tok, win_w, topk_k, layer_id, mlp_ratio):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = TridentCascadeAttention(d, heads, n_tok, win_w, topk_k, layer_id)
        self.ln2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, d * mlp_ratio), nn.GELU(), nn.Linear(d * mlp_ratio, d))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class DiT_S2_Trident(nn.Module):

    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Conv2d(CH, HIDDEN, kernel_size=PATCH, stride=PATCH)
        self.pos_embed = nn.Parameter(torch.randn(1, L, HIDDEN) * 0.02)
        self.time_mlp = nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.SiLU(), nn.Linear(HIDDEN, HIDDEN))
        self.blocks = nn.ModuleList([TridentBlock(HIDDEN, HEADS, L, WIN_W, TOPK_K, layer_id=l, mlp_ratio=MLP_RATIO) for l in range(DEPTH)])
        self.final_ln = nn.LayerNorm(HIDDEN)
        self.final_proj = nn.Linear(HIDDEN, PATCH * PATCH * CH)

    def set_refresh_interval(self, r):
        for blk in self.blocks:
            blk.attn.refresh_interval = max(1, r)

    def reset_cache(self):
        for blk in self.blocks:
            blk.attn.reset_cache()

    def set_step(self, t):
        for blk in self.blocks:
            blk.attn.set_step(t)

    def forward(self, x, t):
        B = x.shape[0]
        h = self.patch_embed(x).flatten(2).transpose(1, 2)
        h = h + self.pos_embed
        temb = self.time_mlp(timestep_embedding(t, HIDDEN))
        h = h + temb.unsqueeze(1)
        for blk in self.blocks:
            h = blk(h)
        h = self.final_ln(h)
        out = self.final_proj(h)
        gs = IMG // PATCH
        out = out.transpose(1, 2).reshape(B, CH, PATCH, PATCH, gs, gs)
        out = out.permute(0, 1, 4, 2, 5, 3).reshape(B, CH, IMG, IMG)
        return out

def psnr(a, b):
    mse = F.mse_loss(a, b).item()
    return float('inf') if mse == 0 else 10 * math.log10(4.0 / mse)

def ssim_simple(a, b, C1=0.01 ** 2, C2=0.03 ** 2):
    a = (a + 1) / 2
    b = (b + 1) / 2
    mu_a, mu_b = (a.mean(), b.mean())
    var_a, var_b = (a.var(), b.var())
    cov = ((a - mu_a) * (b - mu_b)).mean()
    return ((2 * mu_a * mu_b + C1) * (2 * cov + C2) / ((mu_a ** 2 + mu_b ** 2 + C1) * (var_a + var_b + C2))).item()

@torch.no_grad()
def sample(model, n, refresh_interval):
    model.set_refresh_interval(refresh_interval)
    model.reset_cache()
    x = torch.randn(n, CH, IMG, IMG)
    for ti in reversed(range(T_STEPS)):
        model.set_step(T_STEPS - 1 - ti)
        t = torch.full((n,), ti, dtype=torch.long)
        eps_pred = model(x, t)
        a = alphas[ti]
        ab = alpha_bar[ti]
        beta = betas[ti]
        mean = 1 / torch.sqrt(a) * (x - beta / torch.sqrt(1 - ab) * eps_pred)
        if ti > 0:
            x = mean + torch.sqrt(beta) * torch.randn_like(x)
        else:
            x = mean
    return x.clamp(-1, 1)
name = 'path_b_cifar10'
seed = 3
ckpt_path = os.path.join(EXP_DIR, f'ckpt_{name}.pt')
results_path = os.path.join(EXP_DIR, 'path_b_results.json')
steps_per_call = 180
model = DiT_S2_Trident()
model.set_refresh_interval(1)
opt = torch.optim.Adam(model.parameters(), lr=0.0003)
start_step = 0
losses = []
if os.path.exists(ckpt_path):
    ck = torch.load(ckpt_path, weights_only=False)
    model.load_state_dict(ck['model'])
    opt.load_state_dict(ck['opt'])
    start_step = ck['step']
    losses = ck['losses']
    torch.manual_seed(seed + start_step)
    print(f'resumed from step {start_step}')
else:
    torch.manual_seed(seed)
n_params = sum((p.numel() for p in model.parameters()))
t0 = time.time()
end_step = min(start_step + steps_per_call, N_STEPS)
for step in range(start_step, end_step):
    x0 = make_batch_cifar(BATCH)
    t = torch.randint(0, T_STEPS, (BATCH,))
    noise = torch.randn_like(x0)
    xt = q_sample(x0, t, noise)
    model.set_step(0)
    model.reset_cache()
    pred_noise = model(xt, t)
    loss = F.mse_loss(pred_noise, noise)
    opt.zero_grad()
    loss.backward()
    opt.step()
    losses.append(loss.item())
    if step % 50 == 0:
        print(f'step {step:4d}/{N_STEPS}  mse {loss.item():.4f}  elapsed {time.time() - t0:.1f}s')
print(f'chunk training time: {time.time() - t0:.1f}s, now at step {end_step}/{N_STEPS}')
torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'step': end_step, 'losses': losses}, ckpt_path)
if end_step < N_STEPS:
    print(f'NOT DONE -- rerun to continue from step {end_step}')
    sys.exit(0)
print('TRAINING COMPLETE, final loss', losses[-1])
model.eval()
torch.manual_seed(seed + 1000)
N_SAMPLE = 32
x_r1 = sample(model, N_SAMPLE, refresh_interval=1)
sweep_results = {}
for r in REFRESH_INTERVALS:
    torch.manual_seed(seed + 1000)
    x_r = sample(model, N_SAMPLE, refresh_interval=r)
    if r == 1:
        p, s = (float('inf'), 1.0)
        noise_pred_rel_l2 = 0.0
    else:
        p = psnr(x_r, x_r1)
        s = ssim_simple(x_r, x_r1)
        with torch.no_grad():
            x0_eval = make_batch_cifar(64)
            t_eval = torch.randint(0, T_STEPS, (64,))
            noise_eval = torch.randn_like(x0_eval)
            xt_eval = q_sample(x0_eval, t_eval, noise_eval)
            model.set_refresh_interval(1)
            model.reset_cache()
            model.set_step(0)
            eps_r1 = model(xt_eval, t_eval)
            model.set_refresh_interval(r)
            model.reset_cache()
            model.set_step(0)
            eps_r = model(xt_eval, t_eval)
            noise_pred_rel_l2 = (torch.norm(eps_r - eps_r1) / torch.norm(eps_r1)).item() * 100
    sweep_results[f'R={r}'] = dict(psnr_db=p, ssim=s, noise_pred_rel_l2_pct=noise_pred_rel_l2)
    print(f'[R={r:2d}] PSNR={p:.2f} dB  SSIM={s:.4f}  noise-pred rel L2={noise_pred_rel_l2:.3f}%  (vs R=1 reference)')
sweep_results['meta'] = dict(n_params=n_params, final_train_loss=float(losses[-1]), dataset='Real CIFAR-10 (6000 images, 600/class, resized 32x32->8x8)', architecture='DiT-S/2 with TridentCascadeAttention (WIN_W=4, TOPK_K=4, L=16)', note="R=1 is the exact/unapproximated reference; all other rows are PSNR/SSIM/noise-pred-L2 of that R's samples against R=1's samples (same comparison methodology as Table IX's quantization sweep, applied to the candidate-reuse approximation instead).")
with open(results_path, 'w') as f:
    json.dump(sweep_results, f, indent=2)
print('Saved ->', results_path)
