import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, json, time, math, copy, os, sys, io
import pyarrow.parquet as pq
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
T_STEPS = 50
N_STEPS = 400
BATCH = 32
EXP_DIR = '/home/claude/experiments'
os.makedirs(EXP_DIR, exist_ok=True)
PARQUET_PATH = '/mnt/user-data/uploads/1785324957611_train-00000-of-00009-5beebd96eb33b02b.parquet'
N_IMAGES = 6000

def load_lsun_tensor(n_images=N_IMAGES):
    pf = pq.ParquetFile(PARQUET_PATH)
    arrs = []
    count = 0
    for batch in pf.iter_batches(batch_size=500, columns=['image']):
        d = batch.to_pydict()
        for im in d['image']:
            pil = Image.open(io.BytesIO(im['bytes'])).convert('RGB').resize((IMG, IMG), Image.BILINEAR)
            arrs.append(np.asarray(pil, dtype=np.float32) / 255.0)
            count += 1
            if count >= n_images:
                break
        if count >= n_images:
            break
    arr = np.stack(arrs, axis=0)
    t = torch.from_numpy(arr).permute(0, 3, 1, 2)
    t = t * 2 - 1
    return t
print('Loading real LSUN images from parquet shard (resized to 8x8)...')
t0 = time.time()
LSUN_TENSOR = load_lsun_tensor()
print(f'Loaded {LSUN_TENSOR.shape[0]} real LSUN images in {time.time() - t0:.1f}s')

def make_batch_lsun(n):
    idx = np.random.randint(0, LSUN_TENSOR.shape[0], size=n)
    return LSUN_TENSOR[idx].clone()
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

class TransformerBlock(nn.Module):

    def __init__(self, d, heads, mlp_ratio):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, d * mlp_ratio), nn.GELU(), nn.Linear(d * mlp_ratio, d))

    def forward(self, x):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        x = x + self.ffn(self.ln2(x))
        return x

class DiT_S2(nn.Module):

    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Conv2d(CH, HIDDEN, kernel_size=PATCH, stride=PATCH)
        self.pos_embed = nn.Parameter(torch.randn(1, L, HIDDEN) * 0.02)
        self.time_mlp = nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.SiLU(), nn.Linear(HIDDEN, HIDDEN))
        self.blocks = nn.ModuleList([TransformerBlock(HIDDEN, HEADS, MLP_RATIO) for _ in range(DEPTH)])
        self.final_ln = nn.LayerNorm(HIDDEN)
        self.final_proj = nn.Linear(HIDDEN, PATCH * PATCH * CH)

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

def quantize_uniform(w, bits):
    qmax = 2 ** (bits - 1) - 1
    scale = w.abs().max() / qmax
    if scale == 0:
        return w.clone()
    q = torch.clamp(torch.round(w / scale), -qmax - 1, qmax)
    return q * scale

def quantize_ternary(w):
    thresh = 0.7 * w.abs().mean()
    scale = w.abs().mean()
    q = torch.zeros_like(w)
    q[w > thresh] = scale
    q[w < -thresh] = -scale
    return q

def make_quantized(model, mode):
    qm = copy.deepcopy(model)
    with torch.no_grad():
        for p in qm.parameters():
            if mode == 'int8':
                p.copy_(quantize_uniform(p, 8))
            elif mode == 'int4':
                p.copy_(quantize_uniform(p, 4))
            elif mode == 'ternary':
                p.copy_(quantize_ternary(p))
    return qm

def psnr(a, b):
    mse = F.mse_loss(a, b).item()
    if mse == 0:
        return float('inf')
    return 10 * math.log10(4.0 / mse)

def ssim_simple(a, b, C1=0.01 ** 2, C2=0.03 ** 2):
    a = (a + 1) / 2
    b = (b + 1) / 2
    mu_a, mu_b = (a.mean(), b.mean())
    var_a, var_b = (a.var(), b.var())
    cov = ((a - mu_a) * (b - mu_b)).mean()
    return ((2 * mu_a * mu_b + C1) * (2 * cov + C2) / ((mu_a ** 2 + mu_b ** 2 + C1) * (var_a + var_b + C2))).item()

@torch.no_grad()
def sample(model, n):
    x = torch.randn(n, CH, IMG, IMG)
    for ti in reversed(range(T_STEPS)):
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
name = 'lsun_real'
seed = 4
ckpt_path = os.path.join(EXP_DIR, f'ckpt_{name}.pt')
results_path = os.path.join(EXP_DIR, 'lsun_results.json')
steps_per_call = 100
model = DiT_S2()
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
    x0 = make_batch_lsun(BATCH)
    t = torch.randint(0, T_STEPS, (BATCH,))
    noise = torch.randn_like(x0)
    xt = q_sample(x0, t, noise)
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
x_fp32 = sample(model, N_SAMPLE)
ds_results = {}
for mode, mname in [('int8', 'INT8'), ('int4', 'INT4'), ('ternary', 'Ternary')]:
    qmodel = make_quantized(model, mode)
    qmodel.eval()
    torch.manual_seed(seed + 1000)
    x_q = sample(qmodel, N_SAMPLE)
    p = psnr(x_q, x_fp32)
    s = ssim_simple(x_q, x_fp32)
    with torch.no_grad():
        x0_eval = make_batch_lsun(64)
        t_eval = torch.randint(0, T_STEPS, (64,))
        noise_eval = torch.randn_like(x0_eval)
        xt_eval = q_sample(x0_eval, t_eval, noise_eval)
        eps_fp32 = model(xt_eval, t_eval)
        eps_q = qmodel(xt_eval, t_eval)
        noise_pred_rel_l2 = (torch.norm(eps_q - eps_fp32) / torch.norm(eps_fp32)).item() * 100
    ds_results[mname] = dict(psnr_db=p, ssim=s, noise_pred_rel_l2_pct=noise_pred_rel_l2)
    print(f'[{mname}] PSNR={p:.2f} dB  SSIM={s:.4f}  noise-pred rel L2={noise_pred_rel_l2:.3f}%')
ds_results['meta'] = dict(n_params=n_params, final_train_loss=float(losses[-1]), dataset='Real LSUN (6000 images, from HF parquet shard, resized ~256px->8x8)')
with open(results_path, 'w') as f:
    json.dump(ds_results, f, indent=2)
print('Saved ->', results_path)
