import glob
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy import linalg

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trident_sim.attention import DecoupledSpatialTemporalAttention

CFG = dict(
    IMG=32,
    CH=3,
    PATCH=2,
    HIDDEN=int(os.environ.get("HIDDEN", 128)),
    HEADS=int(os.environ.get("HEADS", 4)),
    DEPTH=int(os.environ.get("DEPTH", 4)),
    MLP_RATIO=4,
    WIN_W=4,
    TOPK_K=int(os.environ.get("TOPK_K", 16)),
    T_STEPS=50,
    N_TRAIN_STEPS=int(os.environ.get("N_TRAIN_STEPS", 2000)),
    BATCH=int(os.environ.get("BATCH", 16)),
    LR=3e-4,
    N_TRAIN_IMGS=int(os.environ.get("N_TRAIN_IMGS", 10000)),
    N_EVAL=int(os.environ.get("N_EVAL", 256)),
    REFRESH_INTERVALS=[int(x) for x in os.environ.get("RS", "1,4,8,16").split(",")],
    TIME_BUDGET=float(os.environ.get("TIME_BUDGET", 1e9)),
)
IMG, CH, PATCH = CFG["IMG"], CFG["CH"], CFG["PATCH"]
HIDDEN, HEADS, DEPTH = CFG["HIDDEN"], CFG["HEADS"], CFG["DEPTH"]
L = (IMG // PATCH) ** 2
T_STEPS = CFG["T_STEPS"]

ROOT = "/home/claude/cifar10_repo/CIFAR-10-images-master"
EXP_DIR = os.environ.get("EXP_DIR", "/home/claude/experiments_fid")
os.makedirs(EXP_DIR, exist_ok=True)
FID_WEIGHTS = "/tmp/pt_inception.pth"

torch.set_num_threads(max(1, os.cpu_count() or 1))


def load_split(split, n_per_class):
    classes = sorted(os.listdir(os.path.join(ROOT, split)))
    imgs = []
    for c in classes:
        files = sorted(glob.glob(os.path.join(ROOT, split, c, "*.jpg")))[:n_per_class]
        for fp in files:
            im = Image.open(fp).convert("RGB")
            imgs.append(np.asarray(im, dtype=np.float32) / 255.0)
    arr = np.stack(imgs, 0)
    return torch.from_numpy(arr).permute(0, 3, 1, 2) * 2 - 1


betas = torch.linspace(1e-4, 0.02, T_STEPS)
alphas = 1.0 - betas
alpha_bar = torch.cumprod(alphas, 0)


def q_sample(x0, t, noise):
    ab = alpha_bar[t].view(-1, 1, 1, 1)
    return torch.sqrt(ab) * x0 + torch.sqrt(1 - ab) * noise


def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half).float() / half)
    args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
    return torch.cat([torch.sin(args), torch.cos(args)], -1)


class Block(nn.Module):
    def __init__(self, d, heads, n_tok, win_w, topk_k, layer_id, mlp_ratio):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = DecoupledSpatialTemporalAttention(d, heads, n_tok, win_w, topk_k, layer_id, 1)
        self.ln2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, d * mlp_ratio), nn.GELU(), nn.Linear(d * mlp_ratio, d))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.ffn(self.ln2(x))


class DiTTrident(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Conv2d(CH, HIDDEN, PATCH, PATCH)
        self.pos_embed = nn.Parameter(torch.randn(1, L, HIDDEN) * 0.02)
        self.time_mlp = nn.Sequential(nn.Linear(HIDDEN, HIDDEN), nn.SiLU(), nn.Linear(HIDDEN, HIDDEN))
        self.blocks = nn.ModuleList(
            [Block(HIDDEN, HEADS, L, CFG["WIN_W"], CFG["TOPK_K"], l, CFG["MLP_RATIO"]) for l in range(DEPTH)]
        )
        self.final_ln = nn.LayerNorm(HIDDEN)
        self.final_proj = nn.Linear(HIDDEN, PATCH * PATCH * CH)

    def set_refresh_interval(self, r):
        for b in self.blocks:
            b.attn.refresh_interval = max(1, r)

    def reset_cache(self):
        for b in self.blocks:
            b.attn.reset_cache()

    def set_step(self, s):
        for b in self.blocks:
            b.attn.set_step(s)

    def forward(self, x, t):
        B = x.shape[0]
        h = self.patch_embed(x).flatten(2).transpose(1, 2) + self.pos_embed
        h = h + self.time_mlp(timestep_embedding(t, HIDDEN)).unsqueeze(1)
        for b in self.blocks:
            h = b(h)
        out = self.final_proj(self.final_ln(h))
        g = IMG // PATCH
        out = out.transpose(1, 2).reshape(B, CH, PATCH, PATCH, g, g)
        return out.permute(0, 1, 4, 2, 5, 3).reshape(B, CH, IMG, IMG)


@torch.no_grad()
def sample_batch(model, b, refresh_interval, seed):
    model.set_refresh_interval(refresh_interval)
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(b, CH, IMG, IMG, generator=gen)
    model.reset_cache()
    for i, ti in enumerate(reversed(range(T_STEPS))):
        model.set_step(i)
        t = torch.full((b,), ti, dtype=torch.long)
        eps = model(x, t)
        a, ab, beta = alphas[ti], alpha_bar[ti], betas[ti]
        mean = (1 / torch.sqrt(a)) * (x - (beta / torch.sqrt(1 - ab)) * eps)
        x = mean + torch.sqrt(beta) * torch.randn(x.shape, generator=gen) if ti > 0 else mean
    return x.clamp(-1, 1)


def psnr(a, b):
    mse = F.mse_loss(a, b).item()
    return float("inf") if mse == 0 else 10 * math.log10(4.0 / mse)


def ssim_simple(a, b, C1=0.01 ** 2, C2=0.03 ** 2):
    a = (a + 1) / 2
    b = (b + 1) / 2
    ma, mb = a.mean(), b.mean()
    va, vb = a.var(), b.var()
    cov = ((a - ma) * (b - mb)).mean()
    return (((2 * ma * mb + C1) * (2 * cov + C2)) / ((ma ** 2 + mb ** 2 + C1) * (va + vb + C2))).item()


def build_inception():
    import pytorch_fid.inception as inc

    sd = torch.load(FID_WEIGHTS, map_location="cpu")
    inc.load_state_dict_from_url = lambda *a, **k: sd
    net = inc.InceptionV3([inc.InceptionV3.BLOCK_INDEX_BY_DIM[2048]], resize_input=True, normalize_input=True)
    return net.eval()


@torch.no_grad()
def inception_features(net, imgs, batch=32):
    feats = []
    x = (imgs + 1) / 2
    for i in range(0, x.shape[0], batch):
        f = net(x[i:i + batch])[0].squeeze(-1).squeeze(-1)
        feats.append(f.numpy().astype(np.float64))
    return np.concatenate(feats, 0)


def fid_from_feats(f1, f2):
    mu1, mu2 = f1.mean(0), f2.mean(0)
    s1, s2 = np.cov(f1, rowvar=False), np.cov(f2, rowvar=False)
    covmean, _ = linalg.sqrtm(s1.dot(s2), disp=False)
    if not np.isfinite(covmean).all():
        off = np.eye(s1.shape[0]) * 1e-6
        covmean = linalg.sqrtm((s1 + off).dot(s2 + off))
    covmean = covmean.real
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))


def main():
    t_proc = time.time()
    budget = CFG["TIME_BUDGET"]
    ckpt_path = os.path.join(EXP_DIR, "ckpt.pt")
    res_path = os.path.join(EXP_DIR, "path_b_fid_results.json")
    torch.manual_seed(0)
    np.random.seed(0)

    model = DiTTrident()
    model.set_refresh_interval(1)
    opt = torch.optim.Adam(model.parameters(), lr=CFG["LR"])
    start, losses = 0, []
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start, losses = ck["step"], ck["losses"]
        torch.manual_seed(1000 + start)
        np.random.seed(1000 + start)
        print(f"resumed at step {start}", flush=True)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"params={n_params/1e6:.2f}M L={L} hidden={HIDDEN} depth={DEPTH} heads={HEADS} K={CFG['TOPK_K']}", flush=True)

    step = start
    if step < CFG["N_TRAIN_STEPS"]:
        per_class = CFG["N_TRAIN_IMGS"] // 10
        train = load_split("train", per_class)
        model.train()
        while step < CFG["N_TRAIN_STEPS"] and time.time() - t_proc < budget:
            idx = torch.randint(0, train.shape[0], (CFG["BATCH"],))
            x0 = train[idx]
            if torch.rand(1).item() < 0.5:
                x0 = x0.flip(3)
            t = torch.randint(0, T_STEPS, (CFG["BATCH"],))
            noise = torch.randn_like(x0)
            model.set_step(0)
            model.reset_cache()
            loss = F.mse_loss(model(q_sample(x0, t, noise), t), noise)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
            step += 1
            if step % 50 == 0:
                print(f"step {step}/{CFG['N_TRAIN_STEPS']} loss {np.mean(losses[-50:]):.4f}", flush=True)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "losses": losses}, ckpt_path)
        if step < CFG["N_TRAIN_STEPS"]:
            print(f"NOT DONE (training): step {step}/{CFG['N_TRAIN_STEPS']}", flush=True)
            return
        print("TRAINING COMPLETE", flush=True)

    model.eval()
    results = json.load(open(res_path)) if os.path.exists(res_path) else {}
    feats_path = os.path.join(EXP_DIR, "real_feats.npy")
    net = build_inception()
    if os.path.exists(feats_path):
        f_real = np.load(feats_path)
    else:
        real = load_split("test", CFG["N_EVAL"] // 10 + 1)
        perm = torch.randperm(real.shape[0], generator=torch.Generator().manual_seed(1))
        real = real[perm][: CFG["N_EVAL"]]
        f_real = inception_features(net, real)
        np.save(feats_path, f_real)

    eb = 16
    n_batches = (CFG["N_EVAL"] + eb - 1) // eb
    for r in CFG["REFRESH_INTERVALS"]:
        key = f"R={r}"
        if key in results:
            continue
        part_path = os.path.join(EXP_DIR, f"partial_R{r}.pt")
        parts = torch.load(part_path) if os.path.exists(part_path) else []
        while len(parts) < n_batches:
            if time.time() - t_proc > budget - 70:
                torch.save(parts, part_path)
                print(f"NOT DONE (eval R={r}): {len(parts)}/{n_batches} batches", flush=True)
                return
            bi = len(parts)
            b = min(eb, CFG["N_EVAL"] - bi * eb)
            parts.append(sample_batch(model, b, r, seed=5000 + bi))
        torch.save(parts, part_path)
        s_all = torch.cat(parts, 0)
        torch.save(s_all, os.path.join(EXP_DIR, f"samples_R{r}.pt"))
        f_gen = inception_features(net, s_all)
        rec = dict(fid_vs_real=fid_from_feats(f_gen, f_real))
        ref_path = os.path.join(EXP_DIR, "samples_R1.pt")
        if r != 1 and os.path.exists(ref_path):
            ref = torch.load(ref_path)
            rec["psnr_vs_R1"] = psnr(s_all, ref)
            rec["ssim_vs_R1"] = ssim_simple(s_all, ref)
        results[key] = rec
        json.dump(results, open(res_path, "w"), indent=2)
        print(f"{key}: {rec}", flush=True)

    if "real_train_vs_real_test_fid" not in results:
        f_ref_real = inception_features(net, load_split("train", CFG["N_EVAL"] // 10 + 1)[: CFG["N_EVAL"]])
        results["real_train_vs_real_test_fid"] = fid_from_feats(f_ref_real, f_real)
    results["meta"] = dict(
        n_params=n_params, hidden=HIDDEN, depth=DEPTH, heads=HEADS, topk_k=CFG["TOPK_K"],
        win_w=CFG["WIN_W"], n_tokens=L, img=IMG, train_steps=step,
        n_eval=CFG["N_EVAL"], final_loss=float(np.mean(losses[-100:])),
    )
    json.dump(results, open(res_path, "w"), indent=2)
    print("ALL DONE", flush=True)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
