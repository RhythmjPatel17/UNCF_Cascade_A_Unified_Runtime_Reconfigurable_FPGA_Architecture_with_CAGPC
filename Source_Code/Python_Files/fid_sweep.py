import argparse
import json
import torch
from .attention import TridentStepScheduler
from .reference_dit import ReferenceDiT

def ddpm_sample(model, scheduler, shape, t_steps, device):
    x = torch.randn(shape, device=device)
    scheduler.reset()
    with torch.no_grad():
        for t in range(t_steps):
            scheduler.set_step(t)
            eps_pred = model(x)
            x = x - 0.02 * eps_pred
    return x

def generate_batch(model, scheduler, num_samples, n_tok, patch_in_dim, t_steps, device, batch_size=8):
    outs = []
    remaining = num_samples
    while remaining > 0:
        b = min(batch_size, remaining)
        shape = (b, n_tok, patch_in_dim)
        outs.append(ddpm_sample(model, scheduler, shape, t_steps, device))
        remaining -= b
    return torch.cat(outs, dim=0)

def compute_fid(samples_a, samples_b):
    a = samples_a.flatten(1).mean(dim=0)
    b = samples_b.flatten(1).mean(dim=0)
    return torch.linalg.norm(a - b).item()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--refresh-intervals', type=int, nargs='+', default=[1, 2, 4, 8, 16, 32, 50])
    ap.add_argument('--num-samples', type=int, default=32)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--t-steps', type=int, default=50)
    ap.add_argument('--n-tok', type=int, default=2304)
    ap.add_argument('--hid-d', type=int, default=384)
    ap.add_argument('--n-heads', type=int, default=6)
    ap.add_argument('--win-w', type=int, default=4)
    ap.add_argument('--topk-k', type=int, default=32)
    ap.add_argument('--depth', type=int, default=12)
    ap.add_argument('--patch-in-dim', type=int, default=12)
    ap.add_argument('--device', type=str, default='cpu')
    ap.add_argument('--checkpoint', type=str, default=None, help='TODO: path to your trained DiT-S/2 checkpoint; without this, ReferenceDiT runs with random weights (plumbing smoke-test only)')
    ap.add_argument('--fid-reference', type=str, default='r1', choices=['r1', 'real'], help="'r1' compares each R against the R=1 (no-reuse) model's own outputs, isolating the reuse approximation's effect per R2.5's guidance; 'real' needs --real-images-dir (not implemented here -- wire to your existing eval pipeline)")
    ap.add_argument('--output', type=str, default='fid_sweep_results.json')
    args = ap.parse_args()
    device = torch.device(args.device)
    model = ReferenceDiT(n_tok=args.n_tok, hid_d=args.hid_d, n_heads=args.n_heads, win_w=args.win_w, topk_k=args.topk_k, depth=args.depth, patch_in_dim=args.patch_in_dim, refresh_interval=1).to(device)
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state)
    else:
        print("WARNING: no --checkpoint given; running ReferenceDiT with random weights. This validates the sweep script's plumbing only -- the FID numbers below are meaningless until you supply your trained model.")
    model.eval()
    scheduler = TridentStepScheduler(model)
    torch.manual_seed(0)
    baseline = generate_batch(model, scheduler, args.num_samples, args.n_tok, args.patch_in_dim, args.t_steps, device, args.batch_size)
    results = {}
    for r in args.refresh_intervals:
        model.set_refresh_interval(r)
        torch.manual_seed(0)
        samples = generate_batch(model, scheduler, args.num_samples, args.n_tok, args.patch_in_dim, args.t_steps, device, args.batch_size)
        if args.fid_reference == 'r1' and r == 1:
            fid = 0.0
        else:
            fid = compute_fid(samples, baseline)
        results[r] = fid
        print(f"R={r:3d}  proxy-FID vs R=1 baseline: {fid:.4f}  (placeholder metric -- see compute_fid's TODO)")
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nWrote {args.output}')
    print('\nReminder: this run used the placeholder proxy metric and/or random weights unless you supplied --checkpoint and wired in a real FID implementation. Do not cite these numbers.')
if __name__ == '__main__':
    main()
