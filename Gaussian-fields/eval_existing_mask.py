"""Quick eval of the existing mask_raw_K3 checkpoint to see per-bin spectrum.
Goal: distinguish 'good below Nyquist, bad at edge' vs 'bad everywhere'.
"""
import os, sys, math, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Navier-Stokes'))
from unet import Unet
from train_multiscale_interpolation import (
    HierarchicalMasks, precompute_matern_amplitude, sample_matern_batch, get_fourier_spectrum,
)


def make_unet(R, ic, oc, ch=32):
    dm = (1, 2) if R <= 8 else (1, 2, 2) if R <= 32 else (1, 2, 2, 2)
    return Unet(num_classes=1, in_channels=ic, out_channels=oc, dim=ch, dim_mults=dm,
                resnet_block_groups=min(8, ch), learned_sinusoidal_cond=True,
                random_fourier_features=False, learned_sinusoidal_dim=max(ch, 16),
                attn_dim_head=max(ch, 16), attn_heads=4, use_classes=False)


def embed_mask(zt_F, x_C, F, C, G, R, device):
    B = zt_F.shape[0]
    if len(C) == 0:
        return zt_F.view(B, 1, R, R)
    stride = G // R
    full = torch.zeros(B, G*G, device=device); full[:, F] = zt_F; full[:, C] = x_C
    ch1 = full.view(B, G, G)[:, ::stride, ::stride].unsqueeze(1).contiguous()
    ctx = torch.zeros(B, G*G, device=device); ctx[:, C] = x_C
    ch2 = ctx.view(B, G, G)[:, ::stride, ::stride].unsqueeze(1).contiguous()
    return torch.cat([ch1, ch2], dim=1)


def extract_mask(pred, F, G, R, device):
    B = pred.shape[0]; stride = G // R
    if stride == 1: return pred.view(B, -1)[:, F]
    pf = torch.zeros(B, G, G, device=device)
    pf[:, ::stride, ::stride] = pred.view(B, R, R)
    return pf.view(B, -1)[:, F]


def gen_band(net, embed_fn, extract_fn, scale, n_samples, nF, device, n_rk4):
    zt = scale * torch.randn(n_samples, nF, device=device)
    nodes = torch.linspace(1e-3, 1-1e-3, n_rk4+1)
    for i in range(len(nodes)-1):
        sv = float(nodes[i]); ds = float(nodes[i+1]-nodes[i])
        def vel(z, tv):
            inp = embed_fn(z); tt = torch.full((n_samples,), tv, device=device)
            with torch.no_grad(): p = net(inp, tt, classes=None)
            return extract_fn(p)
        k1=vel(zt, sv); k2=vel(zt+.5*ds*k1, sv+.5*ds)
        k3=vel(zt+.5*ds*k2, sv+.5*ds); k4=vel(zt+ds*k3, sv+ds)
        zt = zt + (ds/6)*(k1+2*k2+2*k3+k4)
    return zt


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--K', type=int, default=3)
    p.add_argument('--G', type=int, default=64)
    p.add_argument('--s', type=float, default=3.0)
    p.add_argument('--n_eval', type=int, default=2000)
    p.add_argument('--ckpt', type=str, default='results/bench/mask_raw_K3')
    p.add_argument('--rk4', type=int, default=4)
    args = p.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda')
    G, K = args.G, args.K

    sigma_sq = 1.0 * ((2*math.pi)**2 + 1.0**2)**args.s
    amp = precompute_matern_amplitude(G, sigma_sq, 1.0, args.s).float()
    test_data = sample_matern_batch(amp, args.n_eval, device='cpu').float()
    truth_np = test_data.numpy()
    kvals, spec_truth = get_fourier_spectrum(truth_np)

    hier = HierarchicalMasks(G, K+1, device='cpu')
    bands = []
    for k in range(K+1):
        si = K - k
        F = torch.nonzero(hier.masks[si].cpu().flatten()).flatten()
        Cl = [torch.nonzero(hier.masks[K-j].cpu().flatten()).flatten() for j in range(k)]
        C = torch.cat(Cl) if Cl else torch.empty(0, dtype=torch.long)
        R = G // (2**(K-k))
        bands.append(dict(F=F, C=C, R=R, in_ch=1 if k==0 else 2, out_ch=1, name=f'mask_s{k}_R{R}'))

    # Estimate sigma2
    est = sample_matern_batch(amp, 10000, device=device).float().view(10000, -1)
    for b in bands:
        Ft = b['F'].to(device); Y = est[:, Ft]
        if len(b['C']) == 0:
            b['sigma2'] = Y.var(dim=0).mean().item()
        else:
            Ct = b['C'].to(device); X = est[:, Ct]
            n = X.shape[0]
            Xa = torch.cat([torch.ones(n, 1, device=device), X], dim=1)
            beta = torch.linalg.solve(Xa.T @ Xa + 1e-6 * torch.eye(Xa.shape[1], device=device), Xa.T @ Y)
            b['sigma2'] = (Y - Xa @ beta).var(dim=0).mean().item()
        b['scale'] = math.sqrt(b['sigma2'])
    del est

    gen = torch.zeros(args.n_eval, G*G, device=device)
    for bi, b in enumerate(bands):
        F = b['F'].to(device); C = b['C'].to(device) if len(b['C']) > 0 else b['C']
        nF = len(b['F']); R = b['R']; scale = b['scale']
        net = make_unet(R, b['in_ch'], b['out_ch'], 32).float().to(device)
        ckpt_path = os.path.join(args.ckpt, b['name'], 'model.pt')
        net.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        net.eval()
        xC_eval = gen[:, C] if len(C) > 0 else None
        ef = lambda z: embed_mask(z, xC_eval, F, C, G, R, device)
        xf = (lambda p: extract_mask(p, F, G, R, device)) if len(C) > 0 else (lambda p: p.view(args.n_eval, -1))
        gen[:, F] = gen_band(net, ef, xf, scale, args.n_eval, nF, device, n_rk4=args.rk4)

    gen_np = gen.view(args.n_eval, G, G).cpu().numpy()
    _, spec_gen = get_fourier_spectrum(gen_np)
    rel = np.abs(spec_gen - spec_truth) / (np.abs(spec_truth) + 1e-30)
    print(f"\nNyquist k = {G//2}")
    print(f"{'k':>4} {'E_truth':>10} {'E_gen':>10} {'ratio':>8} {'rel':>8}")
    for i in range(min(len(kvals), 35)):
        r = spec_gen[i]/(spec_truth[i]+1e-30)
        m = ' ***' if abs(r-1)>1 else (' * ' if abs(r-1)>0.1 else '   ')
        print(f"{kvals[i]:>4.0f} {spec_truth[i]:>10.3e} {spec_gen[i]:>10.3e} {r:>8.4f} {abs(r-1):>8.4f}{m}")
    # summaries
    for cap in [10, 20, 25, 28, 30]:
        sub = rel[:cap]
        print(f"  k<={cap}: mean={sub.mean():.4f}  max={sub.max():.4f}  bins<1={int((sub<1).sum())}/{cap}  bins<0.1={int((sub<0.1).sum())}/{cap}")


if __name__ == '__main__':
    main()
