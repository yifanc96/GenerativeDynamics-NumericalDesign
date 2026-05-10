"""Unified inference eval for Gaussian-field G=64 multiscale-vs-single-scale comparison.

For each method, runs n_seeds independent batches of n_per_seed samples and reports
mean ± std of relative spectrum error per bin and per NFE. Outputs a NumPy npz with
results plus PNG spectrum plots.

Methods evaluated (mask K=3 for multiscale, all at G=64, Matern s=3):
  - mfm:  Multiscale FM (mask perband, vanilla flow)
  - mmf:  Multiscale meanflow (mask perband, flow-map)
  - msc:  Multiscale shortcut (mask perband, flow-map)
  - sfm:  Single-scale FM, raw chen2025scale spectra (linear schedule + Lip-opt)
  - smf:  Single-scale meanflow (gauss base)
  - oracle: analytic Gaussian-conditioning oracle (Schur complement, mask K=3)
"""
import os, sys, math, argparse, json
import numpy as np
import torch
import torch.nn as nn
from time import time as timer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Navier-Stokes'))
from unet import Unet, RandomOrLearnedSinusoidalPosEmb
from train_multiscale_interpolation import (
    HierarchicalMasks, precompute_matern_amplitude, sample_matern_batch, get_fourier_spectrum,
)
from train_multiscale_perband import setup, ridge_regression, embed_mask, extract_mask
from train_multiscale_perband_meanflow import BandMFNet


# --- helpers ---
def make_unet(R, ic, oc, ch=32):
    dm = (1, 2) if R <= 8 else (1, 2, 2) if R <= 32 else (1, 2, 2, 2)
    return Unet(num_classes=1, in_channels=ic, out_channels=oc, dim=ch, dim_mults=dm,
                resnet_block_groups=min(8, ch), learned_sinusoidal_cond=True,
                random_fourier_features=False, learned_sinusoidal_dim=max(ch, 16),
                attn_dim_head=max(ch, 16), attn_heads=4, use_classes=False)


class MeanFlowVelocity(nn.Module):
    """Single-scale meanflow network (matches train_gaussian_field_meanflow.py)."""
    def __init__(self, ch=32, dim_mults=(1, 2, 2, 2)):
        super().__init__()
        self.net = Unet(num_classes=1, in_channels=1, out_channels=1, dim=ch,
                        dim_mults=dim_mults, resnet_block_groups=8,
                        learned_sinusoidal_cond=True, random_fourier_features=False,
                        learned_sinusoidal_dim=32, attn_dim_head=32, attn_heads=4, use_classes=False)
        time_dim = ch * 4
        sinu = RandomOrLearnedSinusoidalPosEmb(32, is_random=False)
        self.r_mlp = nn.Sequential(sinu, nn.Linear(33, time_dim), nn.GELU(), nn.Linear(time_dim, time_dim))
        nn.init.zeros_(self.r_mlp[-1].weight); nn.init.zeros_(self.r_mlp[-1].bias)

    def forward(self, zt, s, r):
        net = self.net
        s_emb = net.time_mlp(s); r_emb = self.r_mlp(r); t_emb = s_emb + r_emb
        x = net.init_conv(zt); r_skip = x.clone(); h = []
        for block1, block2, attn, downsample in net.downs:
            x = block1(x, t_emb, None); h.append(x)
            x = block2(x, t_emb, None); x = attn(x); h.append(x)
            x = downsample(x)
        x = net.mid_block1(x, t_emb, None); x = net.mid_attn(x); x = net.mid_block2(x, t_emb, None)
        for block1, block2, attn, upsample in net.ups:
            x = torch.cat((x, h.pop()), dim=1); x = block1(x, t_emb, None)
            x = torch.cat((x, h.pop()), dim=1); x = block2(x, t_emb, None); x = attn(x)
            x = upsample(x)
        x = torch.cat((x, r_skip), dim=1)
        x = net.final_res_block(x, t_emb, None)
        return net.final_conv(x)


def gen_band_fm(net, embed_fn, extract_fn, scale, n_samples, nF, device, n_rk4):
    zt = scale * torch.randn(n_samples, nF, device=device)
    nodes = torch.linspace(1e-3, 1 - 1e-3, n_rk4 + 1)
    for i in range(len(nodes) - 1):
        sv = float(nodes[i]); ds = float(nodes[i + 1] - nodes[i])
        def vel(z, tv):
            tt = torch.full((n_samples,), tv, device=device)
            inp = embed_fn(z)
            with torch.no_grad():
                p = net(inp, tt, classes=None)
            return extract_fn(p)
        k1 = vel(zt, sv); k2 = vel(zt + .5 * ds * k1, sv + .5 * ds)
        k3 = vel(zt + .5 * ds * k2, sv + .5 * ds); k4 = vel(zt + ds * k3, sv + ds)
        zt = zt + (ds / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return zt


def gen_band_mf(net, embed_fn, extract_fn, scale, n_samples, nF, device, n_mf):
    zt = scale * torch.randn(n_samples, nF, device=device)
    s_vals = torch.linspace(0.0, 1.0, n_mf + 1)
    for j in range(n_mf):
        s_j = float(s_vals[j]); r_j = float(s_vals[j + 1])
        sb = torch.full((n_samples,), s_j, device=device)
        rb = torch.full((n_samples,), r_j, device=device)
        inp = embed_fn(zt)
        with torch.no_grad():
            pred = net(inp, sb, rb)
        vel = extract_fn(pred)
        zt = zt + (r_j - s_j) * vel
    return zt


def gen_single_mf(net, n_samples, G, device, n_mf):
    zt = torch.randn(n_samples, 1, G, G, device=device)
    s_vals = torch.linspace(0.0, 1.0, n_mf + 1)
    for j in range(n_mf):
        s_j = float(s_vals[j]); r_j = float(s_vals[j + 1])
        sb = torch.full((n_samples,), s_j, device=device)
        rb = torch.full((n_samples,), r_j, device=device)
        with torch.no_grad():
            v = net(zt, sb, rb)
        zt = zt + (r_j - s_j) * v
    return zt.squeeze(1)


# --- multiscale eval ---
def setup_mask_bands(G, K, device, amp):
    hier = HierarchicalMasks(G, K + 1, device='cpu')
    bands = []
    for k in range(K + 1):
        si = K - k
        F = torch.nonzero(hier.masks[si].cpu().flatten()).flatten()
        Cl = [torch.nonzero(hier.masks[K - j].cpu().flatten()).flatten() for j in range(k)]
        C = torch.cat(Cl) if Cl else torch.empty(0, dtype=torch.long)
        R = G // (2**(K - k))
        bands.append(dict(F=F, C=C, R=R, in_ch=1 if k == 0 else 2, out_ch=1, name=f'mask_s{k}_R{R}'))

    est = sample_matern_batch(amp, 10000, device=device).float().view(10000, -1)
    for b in bands:
        Ft = b['F'].to(device); Y = est[:, Ft]
        if len(b['C']) == 0:
            b['sigma2'] = Y.var(dim=0).mean().item()
        else:
            Ct = b['C'].to(device); X = est[:, Ct]
            b['sigma2'], _, _ = ridge_regression(Y, X)
        b['scale'] = math.sqrt(b['sigma2'])
    del est
    return bands


def run_multiscale(method, ckpt_dir, bands, G, n_samples, n_rk4, ch, device):
    """method ∈ {'fm','mf','sc'}.  fm uses Unet, mf/sc use BandMFNet."""
    gen = torch.zeros(n_samples, G * G, device=device)
    for bi, b in enumerate(bands):
        F = b['F'].to(device); C = b['C'].to(device) if len(b['C']) > 0 else b['C']
        nF = len(b['F']); R = b['R']; scale = b['scale']
        if method == 'fm':
            net = make_unet(R, b['in_ch'], b['out_ch'], ch).float().to(device)
        else:
            net = BandMFNet(R, b['in_ch'], b['out_ch'], ch).float().to(device)
        sd = torch.load(os.path.join(ckpt_dir, b['name'], 'model.pt'),
                        map_location=device, weights_only=True)
        net.load_state_dict(sd); net.eval()
        xC = gen[:, C] if len(C) > 0 else None
        ef = lambda z: embed_mask(z, xC, F, C, G, R, device)
        xf = (lambda p: extract_mask(p, F, G, R, device)) if len(C) > 0 else (lambda p: p.view(n_samples, -1))
        if method == 'fm':
            gen[:, F] = gen_band_fm(net, ef, xf, scale, n_samples, nF, device, n_rk4)
        else:
            gen[:, F] = gen_band_mf(net, ef, xf, scale, n_samples, nF, device, n_rk4)
    return gen.view(n_samples, G, G).cpu().numpy()


def run_single_mf(ckpt, G, n_samples, n_mf, device):
    net = MeanFlowVelocity(ch=32, dim_mults=(1, 2, 2, 2)).float().to(device)
    sd = torch.load(ckpt, map_location=device, weights_only=True)
    net.load_state_dict(sd); net.eval()
    return gen_single_mf(net, n_samples, G, device, n_mf).cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=int, default=2)
    p.add_argument('--n_seeds', type=int, default=5)
    p.add_argument('--n_per_seed', type=int, default=1000)
    p.add_argument('--K', type=int, default=3)
    p.add_argument('--G', type=int, default=64)
    p.add_argument('--s', type=float, default=3.0)
    p.add_argument('--out', type=str, default='results/eval_gf_unified.npz')
    args = p.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda')
    G, K = args.G, args.K

    amp = setup(G, args.s)

    # Truth spectrum: average over multiple large batches for stable reference
    truth_specs = []
    for s in range(args.n_seeds):
        torch.manual_seed(1000 + s); np.random.seed(1000 + s)
        td = sample_matern_batch(amp, args.n_per_seed, device='cpu').float()
        _, sp = get_fourier_spectrum(td.numpy())
        truth_specs.append(sp)
    truth_specs = np.stack(truth_specs)  # (n_seeds, n_k)
    spec_truth_mean = truth_specs.mean(0)

    bands = setup_mask_bands(G, K, device, amp)

    # Methods to evaluate (only those with available checkpoints)
    method_configs = [
        # (label, method-type, ckpt_dir-or-file, NFE_list)
        ('mfm_long',  'fm', 'results/bench/mask_raw_K3',     [32, 64, 128]),  # current long-train
        ('mmf',       'mf', 'results/perband_mf_K3',         [4, 8, 16, 32]),
        ('msc',       'mf', 'results/perband_shortcut_K3',   [4, 8, 16, 32]),
        ('smf_gauss', 'single_mf', 'results/gf_meanflow_gauss/model_final.pt', [1, 2, 4, 8, 16]),
    ]

    # Note: 'mfm_short' refers to the 100k-step original run; assume snapshot
    # was saved separately. Skip if dir doesn't exist.

    results = {'truth_specs': truth_specs, 'spec_truth_mean': spec_truth_mean}

    for label, mtype, ckpt, nfe_list in method_configs:
        if not os.path.exists(ckpt):
            print(f"SKIP {label}: ckpt {ckpt} not found")
            continue
        print(f"\n=== {label} ({mtype}) ===")
        nfe_results = {}  # NFE -> (seeds, k) array of rel errors
        nfe_specs = {}    # NFE -> (seeds, k) array of generated spectra
        for nfe in nfe_list:
            seed_specs = []
            for s in range(args.n_seeds):
                torch.manual_seed(s); np.random.seed(s)
                if mtype == 'fm':
                    # for vanilla FM: nfe = total integrations across all bands * 4 evals/RK4
                    # interpret nfe as TOTAL NFE; per-band rk4 = nfe / (4 bands * 4 evals) = nfe/16
                    n_rk4_per_band = nfe // (4 * (K + 1))
                    if n_rk4_per_band == 0: n_rk4_per_band = 1
                    gen_np = run_multiscale('fm', ckpt, bands, G, args.n_per_seed, n_rk4_per_band, 32, device)
                elif mtype == 'mf':
                    # MF nfe = 1 eval/step * (K+1) bands * n_mf_steps. So n_mf = nfe / (K+1)
                    n_mf_per_band = nfe // (K + 1)
                    if n_mf_per_band == 0: n_mf_per_band = 1
                    gen_np = run_multiscale('mf', ckpt, bands, G, args.n_per_seed, n_mf_per_band, 32, device)
                elif mtype == 'single_mf':
                    n_mf = nfe  # single-scale: nfe = number of MF steps
                    gen_np = run_single_mf(ckpt, G, args.n_per_seed, n_mf, device)
                _, spec = get_fourier_spectrum(gen_np)
                seed_specs.append(spec)
            seed_specs = np.stack(seed_specs)  # (n_seeds, n_k)
            rel = np.abs(seed_specs - spec_truth_mean[None]) / (np.abs(spec_truth_mean[None]) + 1e-30)
            mean_rel_30 = rel[:, :30].mean(axis=1)  # (n_seeds,)
            print(f"  NFE={nfe:4d}  mean_rel<=30 = {mean_rel_30.mean():.4f} ± {mean_rel_30.std():.4f}  (n_seeds={args.n_seeds})")
            nfe_results[nfe] = rel
            nfe_specs[nfe] = seed_specs
        for nfe, r in nfe_results.items():
            results[f'rel_{label}_NFE{nfe}'] = r
        for nfe, sp in nfe_specs.items():
            results[f'spec_{label}_NFE{nfe}'] = sp

    results['method_labels'] = json.dumps([(m[0], m[1], m[3]) for m in method_configs])
    np.savez_compressed(args.out, **{k: v for k, v in results.items() if not isinstance(v, dict)})
    print(f"\nWrote {args.out}")


if __name__ == '__main__':
    main()
