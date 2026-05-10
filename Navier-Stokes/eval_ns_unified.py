"""Unified NS G=128 inference eval — analog of Gaussian-fields/eval_gf_unified.py.

Methods evaluated (mask K=3 multiscale, NS at hi_size=128, single data file):
  - mfm:  Multiscale FM (1-channel embedding, train_ns_multiscale_v2.py, no_tnorm=True).
  - mmf:  Multiscale meanflow (2-channel embedding, train_ns_perband_meanflow.py).
  - msc:  Multiscale shortcut (2-channel embedding, train_ns_perband_shortcut.py).
"""
import os, sys, math, json, argparse, glob
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from unet import Unet, RandomOrLearnedSinusoidalPosEmb
from train_ns_multiscale_v2 import (
    load_ns_data, get_energy_spectrum, build_mask_bands as build_v2_bands,
    embed_1ch, extract_F as extract_F_v2, estimate_band_stats,
    c_in as c_in_v2, c_out_val as c_out_val_v2,
    generate_band as generate_band_v2_fm, make_unet,
)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Gaussian-fields'))
from train_multiscale_interpolation import HierarchicalMasks
from train_multiscale_perband import ridge_regression, embed_mask, extract_mask
from train_multiscale_perband_meanflow import BandMFNet


def build_perband_bands(G, K, name_prefix='mask_'):
    """2-channel (perband) bands matching train_ns_perband_meanflow.py."""
    hier = HierarchicalMasks(G, K + 1, device='cpu')
    bands = []
    for k in range(K + 1):
        si = K - k
        F = torch.nonzero(hier.masks[si].cpu().flatten()).flatten()
        Cl = [torch.nonzero(hier.masks[K - j].cpu().flatten()).flatten() for j in range(k)]
        C = torch.cat(Cl) if Cl else torch.empty(0, dtype=torch.long)
        R = G // (2**(K - k))
        bands.append(dict(F=F, C=C, R=R, in_ch=1 if k == 0 else 2, out_ch=1, name=f'{name_prefix}s{k}_R{R}'))
    return bands


def run_multiscale_fm_v1(ckpt_dir, bands_pb, train_data, G, n_samples, n_rk4, device, ch=32):
    """v1-style 2-channel multiscale FM (train_ns_multiscale_perband.py).
    Bands have name 's{k}_R{R}' (no mask_ prefix).
    """
    train_flat = train_data.to(device).float().view(train_data.shape[0], -1)
    n_ridge = min(5000, train_flat.shape[0])
    for b in bands_pb:
        Ft = b['F'].to(device); Y = train_flat[:n_ridge, Ft]
        if len(b['C']) == 0:
            b['sigma2'] = Y.var(dim=0).mean().item()
        else:
            Ct = b['C'].to(device); X = train_flat[:n_ridge, Ct]
            b['sigma2'], _, _ = ridge_regression(Y, X)
        b['scale'] = math.sqrt(b['sigma2'])

    gen = torch.zeros(n_samples, G * G, device=device)
    for bi, b in enumerate(bands_pb):
        F = b['F'].to(device); C = b['C'].to(device) if len(b['C']) > 0 else b['C']
        nF = len(b['F']); R = b['R']; scale = b['scale']
        net = make_unet(R, b['in_ch'], b['out_ch'], ch).float().to(device)
        sd = torch.load(os.path.join(ckpt_dir, b['name'], 'model.pt'),
                        map_location=device, weights_only=True)
        net.load_state_dict(sd); net.eval()
        xC = gen[:, C] if len(C) > 0 else None
        ef = lambda z: embed_mask(z, xC, F, C, G, R, device)
        xf = (lambda p: extract_mask(p, F, G, R, device)) if len(C) > 0 else (lambda p: p.view(n_samples, -1))
        # vanilla FM RK4
        zt = scale * torch.randn(n_samples, nF, device=device)
        nodes = torch.linspace(1e-3, 1 - 1e-3, n_rk4 + 1)
        for i in range(len(nodes) - 1):
            sv = float(nodes[i]); ds = float(nodes[i + 1] - nodes[i])
            def vel(z, tv):
                tt = torch.full((n_samples,), tv, device=device)
                with torch.no_grad():
                    p = net(ef(z), tt, classes=None)
                return xf(p)
            k1 = vel(zt, sv); k2 = vel(zt + .5 * ds * k1, sv + .5 * ds)
            k3 = vel(zt + .5 * ds * k2, sv + .5 * ds); k4 = vel(zt + ds * k3, sv + ds)
            zt = zt + (ds / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        gen[:, F] = zt
    return gen.view(n_samples, G, G).cpu()


def gen_band_mf(net, embed_fn, extract_fn, scale, n_samples, nF, device, n_mf):
    zt = scale * torch.randn(n_samples, nF, device=device)
    s_vals = torch.linspace(0.0, 1.0, n_mf + 1)
    for j in range(n_mf):
        s_j = float(s_vals[j]); r_j = float(s_vals[j + 1])
        sb = torch.full((n_samples,), s_j, device=device)
        rb = torch.full((n_samples,), r_j, device=device)
        with torch.no_grad():
            vel = extract_fn(net(embed_fn(zt), sb, rb))
        zt = zt + (r_j - s_j) * vel
    return zt


def run_multiscale_fm(ckpt_dir, bands_v2, train_data, G, n_samples, n_rk4, device, no_tnorm=True, ch=32):
    """Run train_ns_multiscale_v2.py-style multiscale FM (1-channel)."""
    # Estimate per-band sigma2, v_F from training data
    train_flat = train_data.to(device).float().view(train_data.shape[0], -1)
    for b in bands_v2:
        sigma2, v_F = estimate_band_stats(train_flat, b['F'].to(device),
                                           b['C'].to(device) if len(b['C']) > 0 else b['C'], device)
        b['sigma2'] = sigma2; b['v_F'] = v_F; b['scale'] = math.sqrt(sigma2)

    gen = torch.zeros(n_samples, G * G, device=device)
    for bi, b in enumerate(bands_v2):
        F = b['F'].to(device); C = b['C'].to(device) if len(b['C']) > 0 else b['C']
        nF = len(b['F']); R = b['R']
        net = make_unet(R, b['in_ch'], b['out_ch'], ch).float().to(device)
        sd = torch.load(os.path.join(ckpt_dir, b['name'], 'model.pt'),
                        map_location=device, weights_only=True)
        net.load_state_dict(sd); net.eval()
        xC = gen[:, C] if len(C) > 0 else None
        gen[:, F] = generate_band_v2_fm(net, F, C, G, R, b['sigma2'], b['v_F'],
                                         n_samples, device, x_C=xC, n_rk4=n_rk4, no_tnorm=no_tnorm)
    return gen.view(n_samples, G, G).cpu()


def run_multiscale_mf_or_sc(ckpt_dir, bands_pb, train_data, G, n_samples, n_mf, device, ch=32):
    """Run train_ns_perband_meanflow.py-style multiscale (2-channel) for MF or shortcut."""
    train_flat = train_data.to(device).float().view(train_data.shape[0], -1)
    n_ridge = min(5000, train_flat.shape[0])
    for b in bands_pb:
        Ft = b['F'].to(device); Y = train_flat[:n_ridge, Ft]
        if len(b['C']) == 0:
            b['sigma2'] = Y.var(dim=0).mean().item()
        else:
            Ct = b['C'].to(device); X = train_flat[:n_ridge, Ct]
            b['sigma2'], _, _ = ridge_regression(Y, X)
        b['scale'] = math.sqrt(b['sigma2'])

    gen = torch.zeros(n_samples, G * G, device=device)
    for bi, b in enumerate(bands_pb):
        F = b['F'].to(device); C = b['C'].to(device) if len(b['C']) > 0 else b['C']
        nF = len(b['F']); R = b['R']; scale = b['scale']
        net = BandMFNet(R, b['in_ch'], b['out_ch'], ch).float().to(device)
        sd = torch.load(os.path.join(ckpt_dir, b['name'], 'model.pt'),
                        map_location=device, weights_only=True)
        net.load_state_dict(sd); net.eval()
        xC = gen[:, C] if len(C) > 0 else None
        ef = lambda z: embed_mask(z, xC, F, C, G, R, device)
        xf = (lambda p: extract_mask(p, F, G, R, device)) if len(C) > 0 else (lambda p: p.view(n_samples, -1))
        gen[:, F] = gen_band_mf(net, ef, xf, scale, n_samples, nF, device, n_mf)
    return gen.view(n_samples, G, G).cpu()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--n_seeds', type=int, default=3)
    p.add_argument('--n_per_seed', type=int, default=400)
    p.add_argument('--K', type=int, default=3)
    p.add_argument('--G', type=int, default=128)
    p.add_argument('--data_loc', type=str, default='../NSdata/data_file.pt')
    p.add_argument('--out', type=str, default='results/eval_ns_unified.npz')
    args = p.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda')
    G, K = args.G, args.K

    # Load NS data (test split)
    print(f'Loading NS data {args.data_loc} at G={G}...')
    # Multiscale FM v2 was trained on all 5 NS files; use them all for consistent stats
    suffixes = ['', '02', '03', '04', '05']
    data_locs = [f'../NSdata/data_file{s}.pt' for s in suffixes]
    data_locs = [d for d in data_locs if os.path.exists(d)]
    train_data, test_data = load_ns_data(data_locs, G)
    print(f'  train: {train_data.shape}  test: {test_data.shape}')
    # Use test set for truth spectrum, multiple seeds = multiple non-overlapping batches
    n_per_seed = args.n_per_seed
    truth_specs = []
    for s in range(args.n_seeds):
        idx0 = s * n_per_seed; idx1 = idx0 + n_per_seed
        if idx1 > test_data.shape[0]:
            print(f'  WARN: only {test_data.shape[0]} test samples, capping seeds.')
            break
        td = test_data[idx0:idx1]
        kvals, enstr_truth, _ = get_energy_spectrum(td)
        truth_specs.append(enstr_truth)
    truth_specs = np.stack(truth_specs)
    spec_truth_mean = truth_specs.mean(0)
    n_seeds_eff = truth_specs.shape[0]
    print(f'  effective n_seeds = {n_seeds_eff}')

    # Build bands once for each style
    bands_v2 = build_v2_bands(G, K)  # 1-channel (v2 trainer)
    bands_pb = build_perband_bands(G, K)  # 2-channel with mask_ prefix (perband_meanflow / shortcut)
    bands_v1 = build_perband_bands(G, K, name_prefix='')  # 2-channel with s{k}_R{R} (v1 multiscale_perband)

    method_configs = [
        ('mfm', 'fm_v1', 'results/ns_mask_K3_G128', bands_v1, [16, 32, 64, 128]),
        ('mmf', 'mf_pb', 'results/ns_perband_mf_K3', bands_pb, [4, 8, 16, 32]),
        ('msc', 'mf_pb', 'results/ns_perband_shortcut_K3', bands_pb, [4, 8, 16, 32]),
    ]

    results = {'spec_truth_mean': spec_truth_mean, 'truth_specs': truth_specs, 'kvals': kvals}

    for label, mtype, ckpt, bands_use, nfe_list in method_configs:
        if not os.path.exists(ckpt):
            print(f'SKIP {label}: {ckpt} not found'); continue
        # check presence of all band model files
        missing = [b['name'] for b in bands_use if not os.path.exists(os.path.join(ckpt, b['name'], 'model.pt'))]
        if missing:
            # try v2 naming s{k}_R{R}
            v2_names = [b['name'].replace('mask_s', 's') for b in bands_use]
            if all(os.path.exists(os.path.join(ckpt, n, 'model.pt')) for n in v2_names):
                # v2 path: rename for our bands
                for b, vn in zip(bands_use, v2_names):
                    b['name'] = vn
            else:
                print(f'SKIP {label}: missing bands in {ckpt}: {missing}'); continue

        print(f'\n=== {label} ({mtype}) ===')
        nfe_results = {}; nfe_specs = {}
        for nfe in nfe_list:
            seed_specs = []
            for s in range(n_seeds_eff):
                torch.manual_seed(s); np.random.seed(s)
                if mtype == 'fm_v2':
                    n_rk4_per_band = max(1, nfe // (4 * (K + 1)))
                    gen = run_multiscale_fm(ckpt, bands_use, train_data, G, n_per_seed,
                                             n_rk4_per_band, device, no_tnorm=True)
                elif mtype == 'fm_v1':
                    n_rk4_per_band = max(1, nfe // (4 * (K + 1)))
                    gen = run_multiscale_fm_v1(ckpt, bands_use, train_data, G, n_per_seed,
                                                n_rk4_per_band, device)
                else:
                    n_mf_per_band = max(1, nfe // (K + 1))
                    gen = run_multiscale_mf_or_sc(ckpt, bands_use, train_data, G, n_per_seed,
                                                   n_mf_per_band, device)
                _, enst_gen, _ = get_energy_spectrum(gen)
                seed_specs.append(enst_gen)
            seed_specs = np.stack(seed_specs)
            rel = np.abs(seed_specs - spec_truth_mean[None]) / (np.abs(spec_truth_mean[None]) + 1e-30)
            mr60 = rel[:, :60].mean(axis=1)
            print(f'  NFE={nfe:4d}  mean_rel<=60 = {mr60.mean():.4f} ± {mr60.std():.4f}')
            nfe_results[nfe] = rel
            nfe_specs[nfe] = seed_specs
        for nfe, r in nfe_results.items():
            results[f'rel_{label}_NFE{nfe}'] = r
        for nfe, sp in nfe_specs.items():
            results[f'spec_{label}_NFE{nfe}'] = sp

    results['method_labels'] = json.dumps([(m[0], m[1], m[4]) for m in method_configs])
    np.savez_compressed(args.out, **{k: v for k, v in results.items() if not isinstance(v, dict)})
    print(f'\nWrote {args.out}')


if __name__ == '__main__':
    main()
