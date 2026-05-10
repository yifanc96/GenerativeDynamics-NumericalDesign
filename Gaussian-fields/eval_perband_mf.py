"""Eval existing per-band MEAN FLOW / SHORTCUT checkpoints.
User says they remember earlier multiscale Gaussian field results worked very well.
Likely candidates: results/perband_mf_K3 (meanflow) or results/perband_shortcut_K3.
"""
import os, sys, math, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Navier-Stokes'))
from train_multiscale_interpolation import (
    HierarchicalMasks, precompute_matern_amplitude, sample_matern_batch, get_fourier_spectrum,
)
from train_multiscale_perband import setup, ridge_regression, embed_mask, extract_mask
from train_multiscale_perband_meanflow import BandMFNet, generate_band_mf


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=int, default=3)
    p.add_argument('--K', type=int, default=3)
    p.add_argument('--G', type=int, default=64)
    p.add_argument('--s', type=float, default=3.0)
    p.add_argument('--ch', type=int, default=32)
    p.add_argument('--n_eval', type=int, default=2000)
    p.add_argument('--ckpt_dir', type=str, required=True,
                   help='e.g. results/perband_mf_K3 or results/perband_shortcut_K3')
    args = p.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda')
    G, K = args.G, args.K

    amp = setup(G, args.s)
    test_data = sample_matern_batch(amp, args.n_eval, device='cpu').float()
    truth_np = test_data.numpy()
    kvals, spec_truth = get_fourier_spectrum(truth_np)

    hier = HierarchicalMasks(G, K + 1, device='cpu')
    bands = []
    for k in range(K + 1):
        si = K - k
        F = torch.nonzero(hier.masks[si].cpu().flatten()).flatten()
        Cl = [torch.nonzero(hier.masks[K - j].cpu().flatten()).flatten() for j in range(k)]
        C = torch.cat(Cl) if Cl else torch.empty(0, dtype=torch.long)
        R = G // (2**(K - k))
        bands.append(dict(F=F, C=C, R=R, in_ch=1 if k == 0 else 2, out_ch=1,
                          name=f'mask_s{k}_R{R}'))

    # Estimate sigma2 (same way as training)
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

    # Try a range of MF steps per band
    for n_mf in [1, 2, 4]:
        gen = torch.zeros(args.n_eval, G * G, device=device)
        for bi, b in enumerate(bands):
            F = b['F'].to(device); C = b['C'].to(device) if len(b['C']) > 0 else b['C']
            nF = len(b['F']); R = b['R']; scale = b['scale']
            net = BandMFNet(R, b['in_ch'], b['out_ch'], args.ch).float().to(device)
            ckpt = os.path.join(args.ckpt_dir, b['name'], 'model.pt')
            if not os.path.exists(ckpt):
                print(f"missing: {ckpt}"); return
            net.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
            net.eval()
            xC_eval = gen[:, C] if len(C) > 0 else None
            ef = lambda z: embed_mask(z, xC_eval, F, C, G, R, device)
            xf = (lambda p: extract_mask(p, F, G, R, device)) if len(C) > 0 else (lambda p: p.view(args.n_eval, -1))
            gen[:, F] = generate_band_mf(net, ef, xf, scale, args.n_eval, nF, device, n_mf_steps=n_mf)

        gen_np = gen.view(args.n_eval, G, G).cpu().numpy()
        _, spec_gen = get_fourier_spectrum(gen_np)
        rel = np.abs(spec_gen - spec_truth) / (np.abs(spec_truth) + 1e-30)
        nb1 = int((rel[:30] < 1).sum())
        nb01 = int((rel[:30] < 0.1).sum())
        print(f"\n=== {args.ckpt_dir}  MF steps/band={n_mf}  NFE={n_mf*4} ===")
        print(f"mean_rel<=30 = {rel[:30].mean():.4f}  max_rel<=30 = {rel[:30].max():.4f}  bins<1={nb1}/30  bins<0.1={nb01}/30")
        # by range
        for cap in [10, 14, 20, 28, 30]:
            sub = rel[:cap]
            print(f"  k<={cap:>2}: mean={sub.mean():.4f}  max={sub.max():.4f}  bins<0.1={int((sub<0.1).sum())}/{cap}")
        # nyquist edge
        if len(rel) >= 33:
            print(f"  k=31: rel={rel[30]:.4f}   k=32: rel={rel[31]:.4f}")


if __name__ == '__main__':
    main()
