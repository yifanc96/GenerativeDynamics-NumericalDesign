"""Quick per-bin eval for bands 0-2 of the long FM run, sampled at R=32."""
import os, sys, math, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Navier-Stokes'))
from unet import Unet
from train_multiscale_interpolation import (
    HierarchicalMasks, precompute_matern_amplitude, sample_matern_batch, get_fourier_spectrum,
)
from train_multiscale_perband import setup, ridge_regression, embed_mask, extract_mask


def make_unet(R, ic, oc, ch=32):
    dm = (1, 2) if R <= 8 else (1, 2, 2) if R <= 32 else (1, 2, 2, 2)
    return Unet(num_classes=1, in_channels=ic, out_channels=oc, dim=ch, dim_mults=dm,
                resnet_block_groups=min(8, ch), learned_sinusoidal_cond=True,
                random_fourier_features=False, learned_sinusoidal_dim=max(ch, 16),
                attn_dim_head=max(ch, 16), attn_heads=4, use_classes=False)


def gen_band(net, embed_fn, extract_fn, scale, n_samples, nF, device, n_rk4=4):
    zt = scale * torch.randn(n_samples, nF, device=device)
    nodes = torch.linspace(1e-3, 1-1e-3, n_rk4+1)
    for i in range(len(nodes)-1):
        sv = float(nodes[i]); ds = float(nodes[i+1]-nodes[i])
        def vel(z, tv):
            tt = torch.full((n_samples,), tv, device=device)
            inp = embed_fn(z)
            with torch.no_grad():
                p = net(inp, tt, classes=None)
            return extract_fn(p)
        k1 = vel(zt, sv); k2 = vel(zt+.5*ds*k1, sv+.5*ds)
        k3 = vel(zt+.5*ds*k2, sv+.5*ds); k4 = vel(zt+ds*k3, sv+ds)
        zt = zt + (ds/6)*(k1+2*k2+2*k3+k4)
    return zt


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=int, default=2)
    p.add_argument('--K', type=int, default=3)
    p.add_argument('--G', type=int, default=64)
    p.add_argument('--s', type=float, default=3.0)
    p.add_argument('--n_eval', type=int, default=2000)
    p.add_argument('--n_rk4', type=int, default=4)
    args = p.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda')
    G, K = args.G, args.K

    amp = setup(G, args.s)
    test_data = sample_matern_batch(amp, args.n_eval, device='cpu').float()
    truth_np = test_data.numpy()

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

    # Generate up to band 2 (R=32), no band 3
    gen = torch.zeros(args.n_eval, G * G, device=device)
    for bi in [0, 1, 2]:
        b = bands[bi]
        F = b['F'].to(device); C = b['C'].to(device) if len(b['C']) > 0 else b['C']
        nF = len(b['F']); R = b['R']; scale = b['scale']
        net = make_unet(R, b['in_ch'], b['out_ch'], 32).float().to(device)
        net.load_state_dict(torch.load(f'results/bench/mask_raw_K3/{b["name"]}/model.pt',
                                        map_location=device, weights_only=True))
        net.eval()
        xC = gen[:, C] if len(C) > 0 else None
        ef = lambda z: embed_mask(z, xC, F, C, G, R, device)
        xf = (lambda p: extract_mask(p, F, G, R, device)) if len(C) > 0 else (lambda p: p.view(args.n_eval, -1))
        gen[:, F] = gen_band(net, ef, xf, scale, args.n_eval, nF, device, n_rk4=args.n_rk4)

    # Subsample to 32x32 for spectrum eval
    stride = G // 32
    gen_R = gen.view(args.n_eval, G, G)[:, ::stride, ::stride].cpu().numpy()
    tru_R = test_data[:, ::stride, ::stride].numpy()
    _, sg = get_fourier_spectrum(gen_R)
    _, st = get_fourier_spectrum(tru_R)
    rel = np.abs(sg - st) / (np.abs(st) + 1e-30)
    print(f"Per-bin at R=32 (Nyquist k=16):")
    print(f"  k=01..16: " + " ".join(f"{r:.3f}" for r in rel[:16]))
    print(f"  mean(k=1..15)={rel[:15].mean():.4f}  max={rel[:15].max():.4f}  argmax k={1+int(np.argmax(rel[:15]))}")
    print(f"  k=16 (Nyquist): {rel[15]:.3f}")


if __name__ == '__main__':
    main()
