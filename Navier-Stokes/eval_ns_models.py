"""
Re-evaluate trained NS models with t_min/t_max = 1e-3 / 1-1e-3.
"""
import os, sys, math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import scipy.stats as stats

from unet import Unet

def load_ns_data(data_locs, hi_size, train_test_split):
    if isinstance(data_locs, str):
        data_locs = [data_locs]
    avg_pixel_norm = 3.0679163932800293
    all_data = []
    for loc in data_locs:
        data_raw, _ = torch.load(loc)
        data_raw = data_raw / avg_pixel_norm
        data = data_raw.reshape(-1, data_raw.shape[2], data_raw.shape[3])
        if hi_size != data.shape[1]:
            data = nn.functional.interpolate(data.unsqueeze(1), size=(hi_size, hi_size), mode='bilinear').squeeze(1)
        all_data.append(data)
    data = torch.cat(all_data, dim=0)[:, None, :, :]
    num_train = int(data.shape[0] * train_test_split)
    return data[:num_train], data[num_train:]

def get_energy_spectrum(data):
    if isinstance(data, torch.Tensor):
        data = data.cpu()
    fhat = torch.fft.fftn(data, dim=(1, 2), norm='forward')
    fourier_amp = (torch.abs(fhat)**2 * (2 * np.pi)).mean(dim=0)
    npix = data.shape[-1]
    kfreq = np.fft.fftfreq(npix) * npix
    kx, ky = np.meshgrid(kfreq, kfreq)
    knrm = np.sqrt(kx**2 + ky**2).flatten()
    fourier_flat = fourier_amp.numpy().flatten()
    laplace = knrm**2; laplace[0] = 1.0
    energy_flat = fourier_flat / laplace
    kbins = np.arange(0.5, npix // 2 + 1, 1.)
    kvals = 0.5 * (kbins[1:] + kbins[:-1])
    area_weight = np.pi * (kbins[1:]**2 - kbins[:-1]**2)
    enstrophy, _, _ = stats.binned_statistic(knrm, fourier_flat, statistic='mean', bins=kbins)
    enstrophy *= area_weight
    energy, _, _ = stats.binned_statistic(knrm, energy_flat, statistic='mean', bins=kbins)
    energy *= area_weight
    return kvals, enstrophy, energy

def make_data_dependent_noise(next_batch, batch_size):
    N = next_batch.shape[0]
    centered = next_batch - next_batch.mean(dim=0, keepdim=True)
    xi = torch.randn(N, batch_size, device=next_batch.device)
    noise = torch.einsum('nchw,nb->bchw', centered, xi) / math.sqrt(N)
    return noise

class Velocity(nn.Module):
    def __init__(self, C, unet_channels, unet_dim_mults, **kw):
        super().__init__()
        self.net = Unet(
            num_classes=1, in_channels=C, out_channels=C,
            dim=unet_channels, dim_mults=unet_dim_mults,
            resnet_block_groups=kw.get('resnet_block_groups', 8),
            learned_sinusoidal_cond=kw.get('learned_sinusoidal_cond', True),
            random_fourier_features=kw.get('random_fourier_features', False),
            learned_sinusoidal_dim=kw.get('learned_sinusoidal_dim', 32),
            attn_dim_head=kw.get('attn_dim_head', 32),
            attn_heads=kw.get('attn_heads', 4),
            use_classes=False,
        )
    def forward(self, zt, t):
        return self.net(zt, t, classes=None)

@torch.no_grad()
def rk4_sample(model, z0, steps, t_min=1e-3, t_max=1-1e-3):
    tgrid = torch.linspace(t_min, t_max, steps).type_as(z0)
    zt = z0
    for i in range(len(tgrid) - 1):
        t_val = tgrid[i]
        dt = tgrid[i+1] - tgrid[i]
        ones = torch.ones(zt.shape[0], device=zt.device) * t_val
        k1 = model(zt, ones)
        k2 = model(zt + 0.5*dt*k1, ones + 0.5*dt)
        k3 = model(zt + 0.5*dt*k2, ones + 0.5*dt)
        k4 = model(zt + dt*k3, ones + dt)
        zt = zt + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
    return zt

def eval_model(model, train_data, test_data, device, noise_type, hi_size,
               step_counts=[2, 5, 10, 20, 50], t_min=1e-3, t_max=1-1e-3):
    model.eval()
    num_eval = min(200, test_data.shape[0])
    truth_sq = test_data[:num_eval].squeeze(1)
    kvals, enst_truth, ener_truth = get_energy_spectrum(truth_sq)

    if noise_type == 'gauss':
        z0 = torch.randn(num_eval, 1, hi_size, hi_size, device=device)
    else:
        # Each chunk uses a fresh random batch to avoid rank deficiency
        batch_size = 100
        z0_chunks = []
        remaining = num_eval
        while remaining > 0:
            chunk = min(batch_size, remaining)
            perm = torch.randperm(train_data.shape[0])
            noise_src = train_data[perm[:batch_size]].to(device)
            z0_chunks.append(make_data_dependent_noise(noise_src, chunk))
            remaining -= chunk
        z0 = torch.cat(z0_chunks, dim=0)

    bands = {'low': kvals < 8, 'mid': (kvals >= 8) & (kvals < 24), 'high': kvals >= 24}

    for nsteps in step_counts:
        gen = rk4_sample(model, z0.clone(), nsteps, t_min, t_max)
        gen_sq = gen.squeeze(1).cpu()
        _, enst_gen, ener_gen = get_energy_spectrum(gen_sq)
        std_ratio = gen_sq.std().item() / (truth_sq.std().item() + 1e-12)

        band_str = []
        for bname, mask in bands.items():
            enst_err = np.mean(np.abs(enst_truth[mask] - enst_gen[mask]) / (np.abs(enst_truth[mask]) + 1e-12))
            ener_err = np.mean(np.abs(ener_truth[mask] - ener_gen[mask]) / (np.abs(ener_truth[mask]) + 1e-12))
            band_str.append(f"{bname}: enst={enst_err:.4f} ener={ener_err:.4f}")
        print(f"  RK4 steps={nsteps:3d}:  std_ratio={std_ratio:.4f}  {'  '.join(band_str)}")


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--t_min', type=float, default=1e-3)
    p.add_argument('--t_max', type=float, default=0.999)
    args = p.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda')

    data_loc = '../NSdata/data_file.pt'
    train_data, test_data = load_ns_data(data_loc, 128, 0.9)
    print(f"Data: train={train_data.shape[0]}, test={test_data.shape[0]}")

    configs = [
        ("Gauss-base 2M",   "gauss",    "results/ns_gauss_base/model_final.pt",        32, (1,2,2,2)),
        ("DataDep 2M",      "data_dep", "results/ns_data_dep_noise/model_final.pt",     32, (1,2,2,2)),
        ("Gauss-base 18M",  "gauss",    "results/ns_gauss_base_large/model_final.pt",   64, (1,2,4,4)),
        ("DataDep 18M",     "data_dep", "results/ns_data_dep_noise_large/model_final.pt", 64, (1,2,4,4)),
    ]

    for name, noise_type, ckpt, channels, dim_mults in configs:
        if not os.path.exists(ckpt):
            print(f"\n=== {name}: SKIPPED (no checkpoint) ===")
            continue
        print(f"\n=== {name} (t_min={args.t_min}, t_max={args.t_max}) ===")
        model = Velocity(C=1, unet_channels=channels, unet_dim_mults=dim_mults).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        eval_model(model, train_data, test_data, device, noise_type, 128,
                   t_min=args.t_min, t_max=args.t_max)
