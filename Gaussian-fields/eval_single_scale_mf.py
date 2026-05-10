"""Eval single-scale (1-mask) meanflow / FM checkpoints with mean_rel<=30 metric.
Loads model_final.pt from gf_meanflow_gauss / gf_meanflow_data_dep / gf_baseline_data_dep.
"""
import os, sys, math, argparse
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Navier-Stokes'))
from unet import Unet, RandomOrLearnedSinusoidalPosEmb
from train_multiscale_interpolation import (
    precompute_matern_amplitude, sample_matern_batch, get_fourier_spectrum,
)


class MeanFlowVelocity(nn.Module):
    def __init__(self, ch=32, dim_mults=(1, 2, 2, 2)):
        super().__init__()
        self.net = Unet(num_classes=1, in_channels=1, out_channels=1,
                        dim=ch, dim_mults=dim_mults, resnet_block_groups=8,
                        learned_sinusoidal_cond=True, random_fourier_features=False,
                        learned_sinusoidal_dim=32, attn_dim_head=32, attn_heads=4,
                        use_classes=False)
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


def gen_mf(net, n_samples, G, device, n_mf, noise_scale=1.0, sigma_data=None):
    """Single-scale mean flow generation."""
    if sigma_data is None: sigma_data = 1.0
    zt = noise_scale * torch.randn(n_samples, 1, G, G, device=device)
    s_vals = torch.linspace(0.0, 1.0, n_mf + 1)
    for j in range(n_mf):
        s_j = float(s_vals[j]); r_j = float(s_vals[j + 1])
        sb = torch.full((n_samples,), s_j, device=device)
        rb = torch.full((n_samples,), r_j, device=device)
        with torch.no_grad():
            v = net(zt, sb, rb)
        zt = zt + (r_j - s_j) * v
    return zt.squeeze(1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=int, default=2)
    p.add_argument('--ckpt', type=str, required=True)
    p.add_argument('--G', type=int, default=64)
    p.add_argument('--s', type=float, default=3.0)
    p.add_argument('--n_eval', type=int, default=2000)
    p.add_argument('--noise', type=str, default='gauss', choices=['gauss', 'data_dep'])
    args = p.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda')
    G = args.G

    sigma_sq = 1.0 * ((2 * math.pi)**2 + 1.0**2)**args.s
    amp = precompute_matern_amplitude(G, sigma_sq, 1.0, args.s).float()
    test_data = sample_matern_batch(amp, args.n_eval, device='cpu').float()
    truth_np = test_data.numpy()
    kvals, spec_truth = get_fourier_spectrum(truth_np)

    # Default Config: ch=32, dm=(1,2,2,2)
    net = MeanFlowVelocity(ch=32, dim_mults=(1, 2, 2, 2)).float().to(device)
    sd = torch.load(args.ckpt, map_location=device, weights_only=True)
    net.load_state_dict(sd)
    net.eval()

    print(f"\n=== {args.ckpt} ===")
    for n_mf in [1, 2, 4, 8, 16, 32]:
        gen = gen_mf(net, args.n_eval, G, device, n_mf, noise_scale=1.0)
        gen_np = gen.cpu().numpy()
        _, spec_gen = get_fourier_spectrum(gen_np)
        rel = np.abs(spec_gen - spec_truth) / (np.abs(spec_truth) + 1e-30)
        mean30 = rel[:30].mean()
        max30 = rel[:30].max()
        nb1 = int((rel[:30] < 1).sum())
        nb01 = int((rel[:30] < 0.1).sum())
        nyq = rel[30] if len(rel) > 30 else float('nan')
        print(f"  MF_steps={n_mf:2d}  NFE={n_mf:3d}  mean_rel<=30={mean30:.4f}  max_rel<=30={max30:.4f}  bins<1={nb1}/30  bins<0.1={nb01}/30  k=31:{nyq:.3f}")


if __name__ == '__main__':
    main()
