"""Single-scale (1-mask) SHORTCUT trainer for Gaussian field at G=64.

Same network architecture as train_gaussian_field_meanflow.py (MeanFlowVelocity),
but with the bootstrap-based shortcut self-consistency loss (Frans et al. 2024)
instead of JVP. Designed to test whether shortcut at single-scale + Gaussian noise
can match the multiscale shortcut accuracy on Gaussian fields when given enough
training (the previous 50k-step single-scale meanflow checkpoint gave
mean_rel_30 ~ 6-11, suggesting under-training).

Usage:
  python train_gaussian_field_shortcut.py --gpu 1 --max_steps 250000 \
    --noise gauss --batch_size 200 --lr 2e-4 --flow_ratio 0.5 --grad_clip 1.0 \
    --ema_decay 0.9999 --save_dir results/gf_shortcut_gauss_long
"""
import os, sys, math, copy, datetime, argparse
import numpy as np
import torch
import torch.nn as nn
from time import time as timer
import wandb

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Navier-Stokes'))
from unet import Unet, RandomOrLearnedSinusoidalPosEmb
from train_multiscale_interpolation import (
    precompute_matern_amplitude, sample_matern_batch, get_fourier_spectrum,
)


class MeanFlowVelocity(nn.Module):
    """Identical to the GF meanflow trainer's network."""
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


class EMA:
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def copy_to(self, model):
        for n, p in model.named_parameters():
            if p.requires_grad: p.data.copy_(self.shadow[n])


def make_data_dep_z0(prev_batch, B):
    """Empirical-covariance noise from previous batch: z0_j = (1/sqrt(N)) sum_i (x_i - x_bar) xi_{ij}."""
    N = prev_batch.shape[0]; x_bar = prev_batch.mean(0, keepdim=True)
    x_centered = prev_batch - x_bar  # (N, G, G)
    xi = torch.randn(N, B, device=prev_batch.device)  # (N, B)
    z0 = torch.einsum('nhw,nb->bhw', x_centered, xi) / math.sqrt(N)
    return z0


def shortcut_loss(net, zt, target_v, s, r, sigma2):
    """Shortcut self-consistency. zt: (B,1,G,G). s,r: (B,)."""
    w_full = net(zt, s, r)
    m = (s + r) / 2
    with torch.no_grad():
        v_first = net(zt, s, m)
        d_first = (m - s)[:, None, None, None]
        zm = zt + d_first * v_first
        v_second = net(zm, m, r)
    target_shortcut = (v_first + v_second) / 2
    is_fm = (r == s)[:, None, None, None].float()
    target_use = is_fm * target_v + (1.0 - is_fm) * target_shortcut
    return ((w_full - target_use.detach())**2).mean() / sigma2


def gen_mf(net, n_samples, G, device, n_mf, noise_scale=1.0):
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
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--max_steps', type=int, default=250000)
    p.add_argument('--noise', type=str, default='gauss', choices=['gauss', 'data_dep'])
    p.add_argument('--batch_size', type=int, default=200)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--flow_ratio', type=float, default=0.5,
                   help='fraction of steps with r=s (pure FM anchor)')
    p.add_argument('--grad_clip', type=float, default=1.0)
    p.add_argument('--ema_decay', type=float, default=0.9999)
    p.add_argument('--grid_size', type=int, default=64)
    p.add_argument('--s1', type=float, default=3.0)
    p.add_argument('--ls1', type=float, default=1.0)
    p.add_argument('--save_dir', type=str, default='results/gf_shortcut_gauss_long')
    p.add_argument('--test_every', type=int, default=5000)
    p.add_argument('--no_wandb', action='store_true')
    args = p.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda')
    G = args.grid_size

    # Data
    sigma_sq = 1.0 * ((2 * math.pi)**2 + args.ls1**2)**args.s1
    amp = precompute_matern_amplitude(G, sigma_sq, args.ls1, args.s1).float()
    test_data = sample_matern_batch(amp, 500, device='cpu').float()
    truth_np = test_data.numpy()
    kvals, spec_truth = get_fourier_spectrum(truth_np)

    # σ² for loss normalisation: use the data variance (single global scale)
    sigma2 = float(test_data.var())
    print(f'[Data] truth std = {test_data.std():.4f}, sigma2 = {sigma2:.4f}')

    os.makedirs(args.save_dir, exist_ok=True)

    # Network
    net = MeanFlowVelocity(ch=32, dim_mults=(1, 2, 2, 2)).float().to(device)
    npar = sum(p.numel() for p in net.parameters())
    print(f'[Network] {npar:,} params')

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.max_steps)
    ema = EMA(net, decay=args.ema_decay) if args.ema_decay > 0 else None

    # Wandb
    run_name = f'gf_shortcut_{args.noise}_lr{args.lr}_steps{args.max_steps}'
    if not args.no_wandb:
        wandb.init(project='interpolants-design', entity='yifanc96', name=run_name)
        wandb.config.update(vars(args))

    prev_batch = None
    t0 = timer()
    for step in range(1, args.max_steps + 1):
        net.train()
        x1 = sample_matern_batch(amp, args.batch_size, device=device).float()
        if args.noise == 'gauss' or prev_batch is None:
            z0 = torch.randn(args.batch_size, G, G, device=device)
        else:
            z0 = make_data_dep_z0(prev_batch, args.batch_size)
        prev_batch = x1.detach().clone()

        # Sample (s, r) with flow_ratio probability of r = s
        t1 = torch.rand(args.batch_size, device=device) * 0.998 + 0.001
        t2 = torch.rand(args.batch_size, device=device) * 0.998 + 0.001
        s = torch.minimum(t1, t2); r = torch.maximum(t1, t2)
        flow_mask = torch.rand(args.batch_size, device=device) < args.flow_ratio
        r = torch.where(flow_mask, s, r)

        # Interpolant and target
        sw = s[:, None, None]
        zt = (1 - sw) * z0 + sw * x1  # (B, G, G)
        target_v = (-z0 + x1).unsqueeze(1)  # (B, 1, G, G)
        zt = zt.unsqueeze(1)

        loss = shortcut_loss(net, zt, target_v, s, r, sigma2)

        opt.zero_grad(); loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip).item()
        opt.step(); sch.step()
        if ema is not None: ema.update(net)

        if step % 100 == 0 or step == 1:
            el = (timer() - t0) / 60
            lr_now = sch.get_last_lr()[0]
            print(f'  step {step}/{args.max_steps}  loss={loss.item():.4f}  grad={gn:.2f}  lr={lr_now:.2e}  [{el:.1f}m]', flush=True)
            if not args.no_wandb:
                wandb.log({'loss': loss.item(), 'grad': gn, 'lr': lr_now}, step=step)

        if step % args.test_every == 0 or step == args.max_steps:
            # eval with EMA weights
            net.eval()
            backup = None
            if ema is not None:
                backup = {n: p.detach().clone() for n, p in net.named_parameters() if p.requires_grad}
                ema.copy_to(net)
            best_metric = float('inf'); best_metrics = {}
            for n_mf in [1, 2, 4, 8, 16]:
                gen = gen_mf(net, 500, G, device, n_mf)
                _, sg = get_fourier_spectrum(gen.cpu().numpy())
                rel = np.abs(sg - spec_truth) / (np.abs(spec_truth) + 1e-30)
                mr = float(rel[:30].mean()); best_metrics[n_mf] = mr
                if mr < best_metric: best_metric = mr
                if not args.no_wandb:
                    wandb.log({f'eval/mean_rel_30_MF{n_mf}': mr}, step=step)
            print(f'  [eval step {step}] mean_rel<=30 by MF steps: ' + '  '.join(f'MF{k}={v:.4f}' for k, v in best_metrics.items()), flush=True)
            if backup is not None:
                for n, p in net.named_parameters():
                    if p.requires_grad: p.data.copy_(backup[n])

    # Save final (with EMA weights)
    if ema is not None: ema.copy_to(net)
    torch.save(net.state_dict(), os.path.join(args.save_dir, 'model_final.pt'))
    print(f'\nWrote {args.save_dir}/model_final.pt  total {(timer()-t0)/60:.1f} min')

    if not args.no_wandb: wandb.finish()


if __name__ == '__main__':
    main()
