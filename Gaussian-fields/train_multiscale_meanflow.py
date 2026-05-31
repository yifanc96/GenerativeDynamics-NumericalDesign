"""
Multiscale mean flow for Gaussian fields.

Follows the EXACT same training pattern as train_multiscale_interpolation.py:
  - All scales trained JOINTLY (not sequentially)
  - Per-pixel loss_weight balances scale contributions
  - Mixed batch: each sample assigned to a random scale via uniform t

The ONLY change: velocity matching loss → JVP mean flow loss.
This enables few-step (1-2 per phase) generation instead of many ODE steps.
"""

import os, sys, math, datetime, argparse, copy
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
import wandb
from time import time as timer
import torch.fft as torch_fft

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Navier-Stokes'))
from unet import Unet, RandomOrLearnedSinusoidalPosEmb

from train_multiscale_interpolation import (
    precompute_matern_amplitude,
    sample_matern_batch,
    get_fourier_spectrum,
    HierarchicalMasks,
    MultiscaleInterpolant,
    estimate_conditional_variances,
    build_noise_std,
    make_unet,
    Sampler as ODESampler,  # keep for baseline comparison
)


# ─── Per-scale mean flow UNet ───────────────────────────────────────────────

class PerScaleMFNet(nn.Module):
    """Per-scale UNet with (s_local, r_local) conditioning. Input: 2ch."""
    def __init__(self, channels=16, dim_mults=(1, 2, 2)):
        super().__init__()
        lsd = max(channels, 16)
        self.net = make_unet(channels, dim_mults, in_ch=2, out_ch=1)
        time_dim = channels * 4
        sinu = RandomOrLearnedSinusoidalPosEmb(lsd, is_random=False)
        fourier_dim = lsd + 1
        self.r_mlp = nn.Sequential(
            sinu, nn.Linear(fourier_dim, time_dim), nn.GELU(), nn.Linear(time_dim, time_dim),
        )
        nn.init.zeros_(self.r_mlp[-1].weight)
        nn.init.zeros_(self.r_mlp[-1].bias)

    def forward(self, x_2ch, s_local, r_local):
        net = self.net
        s_emb = net.time_mlp(s_local)
        r_emb = self.r_mlp(r_local)
        t_emb = s_emb + r_emb
        x = net.init_conv(x_2ch)
        r_skip = x.clone()
        h = []
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


class MFMultiVelocity(nn.Module):
    """Multi-net mean flow velocity: one PerScaleMFNet per scale.
    Training forward matches MultiVelocity pattern — routes samples by scale.
    """
    def __init__(self, hier_masks, channels_per_net=16, dim_mults=(1, 2, 2)):
        super().__init__()
        self.hier = hier_masks
        self.num_masks = hier_masks.num_masks
        self.nets = nn.ModuleList([
            PerScaleMFNet(channels=channels_per_net, dim_mults=dim_mults)
            for _ in range(self.num_masks)
        ])
        n = sum(p.numel() for p in self.parameters())
        print(f"[MFMultiVelocity] {self.num_masks} nets, {n:,} total params")

    def forward_for_sampling(self, zt, t_scalar, s_local_scalar, r_local_scalar):
        """Inference path: all samples at same global time."""
        B = zt.shape[0]
        k = self.hier.get_active_scale(t_scalar)
        s_batch = torch.full((B,), s_local_scalar, device=zt.device)
        r_batch = torch.full((B,), r_local_scalar, device=zt.device)
        coarse_ctx = self.hier.get_coarse_context_scalar_t(zt, t_scalar)
        inp = torch.cat([zt, coarse_ctx], dim=1)
        raw = self.nets[k](inp, s_batch, r_batch)
        bdot = self.hier.beta_dot_scalar_t(t_scalar, B)
        return bdot * raw


# ─── JVP per scale group ────────────────────────────────────────────────────

def compute_jvp_for_scale(net_k, zt_sub, coarse_sub, s_sub, r_sub, v_sub):
    """JVP for a subset of samples all belonging to the same scale."""
    def fn(z_, s_, r_):
        inp = torch.cat([z_, coarse_sub], dim=1)
        return net_k(inp, s_, r_)
    primals = (zt_sub, s_sub, r_sub)
    tangents = (v_sub, torch.ones_like(s_sub), torch.zeros_like(r_sub))
    return torch.func.jvp(fn, primals, tangents)


# ─── EMA ─────────────────────────────────────────────────────────────────────

class EMA:
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.ema_model = copy.deepcopy(model).eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for ema_p, p in zip(self.ema_model.parameters(), model.parameters()):
            ema_p.mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)
        for ema_b, b in zip(self.ema_model.buffers(), model.buffers()):
            ema_b.copy_(b)


# ─── Mean flow sampler ──────────────────────────────────────────────────────

class MFSampler:
    def __init__(self, num_masks):
        self.num_masks = num_masks

    @torch.no_grad()
    def sample(self, z0, model, hier, steps_per_phase=1):
        zt = z0
        for k in range(self.num_masks):
            s_vals = torch.linspace(0.0, 1.0, steps_per_phase + 1)
            for j in range(steps_per_phase):
                s_j = float(s_vals[j])
                r_j = float(s_vals[j + 1])
                t_global = k + s_j
                zt = zt + (r_j - s_j) * model.forward_for_sampling(zt, t_global, s_j, r_j)
        return zt


# ─── Trainer (follows train_multiscale_interpolation.py pattern) ─────────────

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        cfg = config
        sigma_sq = cfg.sigma_sq * ((2 * math.pi)**2 + cfg.ls**2)**cfg.s
        self.amplitude = precompute_matern_amplitude(cfg.grid_size, sigma_sq, cfg.ls, cfg.s)
        self.test_data = sample_matern_batch(self.amplitude, 500, device='cpu')
        print(f"[Data] Matern grid={cfg.grid_size}, s={cfg.s}, ls={cfg.ls}, test std={self.test_data.std():.4f}")

        self.hier = HierarchicalMasks(cfg.grid_size, cfg.num_masks, device=self.device)
        var_data = sample_matern_batch(self.amplitude, 2000, device='cpu')
        cond_vars = estimate_conditional_variances(var_data, [m.cpu() for m in self.hier.masks])
        scaled_vars = [v * cfg.noise_scale for v in cond_vars]
        for i, (raw, sc) in enumerate(zip(cond_vars, scaled_vars)):
            n_pts = int(self.hier.masks[i].sum().item())
            print(f"  scale {i} ({n_pts:5d} pts): cond_var={raw:.8f}  noise_std={sc**0.5:.6f}")
        self.noise_std = build_noise_std(self.hier.masks, scaled_vars, device=self.device)

        # Per-pixel loss weight (SAME as original: equalizes scale contributions)
        loss_weight = torch.zeros(cfg.grid_size, cfg.grid_size, device=self.device)
        for i, m in enumerate(self.hier.masks):
            n_k = m.sum().item()
            w_k = 1.0 / (n_k * scaled_vars[i])
            loss_weight += w_k * m
        loss_weight = loss_weight / loss_weight.mean()
        self.loss_weight = loss_weight[None, None, :, :]
        for i, m in enumerate(self.hier.masks):
            n_k = int(m.sum().item())
            w_i = (loss_weight[0, 0] * m).sum().item() / n_k if n_k > 0 else 0
            print(f"  scale {i} ({n_k:5d} pts): per-pixel loss weight={w_i:.4f}")
        del var_data

        self.interpolant = MultiscaleInterpolant(self.hier)
        self.mf_sampler = MFSampler(cfg.num_masks)
        self.ode_sampler = ODESampler(cfg.num_masks)  # for baseline comparison

        self.model = MFMultiVelocity(
            self.hier, channels_per_net=cfg.multi_channels, dim_mults=cfg.multi_dim_mults
        ).to(self.device)
        self.ema = EMA(self.model, decay=cfg.ema_decay)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.base_lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=cfg.max_steps, eta_min=cfg.base_lr * 0.01)
        self.time_dist = torch.distributions.Uniform(low=cfg.t_min_train, high=cfg.t_max_train)
        self.global_step = 0
        self.best_metric = float('inf')

    def make_noise_iid(self, batch_size):
        G = self.config.grid_size
        z = torch.randn(batch_size, 1, G, G, device=self.device)
        return z * self.noise_std[None, None, :, :]

    def train_step(self):
        """
        Joint training across all scales (matches original pattern).
        Each sample in the batch is assigned to a random scale via t.
        """
        cfg = self.config
        batch = sample_matern_batch(self.amplitude, cfg.batch_size, device=self.device)
        z0 = self.make_noise_iid(cfg.batch_size)
        z1 = batch.unsqueeze(1)

        # Sample global time (each sample gets a random scale)
        t_global = cfg.num_masks * self.time_dist.sample((cfg.batch_size,)).to(self.device)

        # Determine which scale each sample belongs to
        scale_idx = t_global.int().clamp(0, cfg.num_masks - 1)  # (B,)
        s_local = (t_global - scale_idx.float()).clamp(0, 1)     # local time within phase

        # Sample r_local for mean flow (r >= s within [0,1])
        r_raw = torch.rand(cfg.batch_size, device=self.device)
        # r_local uniform in [s_local, 1]
        r_local = s_local + (1 - s_local) * r_raw
        # flow_ratio: fraction uses r = s (pure flow matching)
        flow_mask = torch.rand(cfg.batch_size, device=self.device) < cfg.flow_ratio
        r_local = torch.where(flow_mask, s_local, r_local)

        # Interpolate at global time
        zt = self.interpolant.It(z0, z1, t_global)
        target = self.interpolant.Rt(z0, z1, t_global)  # alpha_dot*z0 + beta_dot*z1
        coarse_ctx = self.hier.get_coarse_context(zt, t_global)

        # Compute velocity tangent per pixel
        v_tangent = target.clone()  # = alpha_dot*z0 + beta_dot*z1 = dz_t/dt

        # Process each scale separately (like MultiVelocity.forward)
        w_all = torch.zeros_like(zt)
        dw_all = torch.zeros_like(zt)

        for k in range(cfg.num_masks):
            active = (scale_idx == k)
            if not active.any():
                continue
            idx = active.nonzero(as_tuple=True)[0]

            w_k, dw_k = compute_jvp_for_scale(
                self.model.nets[k],
                zt[idx], coarse_ctx[idx],
                s_local[idx], r_local[idx],
                v_tangent[idx],
            )
            w_all[idx] = w_k
            dw_all[idx] = dw_k

        # Apply beta_dot mask (only active scale pixels contribute)
        bdot = self.hier.beta_dot(t_global)[:, None, :, :]
        w_masked = bdot * w_all
        dw_masked = bdot * dw_all

        # Mean flow target
        dr = (r_local - s_local)[:, None, None, None]
        w_tgt = target + dr * dw_masked

        error = w_masked - w_tgt.detach()

        # Weighted loss (SAME as original)
        loss = (self.loss_weight * error.pow(2)).sum(dim=(1, 2, 3)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=cfg.grad_clip)
        self.optimizer.step()
        self.scheduler.step()
        self.ema.update(self.model)
        return loss.item(), grad_norm.item()

    def fit(self):
        cfg = self.config
        t0 = timer()
        print(f"[Training] JOINT, max_steps={cfg.max_steps}, batch={cfg.batch_size}")
        print(f"[Training] flow_ratio={cfg.flow_ratio}, grad_clip={cfg.grad_clip}")
        self.test_model()

        while self.global_step < cfg.max_steps:
            loss, gn = self.train_step()
            if self.global_step % cfg.print_loss_every == 0:
                lr = self.scheduler.get_last_lr()[0]
                print(f"  step {self.global_step}/{cfg.max_steps}  loss={loss:.4f}  grad={gn:.2f}  lr={lr:.2e}  [{(timer()-t0)/60:.1f}m]")
                if cfg.use_wandb:
                    wandb.log({"loss": loss, "grad_norm": gn, "lr": lr}, step=self.global_step)
            if self.global_step > 0 and self.global_step % cfg.test_every == 0:
                self.test_model()
            self.global_step += 1

        print("[Training] Done.")
        self.test_model()
        os.makedirs(cfg.save_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(cfg.save_dir, 'model_final.pt'))

    @torch.no_grad()
    def test_model(self):
        cfg = self.config
        self.model.eval()
        eval_model = self.ema.ema_model
        eval_model.eval()

        num_eval = cfg.num_eval
        truth_np = self.test_data[:num_eval].numpy()
        kvals, spec_truth = get_fourier_spectrum(truth_np)
        z0 = self.make_noise_iid(num_eval)

        results = {}

        # Mean flow sampling at various steps-per-phase
        for spp in cfg.eval_steps_per_phase:
            gen = self.mf_sampler.sample(z0.clone(), eval_model, self.hier, steps_per_phase=spp)
            gen_np = gen.squeeze(1).cpu().numpy()
            _, spec_gen = get_fourier_spectrum(gen_np)
            nfe = spp * cfg.num_masks
            std_ratio = gen_np.std() / (truth_np.std() + 1e-12)
            spec_rel = np.mean(np.abs(spec_truth - spec_gen) / (np.abs(spec_truth) + 1e-12))
            spec_l1 = np.mean(np.abs(spec_truth - spec_gen))
            tag = f"MF_spp{spp}"
            results[tag] = dict(spec_gen=spec_gen, gen_np=gen_np,
                                std_ratio=std_ratio, spec_rel=spec_rel, spec_l1=spec_l1)
            print(f"    {tag} (NFE={nfe}):  spec_relErr={spec_rel:.4f}  spec_L1={spec_l1:.6f}  std_ratio={std_ratio:.4f}")
            if cfg.use_wandb:
                wandb.log({f"spec_relErr/{tag}": spec_rel, f"spec_L1/{tag}": spec_l1,
                           f"std_ratio/{tag}": std_ratio}, step=self.global_step)

        # Best checkpoint tracking (use spp=2 as primary)
        primary_spp = cfg.eval_steps_per_phase[1] if len(cfg.eval_steps_per_phase) > 1 else cfg.eval_steps_per_phase[0]
        primary = results[f"MF_spp{primary_spp}"]['spec_l1']
        if primary < self.best_metric:
            self.best_metric = primary
            os.makedirs(cfg.save_dir, exist_ok=True)
            torch.save(eval_model.state_dict(), os.path.join(cfg.save_dir, 'model_best.pt'))
            print(f"    [BEST] step={self.global_step}  L1={primary:.6f}")
            if cfg.use_wandb:
                wandb.log({"best/L1": primary, "best/step": self.global_step}, step=self.global_step)

        # Spectrum plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        ax = axes[0]
        ax.loglog(kvals, spec_truth, 'k-', lw=2, label='truth')
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(cfg.eval_steps_per_phase)))
        for i, spp in enumerate(cfg.eval_steps_per_phase):
            tag = f"MF_spp{spp}"
            nfe = spp * cfg.num_masks
            ax.loglog(kvals, results[tag]['spec_gen'], '--', color=colors[i], label=f'spp={spp} ({nfe} NFE)')
        ax.set_xlabel('k'); ax.set_ylabel('E(k)'); ax.legend(fontsize=6); ax.set_title('Energy Spectrum')

        spp0 = cfg.eval_steps_per_phase[0]
        sppN = cfg.eval_steps_per_phase[-1]
        vmax = max(abs(truth_np[0].min()), abs(truth_np[0].max()))
        combined = np.concatenate([truth_np[0], results[f'MF_spp{spp0}']['gen_np'][0],
                                   results[f'MF_spp{sppN}']['gen_np'][0]], axis=1)
        axes[1].imshow(combined, cmap='RdBu', vmin=-vmax, vmax=vmax)
        axes[1].set_title(f'Truth | spp={spp0} | spp={sppN}'); axes[1].axis('off')
        plt.tight_layout()
        if cfg.use_wandb:
            wandb.log({"spectrum_comparison": wandb.Image(fig)}, step=self.global_step)
        plt.close()


# ─── Config ──────────────────────────────────────────────────────────────────

class Config:
    def __init__(self):
        self.use_wandb = True
        self.wandb_project = 'interpolants-design'
        self.wandb_entity = 'yifanc96'

        self.grid_size = 64
        self.sigma_sq = 1.0
        self.ls = 1.0
        self.s = 3.0
        self.batch_size = 200

        self.num_masks = 4
        self.noise_scale = 1.0

        # training (matches original)
        self.base_lr = 2e-4
        self.max_steps = 50000
        self.t_min_train = 1e-3
        self.t_max_train = 1.0 - 1e-3
        self.print_loss_every = 50
        self.test_every = 2500
        self.num_eval = 200
        self.eval_steps_per_phase = [1, 2, 4]

        # mean-flow specific
        self.flow_ratio = 0.5
        self.grad_clip = 1.0
        self.ema_decay = 0.9999

        self.multi_channels = 16
        self.multi_dim_mults = (1, 2, 2)

        self.save_dir = 'results/multiscale_meanflow'


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--grid_size', type=int, default=64)
    p.add_argument('--batch_size', type=int, default=200)
    p.add_argument('--max_steps', type=int, default=50000)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--s', type=float, default=3.0)
    p.add_argument('--ls', type=float, default=1.0)
    p.add_argument('--num_masks', type=int, default=4)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--test_every', type=int, default=2500)
    p.add_argument('--save_dir', type=str, default='results/multiscale_meanflow')
    p.add_argument('--no_wandb', action='store_true')
    p.add_argument('--multi_channels', type=int, default=16)
    p.add_argument('--flow_ratio', type=float, default=0.5)
    p.add_argument('--noise_scale', type=float, default=1.0)
    args = p.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    config = Config()
    config.grid_size = args.grid_size
    config.batch_size = args.batch_size
    config.max_steps = args.max_steps
    config.base_lr = args.lr
    config.s = args.s
    config.ls = args.ls
    config.num_masks = args.num_masks
    config.test_every = args.test_every
    config.save_dir = args.save_dir
    config.use_wandb = not args.no_wandb
    config.multi_channels = args.multi_channels
    config.flow_ratio = args.flow_ratio
    config.noise_scale = args.noise_scale
    config.num_eval = 200

    os.makedirs(config.save_dir, exist_ok=True)

    date = str(datetime.datetime.now())
    log_base = date[date.find("-"):date.rfind(".")].replace("-", "").replace(":", "").replace(" ", "_")
    run_name = f"GF_ms_mf_masks{config.num_masks}_s{config.s}_{log_base}"

    if config.use_wandb:
        wandb.init(project=config.wandb_project, entity=config.wandb_entity, name=run_name)
        for k, v in vars(config).items():
            if isinstance(v, (int, float, bool, str, list, tuple)):
                setattr(wandb.config, k, v)

    trainer = Trainer(config)
    trainer.fit()

    if config.use_wandb:
        wandb.finish()
