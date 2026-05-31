"""
Streamlined retraining of the Navier-Stokes UNet under the matched-spectrum
prior, on the available local data files.

Faithful to NSunconditional-Gaussbase-spectrumnoise-matchingvar.py (no W&B,
no checkpoint hierarchy, just a clean training loop). After training we apply
both the linear schedule and the designed scale-adaptive schedule via the
transfer formula, and we compute the resulting enstrophy spectrum.

Targets the rebuttal question R1.Q3: what happens when the scale-adaptive
schedule is applied to the spectrum-noise prior on Navier-Stokes?

Run with:
    /home/yifanchen/miniconda3/envs/gpu/bin/python train_ns_spectrum_noise.py \
        --max_steps 30000 --hi 128 --batch 100
"""

import argparse
import math
import os
import sys
import time as _t

import numpy as np
import torch
import torch.nn as nn
import scipy.stats as stats
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from unet import Unet  # noqa

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
HOME = os.path.dirname(os.path.abspath(__file__))
DATA_LOCS = [
    '/home/yifanchen/research/GenerativeDynamics-NumericalDesign/NSdata/data_file.pt',
    '/home/yifanchen/research/GenerativeDynamics-NumericalDesign/NSdata/data_file02.pt',
    '/home/yifanchen/research/GenerativeDynamics-NumericalDesign/NSdata/data_file03.pt',
    '/home/yifanchen/research/GenerativeDynamics-NumericalDesign/NSdata/data_file04.pt',
    '/home/yifanchen/research/GenerativeDynamics-NumericalDesign/NSdata/data_file05.pt',
]

# ─── Data ───────────────────────────────────────────────────────────────

def load_ns(hi_size, batch_size, train_test_split=0.9):
    avg_pixel_norm = 3.0679163932800293  # standard normalization for NS, matches existing scripts
    chunks = []
    for loc in DATA_LOCS:
        data_raw, _ = torch.load(loc, weights_only=False)
        Ntj, Nts, Nx, Ny = data_raw.shape
        print(f"[Data] {os.path.basename(loc)}: {Ntj} traj x {Nts} snaps x {Nx}x{Ny}")
        data_raw = data_raw / avg_pixel_norm
        data = data_raw.reshape(-1, Nx, Ny)
        if hi_size != Nx:
            data = nn.functional.interpolate(data.unsqueeze(1), size=(hi_size, hi_size), mode='bilinear').squeeze(1)
        chunks.append(data)
    data = torch.cat(chunks, dim=0)[:, None, :, :]
    print(f"[Data] total: {data.shape}, std={data.std():.4f}")
    n_train = int(data.shape[0] * train_test_split)
    print(f"[Data] train={n_train}, test={data.shape[0]-n_train}")
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(data[:n_train]),
        batch_size=batch_size, shuffle=True, drop_last=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(data[n_train:]),
        batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, data[:n_train], data[n_train:], avg_pixel_norm

# ─── UNet wrapper ───────────────────────────────────────────────────────

def make_model(hi_size):
    arch = Unet(
        num_classes=1,
        in_channels=1,
        out_channels=1,
        dim=32,
        dim_mults=(1, 2, 2, 2),
        resnet_block_groups=8,
        learned_sinusoidal_cond=True,
        random_fourier_features=False,
        learned_sinusoidal_dim=32,
        attn_dim_head=32,
        attn_heads=4,
        use_classes=False,
    ).to(DEVICE)
    n = sum(int(np.prod(p.shape)) for p in arch.parameters())
    print(f"[Net] params = {n:,}")
    return arch


def model_forward(arch, zt, t):
    # signature matches existing Unet wrapper expectation: (x, t, y)
    y = None
    return arch(zt, t, y)

# ─── Spectrum-matched noise sampler ─────────────────────────────────────

def make_noise_fn(hi_size, kind='matched'):
    """Per-mode std defining the prior covariance.

    kind='matched' : std = enstrohpy_spectrum_amplitude / 5 (same as the existing
                     matchingvar script — Fourier spectrum matches the data).
    kind='mulk'    : std = (enstrohpy_spectrum_amplitude / 5) * |k|
                     (rougher than data: high-k modes amplified by |k|).
    """
    amp_path = os.path.join(HOME, 'enstrohpy_spectrum_amplitude.pt')
    spectrum_amp = torch.load(amp_path, weights_only=False)[None].to(DEVICE) / 5.0  # (1,1,H,W)
    if kind == 'mulk':
        kfreq = torch.fft.fftfreq(hi_size, device=DEVICE) * hi_size
        kx, ky = torch.meshgrid(kfreq, kfreq, indexing='ij')
        k_mag = torch.sqrt(kx ** 2 + ky ** 2)[None, None, :, :]
        spectrum_amp = spectrum_amp * k_mag
        print('[noise] mul-k spectrum noise (rougher than data)')
    elif kind == 'matched':
        print('[noise] matched-spectrum noise')
    else:
        raise ValueError(kind)

    def sample(B):
        re = torch.randn(B, 1, hi_size, hi_size, device=DEVICE)
        im = torch.randn(B, 1, hi_size, hi_size, device=DEVICE)
        fourier = (re + 1j * im) * spectrum_amp
        return torch.fft.ifftn(fourier, dim=(2, 3), norm='forward').real

    return sample, spectrum_amp

# ─── Train ──────────────────────────────────────────────────────────────

def train(args):
    train_loader, test_loader, train_data, test_data, apn = load_ns(args.hi, args.batch, args.split)
    model = make_model(args.hi)
    noise_fn, spec_amp = make_noise_fn(args.hi, kind=args.noise_kind)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    t_dist = torch.distributions.Uniform(low=args.t_min, high=args.t_max)

    # optional resume from checkpoint
    start_step = 0
    if getattr(args, 'resume', None):
        state = torch.load(args.resume, weights_only=False)
        model.load_state_dict(state)
        # Try to parse step number from filename like ns_spectrum_<kind>_step<N>_hi<H>.pt
        import re
        m = re.search(r'step(\d+)', os.path.basename(args.resume))
        if m:
            start_step = int(m.group(1))
        print(f"[Train] resuming from {args.resume} at step {start_step}")

    print(f"[Train] starting; max_steps={args.max_steps}, start_step={start_step}")
    t0 = _t.time()
    step = start_step
    epoch = 0
    log_steps = []
    log_loss = []
    while step < args.max_steps:
        epoch += 1
        for batch in train_loader:
            if step >= args.max_steps:
                break
            x1 = batch[0].to(DEVICE)
            B = x1.shape[0]
            z0 = noise_fn(B)
            t = t_dist.sample((B,)).to(DEVICE).type_as(x1)
            tw = t[:, None, None, None]
            zt = (1 - tw) * z0 + tw * x1
            target = x1 - z0  # alpha_dot=-1, beta_dot=1; linear interpolant
            pred = model_forward(model, zt, t)
            loss = (pred - target).pow(2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1e4)
            opt.step()
            step += 1
            if step % 100 == 0:
                el = _t.time() - t0
                print(f"[step {step}] loss={loss.item():.4e}  elapsed={el:.1f}s")
                log_steps.append(step)
                log_loss.append(loss.item())
            if step % args.lr_decay_every == 0:
                # cosine
                scale = step / args.max_steps
                lr = args.lr * 0.5 * (1.0 + math.cos(math.pi * scale))
                for pg in opt.param_groups:
                    pg['lr'] = lr

    el = _t.time() - t0
    print(f"[Train] done in {el/60:.1f} min")

    # save model
    ck_path = os.path.join(HOME, f'ns_spectrum_{args.noise_kind}_step{args.max_steps}_hi{args.hi}.pt')
    torch.save(model.state_dict(), ck_path)
    print(f"[Train] saved ckpt to {ck_path}")
    np.save(os.path.join(HOME, f'ns_spectrum_{args.noise_kind}_loss_step{args.max_steps}.npy'),
            np.array([log_steps, log_loss]))
    return model, noise_fn, spec_amp, test_data, apn


# ─── Schedule transfer (Section 5 of the paper) ─────────────────────────

def transfer_drift(b_dagger_eval, alpha_t, beta_t, alpha_dot_t, beta_dot_t,
                   t_dagger):
    """Given a learned drift b_dagger trained under the linear schedule
    (alpha=1-t, beta=t), return the drift under the new schedule
    (alpha_t, beta_t).

    Following Proposition (transfer formula):
        b_t(x) = (alpha_dot_t/alpha_t) x +
                 (beta_dot_t - alpha_dot_t beta_t/alpha_t) *
                 ((1-t_dag) b_dagger(t_dag, t_dag/beta_t * x) + t_dag/beta_t * x)

    Here b_dagger_eval is a callable (t_dag, x) -> drift.
    """
    pass  # done inline per call below


def schedule_linear(t):
    return 1.0 - t, -1.0, t, 1.0


def schedule_designed_factory(lambda_star):
    r = float(lambda_star)
    log_r = math.log(r) if r > 0 else -1.0
    eps = 1e-12

    def sched(t):
        a2 = max((r - r ** t) / (r - 1.0), eps)
        b2 = max((r ** t - 1.0) / (r - 1.0), eps)
        a = math.sqrt(a2)
        b = math.sqrt(b2)
        # alpha_dot = -0.5 / a * r^t * log(r) / (r-1)
        da = -0.5 * (r ** t) * log_r / ((r - 1.0) * a)
        db = 0.5 * (r ** t) * log_r / ((r - 1.0) * b)
        return a, da, b, db
    return sched


def integrate_rk4_with_schedule(model, z0, sched, steps, t_min=1e-3, t_max=1.0 - 1e-3):
    """Integrate the ODE dX/dt = b_t(X) using the transfer formula:
    b_t(X) = (alpha_dot/alpha) X + (beta_dot - alpha_dot beta/alpha) *
             ((1 - t_dag) b_dagger(t_dag, t_dag/beta X) + t_dag/beta X)
    where b_dagger is the drift learned under linear schedule.
    """
    tgrid = torch.linspace(t_min, t_max, steps + 1, device=DEVICE)
    z = z0.clone()
    model.eval()
    for i in range(steps):
        ti = tgrid[i].item()
        dt = (tgrid[i + 1] - tgrid[i]).item()

        def b_func(z_in, t_in):
            a, ad, b, bd = sched(t_in)
            t_dag = 1.0 / (1.0 + a / max(b, 1e-12))
            scale = t_dag / max(b, 1e-12)
            arg = scale * z_in
            t_arr = torch.full((arg.shape[0],), t_dag, device=DEVICE)
            with torch.no_grad():
                bdag = model_forward(model, arg, t_arr)
            return (ad / max(a, 1e-12)) * z_in + (bd - ad * b / max(a, 1e-12)) * ((1 - t_dag) * bdag + scale * z_in)

        k1 = b_func(z, ti)
        k2 = b_func(z + 0.5 * dt * k1, ti + 0.5 * dt)
        k3 = b_func(z + 0.5 * dt * k2, ti + 0.5 * dt)
        k4 = b_func(z + dt * k3, ti + dt)
        z = z + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return z


def integrate_rk4_linear(model, z0, steps, t_min=1e-3, t_max=1.0 - 1e-3):
    """Direct integration with the linear schedule (no transfer)."""
    tgrid = torch.linspace(t_min, t_max, steps + 1, device=DEVICE)
    z = z0.clone()
    model.eval()
    for i in range(steps):
        ti = tgrid[i].item()
        dt = (tgrid[i + 1] - tgrid[i]).item()
        def b(z_in, t_in):
            t_arr = torch.full((z_in.shape[0],), t_in, device=DEVICE)
            with torch.no_grad():
                return model_forward(model, z_in, t_arr)
        k1 = b(z, ti)
        k2 = b(z + 0.5 * dt * k1, ti + 0.5 * dt)
        k3 = b(z + 0.5 * dt * k2, ti + 0.5 * dt)
        k4 = b(z + dt * k3, ti + dt)
        z = z + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return z

# ─── Spectra ────────────────────────────────────────────────────────────

def radial_spectrum(field):
    if isinstance(field, torch.Tensor):
        field = field.cpu()
    if field.dim() == 4:
        field = field.squeeze(1)
    fhat = torch.fft.fftn(field, dim=(1, 2), norm='forward')
    amp2 = (fhat.abs() ** 2).mean(dim=0).numpy()
    npix = amp2.shape[-1]
    kfreq = np.fft.fftfreq(npix) * npix
    kxg, kyg = np.meshgrid(kfreq, kfreq, indexing='ij')
    knrm = np.sqrt(kxg ** 2 + kyg ** 2).flatten()
    amp_flat = amp2.flatten()
    kbins = np.arange(0.5, npix // 2 + 1, 1.0)
    kvals = 0.5 * (kbins[1:] + kbins[:-1])
    Abins, _, _ = stats.binned_statistic(knrm, amp_flat, statistic='mean', bins=kbins)
    Abins *= np.pi * (kbins[1:] ** 2 - kbins[:-1] ** 2)
    return kvals, Abins


def banded(kvals, S_gen, S_truth, lo, hi):
    mask = (kvals >= lo) & (kvals < hi)
    rel = np.abs(S_gen[mask] - S_truth[mask]) / np.abs(S_truth[mask])
    return float(rel.mean())


# ─── Main: train + evaluate linear vs designed ─────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--max_steps', type=int, default=30000)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--hi', type=int, default=128)
    p.add_argument('--batch', type=int, default=100)
    p.add_argument('--split', type=float, default=0.9)
    p.add_argument('--t_min', type=float, default=1e-3)
    p.add_argument('--t_max', type=float, default=1 - 1e-3)
    p.add_argument('--lr_decay_every', type=int, default=200)
    p.add_argument('--num_eval', type=int, default=200, help='# samples to draw for spectrum estimation')
    p.add_argument('--steps_list', type=int, nargs='+', default=[10, 20, 50])
    p.add_argument('--lambda_star_override', type=float, default=None,
                   help='if set, use this instead of auto rule')
    p.add_argument('--noise_kind', type=str, default='matched', choices=['matched', 'mulk'])
    p.add_argument('--resume', type=str, default=None,
                   help='path to existing ckpt to resume training from')
    p.add_argument('--skip_train', action='store_true')
    p.add_argument('--ckpt', type=str, default=None,
                   help='path to existing ckpt to evaluate without training')
    args = p.parse_args()

    if args.skip_train:
        # try to load existing
        train_loader, test_loader, train_data, test_data, apn = load_ns(args.hi, args.batch, args.split)
        model = make_model(args.hi)
        noise_fn, spec_amp = make_noise_fn(args.hi, kind=args.noise_kind)
        if args.ckpt:
            state = torch.load(args.ckpt, weights_only=False)
            model.load_state_dict(state)
            print(f"[Eval] loaded ckpt {args.ckpt}")
        else:
            raise SystemExit('--skip_train requires --ckpt')
    else:
        model, noise_fn, spec_amp, test_data, apn = train(args)

    # Compute lambda* via the auto rule using the empirical spectra
    print("\n[Eval] computing data spectrum on test set ...")
    truth_spec_field = test_data[:args.num_eval].squeeze(1).cpu()
    kvals, S_truth = radial_spectrum(truth_spec_field)
    # noise spectrum (sample once, large batch)
    print("[Eval] computing noise spectrum ...")
    with torch.no_grad():
        noise_batch = noise_fn(args.num_eval).squeeze(1).cpu()
    _, S_noise = radial_spectrum(noise_batch)
    lambda_star_auto = float(S_truth[-1] / S_noise[-1])
    print(f"[Eval] auto lambda* = {lambda_star_auto:.4e}")
    if args.lambda_star_override:
        lambda_star = args.lambda_star_override
        print(f"[Eval] using override lambda* = {lambda_star}")
    else:
        lambda_star = lambda_star_auto

    # Generate samples with linear vs designed
    rows = []
    for steps in args.steps_list:
        # Linear (direct, no transfer)
        with torch.no_grad():
            z0 = noise_fn(args.num_eval)
        gen_lin = integrate_rk4_linear(model, z0, steps=steps, t_min=args.t_min, t_max=args.t_max)
        _, S_gen_lin = radial_spectrum(gen_lin.cpu())

        # Designed (with transfer formula)
        sched = schedule_designed_factory(lambda_star)
        gen_des = integrate_rk4_with_schedule(model, z0, sched, steps=steps,
                                              t_min=args.t_min, t_max=args.t_max)
        _, S_gen_des = radial_spectrum(gen_des.cpu())

        mid_lin = banded(kvals, S_gen_lin, S_truth, 8, 24)
        high_lin = banded(kvals, S_gen_lin, S_truth, 24, args.hi // 2 + 1)
        mid_des = banded(kvals, S_gen_des, S_truth, 8, 24)
        high_des = banded(kvals, S_gen_des, S_truth, 24, args.hi // 2 + 1)
        rows.append((steps, mid_lin, high_lin, mid_des, high_des))
        print(f"[Eval] steps={steps:3d} linear  mid={mid_lin:.3e} high={high_lin:.3e}")
        print(f"[Eval] steps={steps:3d} designed mid={mid_des:.3e} high={high_des:.3e}")

        # plot spectrum
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.plot(kvals, S_truth, 'k-', lw=2, label='Truth (test)')
        ax.plot(kvals, S_gen_lin, 'r--', lw=1.2, label=f'Spectrum noise + linear ({steps} steps)')
        ax.plot(kvals, S_gen_des, 'b--', lw=1.2, label=f'Spectrum noise + designed ({steps} steps)')
        ax.plot(kvals, S_noise, 'g:', lw=1.0, label='Noise prior spectrum')
        ax.set_xscale('log', base=2)
        ax.set_yscale('log')
        ax.set_xlabel('Wavenumber k')
        ax.set_ylabel('Enstrophy spectrum')
        ax.set_title(f'NS {args.hi}x{args.hi}: spectrum noise prior, $\\lambda^\\star$={lambda_star:.3e}')
        ax.legend(fontsize=8)
        ax.grid(True, which='both', alpha=0.3)
        plt.tight_layout()
        out = os.path.join(HOME, f'ns_spectrum_{args.noise_kind}_des_vs_lin_steps{steps}_step{args.max_steps}.pdf')
        plt.savefig(out, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"[Eval] saved fig {out}")

    # save table
    out_path = os.path.join(HOME, f'ns_spectrum_{args.noise_kind}_results_step{args.max_steps}.txt')
    with open(out_path, 'w') as f:
        f.write(f"# NS spectrum noise prior, hi={args.hi}, ckpt at step {args.max_steps}\n")
        f.write(f"# lambda* (auto) = {lambda_star:.6e}\n")
        f.write("# steps  mid_lin  high_lin  mid_des  high_des\n")
        for r in rows:
            f.write("\t".join(f"{x:.6e}" if isinstance(x, float) else str(x) for x in r) + "\n")
    print(f"[Eval] wrote {out_path}")


if __name__ == '__main__':
    main()
