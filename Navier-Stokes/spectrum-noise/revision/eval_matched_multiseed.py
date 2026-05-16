"""
Re-evaluate the trained matched-spectrum NS UNet at the 50000-step checkpoint
with multiple noise seeds to get error bars for Table 4.5.
"""
import math, os, sys
import numpy as np, torch, torch.nn as nn
import scipy.stats as stats

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
HOME = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HOME)
from unet import Unet


def make_unet():
    return Unet(num_classes=1, in_channels=1, out_channels=1, dim=32,
                dim_mults=(1, 2, 2, 2), resnet_block_groups=8,
                learned_sinusoidal_cond=True, random_fourier_features=False,
                learned_sinusoidal_dim=32, attn_dim_head=32, attn_heads=4, use_classes=False)


def model_call(net, zt, t):
    """Wrapper matching the training script's forward signature."""
    return net(zt, t, classes=None)


def load_data(hi=128, split=0.9, batch=200):
    avg = 3.0679163932800293
    chunks = []
    for loc in [
        '/home/yifanchen/research/GenerativeDynamics-NumericalDesign/NSdata/data_file.pt',
        '/home/yifanchen/research/GenerativeDynamics-NumericalDesign/NSdata/data_file02.pt',
        '/home/yifanchen/research/GenerativeDynamics-NumericalDesign/NSdata/data_file03.pt',
        '/home/yifanchen/research/GenerativeDynamics-NumericalDesign/NSdata/data_file04.pt',
        '/home/yifanchen/research/GenerativeDynamics-NumericalDesign/NSdata/data_file05.pt',
    ]:
        d, _ = torch.load(loc, weights_only=False)
        d = d / avg
        d = d.reshape(-1, d.shape[2], d.shape[3])
        if d.shape[1] != hi:
            d = nn.functional.interpolate(d.unsqueeze(1), size=(hi, hi), mode='bilinear').squeeze(1)
        chunks.append(d)
    data = torch.cat(chunks)[:, None, :, :]
    n_train = int(data.shape[0] * split)
    return data[n_train:]  # test set


def make_noise_fn(hi):
    spec = torch.load(os.path.join(HOME, 'enstrohpy_spectrum_amplitude.pt'), weights_only=False)[None].to(DEVICE) / 5.0
    def sample(B, gen):
        re = torch.randn(B, 1, hi, hi, generator=gen, device=DEVICE)
        im = torch.randn(B, 1, hi, hi, generator=gen, device=DEVICE)
        return torch.fft.ifftn((re + 1j * im) * spec, dim=(2, 3), norm='forward').real
    return sample


def integrate_rk4_linear(model, z0, steps, t_min=1e-3, t_max=1 - 1e-3):
    tg = torch.linspace(t_min, t_max, steps + 1, device=DEVICE)
    z = z0.clone()
    model.eval()
    for i in range(steps):
        ti, dt = tg[i].item(), (tg[i + 1] - tg[i]).item()
        def b(z_, t_):
            ta = torch.full((z_.shape[0],), t_, device=DEVICE)
            with torch.no_grad():
                return model_call(model, z_, ta)
        k1 = b(z, ti); k2 = b(z + .5 * dt * k1, ti + .5 * dt)
        k3 = b(z + .5 * dt * k2, ti + .5 * dt); k4 = b(z + dt * k3, ti + dt)
        z = z + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return z


def integrate_rk4_designed(model, z0, lam, steps, t_min=1e-3, t_max=1 - 1e-3):
    log_r = math.log(lam)
    eps = 1e-12
    def sched(t):
        a2 = max((lam - lam ** t) / (lam - 1.0), eps)
        b2 = max((lam ** t - 1.0) / (lam - 1.0), eps)
        a, b = math.sqrt(a2), math.sqrt(b2)
        da = -0.5 * (lam ** t) * log_r / ((lam - 1.0) * a)
        db = 0.5 * (lam ** t) * log_r / ((lam - 1.0) * b)
        return a, da, b, db
    tg = torch.linspace(t_min, t_max, steps + 1, device=DEVICE)
    z = z0.clone()
    model.eval()
    for i in range(steps):
        ti, dt = tg[i].item(), (tg[i + 1] - tg[i]).item()
        def b_func(z_in, tt):
            a, ad, bb, bd = sched(tt)
            t_dag = 1.0 / (1.0 + a / max(bb, 1e-12))
            scale = t_dag / max(bb, 1e-12)
            arg = scale * z_in
            ta = torch.full((arg.shape[0],), t_dag, device=DEVICE)
            with torch.no_grad():
                bdag = model_call(model, arg, ta)
            return (ad / max(a, 1e-12)) * z_in + (bd - ad * bb / max(a, 1e-12)) * ((1 - t_dag) * bdag + scale * z_in)
        k1 = b_func(z, ti); k2 = b_func(z + .5 * dt * k1, ti + .5 * dt)
        k3 = b_func(z + .5 * dt * k2, ti + .5 * dt); k4 = b_func(z + dt * k3, ti + dt)
        z = z + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return z


def radial_spec(field):
    if field.dim() == 4:
        field = field.squeeze(1)
    fhat = torch.fft.fftn(field.cpu(), dim=(1, 2), norm='forward')
    a2 = (fhat.abs() ** 2).mean(0).numpy()
    npix = a2.shape[-1]
    kf = np.fft.fftfreq(npix) * npix
    kx, ky = np.meshgrid(kf, kf, indexing='ij')
    kn = np.sqrt(kx ** 2 + ky ** 2).flatten()
    af = a2.flatten()
    kbins = np.arange(0.5, npix // 2 + 1, 1.0)
    kvals = 0.5 * (kbins[1:] + kbins[:-1])
    A, _, _ = stats.binned_statistic(kn, af, statistic='mean', bins=kbins)
    return kvals, A * np.pi * (kbins[1:] ** 2 - kbins[:-1] ** 2)


def banded(kvals, sg, st, lo, hi):
    m = (kvals >= lo) & (kvals < hi)
    return float((np.abs(sg[m] - st[m]) / np.abs(st[m])).mean())


def main():
    hi = 128
    test = load_data(hi=hi).to(DEVICE)
    print(f'test set: {test.shape}')

    model = make_unet().to(DEVICE)
    state = torch.load(os.path.join(HOME, 'ns_spectrum_matched_step50000_hi128.pt'),
                       map_location=DEVICE, weights_only=False)
    model.load_state_dict(state)

    noise_fn = make_noise_fn(hi)
    # truth spectrum
    kvals, S_truth = radial_spec(test[:200].cpu())

    # auto lambda
    g0 = torch.Generator(device=DEVICE).manual_seed(0)
    noise_batch = noise_fn(2000, g0).cpu()
    _, S_noise = radial_spec(noise_batch)
    lam_auto = float(S_truth[-1] / S_noise[-1])
    print(f'auto lambda* = {lam_auto:.4e}')

    seeds = [42, 123, 999]
    steps_list = [10, 20, 50]

    rows = []
    for steps in steps_list:
        for sched_name in ['linear', 'designed']:
            highs = []
            mids = []
            for seed in seeds:
                gen = torch.Generator(device=DEVICE).manual_seed(seed)
                z0 = noise_fn(200, gen)
                if sched_name == 'linear':
                    out = integrate_rk4_linear(model, z0, steps)
                else:
                    out = integrate_rk4_designed(model, z0, lam_auto, steps)
                _, S_gen = radial_spec(out)
                mids.append(banded(kvals, S_gen, S_truth, 8, 24))
                highs.append(banded(kvals, S_gen, S_truth, 24, hi // 2 + 1))
            mids, highs = np.array(mids), np.array(highs)
            print(f'  steps={steps:>3} {sched_name:>8}: mid={mids.mean():.3e}±{mids.std():.3e}'
                  f' high={highs.mean():.3e}±{highs.std():.3e}')
            rows.append((steps, sched_name, mids.mean(), mids.std(), highs.mean(), highs.std()))

    out_path = os.path.join(HOME, 'ns_spectrum_matched_results_50k_3seeds.txt')
    with open(out_path, 'w') as f:
        f.write('# steps  schedule  mid_mean  mid_std  high_mean  high_std\n')
        for r in rows:
            f.write('\t'.join(str(x) for x in r) + '\n')
    print(f'\nwrote: {out_path}')


if __name__ == '__main__':
    main()
