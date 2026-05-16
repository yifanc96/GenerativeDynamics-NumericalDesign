"""Load trained hat_b for a given (lag, seed); sweep schedules with a bigger ensemble
and larger test set; save JSON with all metrics."""
import argparse
import json
import os

import numpy as np
import torch

from data import load_snapshots, split_train_val_test, make_loaders
from interpolant_ns import INTERPOLANTS
from network_ns import DriftNet
from schedules_ns import list_schedules, make_g
from drift_compose_ns import compose_drift
from sampler_ns import em_sample
from metrics_ns import (crps_ensemble, rmse_ensemble_mean, ensemble_spread,
                        spread_skill_ratio, spectrum_rmse, enstrophy_w2, vorticity_pdf_w2,
                        rank_histogram, anomaly_correlation_coefficient, radial_spectrum)


def load_checkpoint(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    args = ck['args']
    norm = ck['norm']
    net = DriftNet(unet_channels=args['unet_channels'],
                   unet_dim_mults=tuple(args['unet_dim_mults'])).to(device)
    net.load_state_dict(ck['state'])
    net.eval()
    return net, args, norm


@torch.no_grad()
def evaluate(ckpt_path, args_eval, device):
    net, ck_args, norm = load_checkpoint(ckpt_path, device)
    ip = INTERPOLANTS[ck_args.get('interpolant', 'linlin')]()
    omega_all, sim_args = load_snapshots(ck_args['data'])
    N_traj = omega_all.shape[0]
    train_idx, val_idx, test_idx = split_train_val_test(
        N_traj, ck_args['n_train'], ck_args['n_val'], ck_args['n_test'],
        seed=ck_args['data_seed'])
    # Skip train/val loaders (not needed for eval) — just build test loader.
    from data import PairDataset
    test_ds = PairDataset(omega_all[test_idx], ck_args['lag'],
                          coarsen_factor=ck_args['coarsen'])
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=args_eval.batch_size,
                                          shuffle=False, num_workers=0)
    # Use `args_eval.n_test_samples` pairs
    ns = args_eval.n_test_samples
    H = omega_all.shape[-1]

    # Climate mean (for ACC) from training trajectories (cheap)
    clim = omega_all.float().mean()

    out = {}

    def b_fn(x, t, x0): return net(x, t, x0)

    # Iterate test loader, build ensemble forecasts per schedule
    for name in list_schedules():
        g_fn = make_g(name, interpolant=ip)
        bg = compose_drift(b_fn, g_fn, ip)
        ens_all, truth_all = [], []
        done = 0
        for batch in test_dl:
            if done >= ns:
                break
            x1 = (batch['x1'].to(device) / norm)
            x0_up = (batch['x0_up'].to(device) / norm)
            B = x1.shape[0]
            ens = torch.zeros(args_eval.ensemble, B, 1, H, H, device=device)
            t_eps_use = getattr(args_eval, 't_eps_override', None) or ck_args.get('t_eps', 1e-3)
            for k in range(args_eval.ensemble):
                ens[k] = em_sample(bg, g_fn, (B, 1, H, H),
                                   n_steps=args_eval.n_em,
                                   t_min=t_eps_use,
                                   t_max=1.0 - t_eps_use,
                                   cond=(x0_up,), device=device)
            ens_all.append(ens.cpu())
            truth_all.append(x1.cpu())
            done += B
        ens_all = torch.cat(ens_all, dim=1)[:, :ns]      # (K, ns, 1, H, H) in normalised units
        truth_all = torch.cat(truth_all, dim=0)[:ns]     # (ns, 1, H, H)
        # Un-normalise for physical metrics
        ens_phys = ens_all * norm
        tru_phys = truth_all * norm
        clim_phys = clim

        rec = {}
        rec['crps'] = crps_ensemble(ens_phys, tru_phys)
        rec['rmse'] = rmse_ensemble_mean(ens_phys, tru_phys)
        rec['spread'] = ensemble_spread(ens_phys)
        rec['ssr'] = spread_skill_ratio(ens_phys, tru_phys)
        rec['acc'] = anomaly_correlation_coefficient(ens_phys, tru_phys, clim_phys)
        rec['spec_rmse'] = spectrum_rmse(ens_phys.squeeze(2), tru_phys.squeeze(1))
        rec['enstrophy_w2'] = enstrophy_w2(ens_phys.squeeze(2), tru_phys.squeeze(1))
        rec['pdf_w2'] = vorticity_pdf_w2(ens_phys.squeeze(2), tru_phys.squeeze(1))
        rec['rank_hist'] = rank_histogram(ens_phys, tru_phys).tolist()
        out[name] = rec
        print(f"  {name:10s}: CRPS={rec['crps']:.3e}  RMSE={rec['rmse']:.3e}  "
              f"SSR={rec['ssr']:.3e}  ens_W2={rec['enstrophy_w2']:.3e}  "
              f"spec={rec['spec_rmse']:.3e}")
    return out, ck_args


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', type=str, required=True)
    p.add_argument('--out', type=str, required=True)
    p.add_argument('--ensemble', type=int, default=50)
    p.add_argument('--n_em', type=int, default=100)
    p.add_argument('--n_test_samples', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--cpu', action='store_true')
    p.add_argument('--t_eps_override', type=float, default=None,
                   help='Override the sampler t_min/t_max buffer (default: checkpoint value)')
    args = p.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"[eval] {args.ckpt}")
    results, ck_args = evaluate(args.ckpt, args, device)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump({'results': results, 'ck_args': ck_args, 'eval_args': vars(args)}, f, indent=2)
    print(f"[saved] {args.out}")


if __name__ == '__main__':
    main()
