"""
Load the pure flow matching NS baseline and evaluate at K=1,2,4,8,16 steps
to fairly compare with mean flow results (which used those same step counts).
"""
import os, sys, math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import scipy.stats as stats

# Import helpers from the data-dep-noise script
from train_ns_data_dep_noise import (
    load_ns_data,
    get_energy_spectrum,
    Velocity,
    Sampler,
    Config,
    make_data_dependent_noise,
)


def evaluate(model_path, config, noise_type, device, step_counts=(2, 4, 8, 16)):
    model = Velocity(config).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    sampler = Sampler(config)

    _, _, train_data, test_data, _ = load_ns_data(
        config.data_loc, config.hi_size, config.batch_size, config.train_test_split
    )

    num_eval = min(200, test_data.shape[0])
    truth = test_data[:num_eval]
    truth_sq = truth.squeeze(1)
    kvals, enst_truth, ener_truth = get_energy_spectrum(truth_sq)

    if noise_type == 'gauss':
        z0 = torch.randn(num_eval, 1, config.hi_size, config.hi_size, device=device)
    else:
        perm = torch.randperm(train_data.shape[0])
        noise_src = train_data[perm[:config.batch_size]].to(device)
        z0 = make_data_dependent_noise(noise_src, num_eval)

    print(f"\n[{noise_type}] Baseline pure flow matching: {model_path}")
    print(f"  {'method':<10} | {'low enst':>10} | {'mid enst':>10} | {'high enst':>10} | {'std_ratio':>10}")
    print("  " + "-" * 64)

    for nsteps in step_counts:
        with torch.no_grad():
            for method, fn in [('EM', sampler.EM), ('RK4', sampler.RK4)]:
                gen = fn(z0.clone(), model, nsteps)
                gen_sq = gen.squeeze(1).cpu()
                _, enst_gen, ener_gen = get_energy_spectrum(gen_sq)

                std_ratio = gen_sq.std().item() / (truth_sq.std().item() + 1e-12)
                bands = {
                    'low': kvals < 8,
                    'mid': (kvals >= 8) & (kvals < 24),
                    'high': kvals >= 24,
                }
                bm = {}
                for bn, mask in bands.items():
                    bm[bn] = np.mean(
                        np.abs(enst_truth[mask] - enst_gen[mask])
                        / (np.abs(enst_truth[mask]) + 1e-12)
                    )

                tag = f"{method}{nsteps}"
                print(
                    f"  {tag:<10} | {bm['low']:>10.4f} | {bm['mid']:>10.4f} | "
                    f"{bm['high']:>10.4f} | {std_ratio:>10.4f}"
                )


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=int, default=6)
    p.add_argument('--data_loc', type=str, default='../NSdata/data_file.pt')
    args = p.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda')

    config = Config()
    config.data_loc = args.data_loc
    config.use_wandb = False

    # Evaluate both baselines
    evaluate(
        'results/ns_data_dep_noise/model_final.pt',
        config, 'data_dep', device,
    )
    evaluate(
        'results/ns_gauss_base/model_final.pt',
        config, 'gauss', device,
    )
