"""Re-render AR-animation frames from cached arrays.npz with custom colour scales.
Cheap (~ms per frame). No resampling needed.

Usage:
  python ar_rerender.py --dir figs/ar_anim_lag1 --err_scale 4.0 --sd_scale 3.0
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ar_animation_frames import save_frame


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dir', type=str, required=True)
    p.add_argument('--err_scale', type=float, default=4.0)
    p.add_argument('--sd_scale',  type=float, default=3.0)
    p.add_argument('--vlim',      type=float, default=0.0, help='0 = use cached vlim')
    args = p.parse_args()

    z = np.load(os.path.join(args.dir, 'arrays.npz'))
    truth = z['truth']; mean = z['mean']; err = z['err']; sd = z['sd']
    vlim = float(z['vlim']) if args.vlim <= 0 else args.vlim
    print(f'[re-render] {truth.shape[0]} frames, vlim=±{vlim:.2f}, '
          f'err_scale={args.err_scale}, sd_scale={args.sd_scale}')
    for k in range(truth.shape[0]):
        path = os.path.join(args.dir, f'frame_{k:03d}.png')
        save_frame(path, truth[k], mean[k], err[k], sd[k],
                   vlim, args.err_scale, args.sd_scale)
    print(f'[done] {truth.shape[0]} frames rewritten')


if __name__ == '__main__':
    main()
