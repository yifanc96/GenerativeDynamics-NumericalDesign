"""Animate AR-filter forecasts vs truth side-by-side (MP4 + GIF).

Reads ar_filter/results.pt and renders:
  - ar_animation.mp4 : truth | Föllmer ens-mean | baseline ens-mean | const ens-mean | ODE ens-mean
    One frame per step. Dot at observation steps.
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as mpl_anim
import numpy as np
import seaborn as sns
import torch


SCHEDULES_TO_SHOW = ['follmer', 'baseline', 'const', 'zero']
PRETTY = {
    'follmer':  r'Föllmer $\sqrt{1-t^2}$',
    'baseline': r'baseline $1-t$',
    'const':    r'const $g{=}1$',
    'zero':     r'ODE ($g{=}0$)',
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--src', type=str, default='./figs/ar_filter/results.pt')
    p.add_argument('--out', type=str, default='./figs/ar_filter/ar_animation.mp4')
    p.add_argument('--fps', type=int, default=6)
    args = p.parse_args()

    d = torch.load(args.src, weights_only=False)
    results = d['results']
    ckargs = d['ck_args']
    runargs = d['args']

    # Collect trajectories for IC 0 across the chosen schedules.
    traj_per_sched = {}
    for sc in SCHEDULES_TO_SHOW:
        per_ic = results[sc]
        if '_traj' not in per_ic[0]:
            raise RuntimeError(f'no _traj for schedule {sc}')
        traj_per_sched[sc] = per_ic[0]['_traj']
    truth = traj_per_sched[SCHEDULES_TO_SHOW[0]]['truth']
    means = {sc: traj_per_sched[sc]['x_mean'] for sc in SCHEDULES_TO_SHOW}
    stds = {sc: traj_per_sched[sc]['x_std'] for sc in SCHEDULES_TO_SHOW}
    n_frames = len(means[SCHEDULES_TO_SHOW[0]])
    H = means[SCHEDULES_TO_SHOW[0]][0].shape[-1]
    print(f'[animate] {n_frames} frames, H={H}, saving {args.out}')

    # Determine colour range from truth + means (cast through np.asarray for safety)
    all_arrs = [np.asarray(m) for m_list in means.values() for m in m_list] + [np.asarray(t) for t in truth if t is not None]
    vlim = float(np.percentile(np.abs(np.concatenate([a.flatten() for a in all_arrs])), 99.5))

    cols = 1 + len(SCHEDULES_TO_SHOW) + 1      # truth | schedules | std-of-Follmer
    fig, axes = plt.subplots(1, cols, figsize=(2.8 * cols, 3.0))
    cmap = sns.cm.icefire
    # initial imshow handles
    ims = []
    # truth
    im = axes[0].imshow(np.zeros((H, H)), cmap=cmap, vmin=-vlim, vmax=vlim)
    axes[0].set_title('truth $\\omega_{t+\\tau}$', fontsize=10)
    axes[0].set_xticks([]); axes[0].set_yticks([])
    ims.append(im)
    # schedules
    for i, sc in enumerate(SCHEDULES_TO_SHOW, 1):
        im = axes[i].imshow(np.zeros((H, H)), cmap=cmap, vmin=-vlim, vmax=vlim)
        axes[i].set_title(f'{PRETTY[sc]}\n(ens mean)', fontsize=10)
        axes[i].set_xticks([]); axes[i].set_yticks([])
        ims.append(im)
    # std of follmer
    ax_std = axes[-1]
    im_std = ax_std.imshow(np.zeros((H, H)), cmap='plasma', vmin=0, vmax=vlim / 2)
    ax_std.set_title('Föllmer ens-std', fontsize=10)
    ax_std.set_xticks([]); ax_std.set_yticks([])

    # Top title slot for step number
    step_text = fig.suptitle('', fontsize=12)

    last_truth = np.zeros((H, H))

    def update(frame):
        nonlocal last_truth
        t_frame = truth[frame]
        if t_frame is not None:
            last_truth = np.asarray(t_frame)
        ims[0].set_data(last_truth)
        for i, sc in enumerate(SCHEDULES_TO_SHOW, 1):
            ims[i].set_data(np.asarray(means[sc][frame]))
        im_std.set_data(np.asarray(stds['follmer'][frame]))
        # Frame info
        R = runargs['rollout_len']
        cycle = frame // R
        is_obs = (frame > 0) and (frame % R == 0)
        tag = f'step {frame}/{n_frames - 1}' + ('  (observation reweight)' if is_obs else '')
        step_text.set_text(tag)
        return ims + [im_std]

    anim = mpl_anim.FuncAnimation(fig, update, frames=n_frames, interval=1000 // args.fps, blit=False)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    try:
        writer = mpl_anim.FFMpegWriter(fps=args.fps, bitrate=4000, codec='libx264')
        anim.save(args.out, writer=writer)
        print(f'[saved] {args.out}')
    except Exception as e:
        print(f'[ffmpeg fallback GIF] {e}')
        gif = args.out.replace('.mp4', '.gif')
        anim.save(gif, writer='pillow', fps=args.fps)
        print(f'[saved] {gif}')
    plt.close(fig)


if __name__ == '__main__':
    main()
