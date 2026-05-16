"""Render AR rollout into an MP4 (saved, not displayed).

Layout per frame, 5 rows x 1 column:
  Row 1: truth                      (shared icefire scale)
  Row 2: prior ensemble mean        (icefire)
  Row 3: localised-EnKF ens mean    (icefire)
  --- gap ---
  Row 4: |EnKF mean - truth|        (magma)
  Row 5: EnKF ensemble std          (magma, same scale as row 4 to test calibration)

Reads cached arrays.npz from prior + enkf_loc directories. Truth is the same
in both, taken from enkf_loc (consistency check inside).
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.animation as mpl_anim
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prior_dir', type=str, default='figs/ar_anim_prior')
    p.add_argument('--da_dir',    type=str, default='figs/ar_anim_enkf_loc')
    p.add_argument('--out',       type=str, default='figs/ar_animation.mp4')
    p.add_argument('--fps',       type=int, default=8)
    p.add_argument('--err_scale', type=float, default=0.0,
                   help='cap for the |err| / std panels; 0 = pick max(err, std)')
    args = p.parse_args()

    zp = np.load(os.path.join(args.prior_dir, 'arrays.npz'))
    zd = np.load(os.path.join(args.da_dir,    'arrays.npz'))
    truth = zd['truth']                  # (T, H, H)
    prior_mean = zp['mean']
    da_mean    = zd['mean']
    da_err     = zd['err']
    da_sd      = zd['sd']

    # Sanity check truth match
    assert np.allclose(zd['truth'], zp['truth']), 'truth arrays differ between prior_dir and da_dir'
    T, H, _ = truth.shape

    vlim = float(np.percentile(np.abs(truth), 99.5))
    err_cap = args.err_scale if args.err_scale > 0 else \
              float(np.percentile(np.concatenate([da_err.flatten(), da_sd.flatten()]), 99.5))

    print(f'[render] {T} frames, H={H}, vlim=±{vlim:.2f}, err/std scale=0..{err_cap:.2f}')
    print(f'[render] saving {args.out} at {args.fps} fps ({T / args.fps:.1f}s)')

    sns.set_theme(context='paper', style='white', font_scale=0.9)
    cmap_v = sns.cm.icefire
    cmap_e = 'magma'

    # Use GridSpec for 5 rows with a gap between rows 3 and 4
    fig = plt.figure(figsize=(4.6, 14.4))
    gs = fig.add_gridspec(
        nrows=6, ncols=2,
        height_ratios=[1.0, 1.0, 1.0, 0.12, 1.0, 1.0],   # row 4 (idx 3) is the gap
        width_ratios=[20, 1],
        hspace=0.07, wspace=0.05,
        left=0.03, right=0.92, top=0.98, bottom=0.02,
    )
    panels = []     # list of (ax, im, kind)  kind in {'v','e'}
    rows = [
        (0, truth,      cmap_v, -vlim, vlim, 'truth'),
        (1, prior_mean, cmap_v, -vlim, vlim, 'prior'),
        (2, da_mean,    cmap_v, -vlim, vlim, 'da-mean'),
        (4, da_err,     cmap_e, 0.0,   err_cap, 'da-err'),
        (5, da_sd,      cmap_e, 0.0,   err_cap, 'da-sd'),
    ]
    cax_groups = []
    for ridx, arr, cmap, vmin, vmax, _ in rows:
        ax = fig.add_subplot(gs[ridx, 0])
        im = ax.imshow(arr[0], cmap=cmap, vmin=vmin, vmax=vmax, origin='lower',
                       interpolation='bilinear')
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        cax = fig.add_subplot(gs[ridx, 1])
        cb = fig.colorbar(im, cax=cax)
        cb.ax.tick_params(labelsize=7)
        panels.append((ax, im, arr))
        cax_groups.append(cax)

    def update(k):
        for (ax, im, arr) in panels:
            im.set_data(arr[k])
        return [im for (_, im, _) in panels]

    anim = mpl_anim.FuncAnimation(fig, update, frames=T,
                                  interval=int(1000 / args.fps), blit=False)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    # Resolve ffmpeg via imageio-ffmpeg (no system ffmpeg required)
    try:
        import imageio_ffmpeg
        plt.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    try:
        writer = mpl_anim.FFMpegWriter(fps=args.fps, bitrate=4500, codec='libx264',
                                       extra_args=['-pix_fmt', 'yuv420p'])
        anim.save(args.out, writer=writer, dpi=150)
        print(f'[saved mp4] {args.out}')
    except Exception as e:
        print(f'[ffmpeg failed: {e}; falling back to gif]')
        gif = args.out.replace('.mp4', '.gif')
        anim.save(gif, writer='pillow', fps=args.fps, dpi=110)
        print(f'[saved gif] {gif}')
    plt.close(fig)


if __name__ == '__main__':
    main()
