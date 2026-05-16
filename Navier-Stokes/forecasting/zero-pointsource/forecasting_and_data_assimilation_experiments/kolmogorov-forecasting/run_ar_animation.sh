#!/usr/bin/env bash
# Autoregressive forecast frames for animation. Two versions:
#   - ar_anim_da/   : with noisy observation assimilated each step -> bounded error
#   - ar_anim_prior/: pure forecast, no obs -> error grows over time
# lag=1 base, 80 steps, 8 ensemble members, Follmer schedule.
set -euo pipefail
cd "$(dirname "$0")"
source /home/yifanchen/miniconda3/etc/profile.d/conda.sh
conda activate gpu
export LD_LIBRARY_PATH=/home/yifanchen/miniconda3/envs/gpu/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}

COMMON="--ckpt runs/lag1_seed0.pt --n_steps 80 --n_em 50 --traj_idx 0 --t_start 20"

# 1) AR + DA with informative obs (2x coarse, sigma_y=0.1, 32 particles)
#    -> RMSE stays < 0.5 every step
python ar_animation_frames.py $COMMON --n_ens 32 \
    --out_dir figs/ar_anim_da \
    --assimilate --obs_factor 2 --sigma_y 0.1 --guidance_eta 1.0 \
    --err_scale 1.0 --sd_scale 1.0

# 2) Pure AR (no obs). Uses the same n_ens=8 baseline.
python ar_animation_frames.py $COMMON --n_ens 8 \
    --out_dir figs/ar_anim_prior \
    --err_scale 10.0 --sd_scale 5.0

echo '[ar-anim] both modes done'
