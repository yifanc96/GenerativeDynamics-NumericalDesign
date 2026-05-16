#!/usr/bin/env bash
# EnKF baselines for the AR animation: vanilla + Gaspari-Cohn localised.
# Same forecast (Follmer prior) and same observations as figs/ar_anim_da/.
set -euo pipefail
cd "$(dirname "$0")"
source /home/yifanchen/miniconda3/etc/profile.d/conda.sh
conda activate gpu
export LD_LIBRARY_PATH=/home/yifanchen/miniconda3/envs/gpu/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}

COMMON="--ckpt runs/lag1_seed0.pt --n_steps 80 --n_em 100 --traj_idx 0 --t_start 20 --n_ens 32 \
        --assimilate --obs_factor 4 --sigma_y 0.03 --inflation 1.05 \
        --err_scale 1.0 --sd_scale 1.0"

# Vanilla stochastic EnKF (rank-deficient: 32 particles vs 1024 obs)
CUDA_VISIBLE_DEVICES=4 python ar_animation_frames.py $COMMON \
    --da_method enkf --out_dir figs/ar_anim_enkf > logs_ar_enkf.log 2>&1 &
PID_VANILLA=$!

# Localised EnKF (Gaspari-Cohn radius 8 grid pixels)
CUDA_VISIBLE_DEVICES=5 python ar_animation_frames.py $COMMON \
    --da_method enkf_loc --loc_radius 8.0 --out_dir figs/ar_anim_enkf_loc > logs_ar_enkf_loc.log 2>&1 &
PID_LOC=$!

wait $PID_VANILLA $PID_LOC
echo '[enkf] both done'
