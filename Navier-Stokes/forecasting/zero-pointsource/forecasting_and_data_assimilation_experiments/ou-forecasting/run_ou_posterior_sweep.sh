#!/usr/bin/env bash
# OU posterior-sampling sweep: guided Doob/natural × eta 0.1/0.5/1.0 at sigma_y=0.1,
# plus uncond baseline. 5 seeds × 20 ICs × 100 particles × 200 EM steps.
# 1D so --cpu is fast enough and avoids GPU contention with the NS sweep.
set -euo pipefail
cd "$(dirname "$0")"
source /home/yifanchen/miniconda3/etc/profile.d/conda.sh
conda activate gpu
mkdir -p figs_posterior_ou logs_ou_posterior

COMMON="--target ou_forecast --seeds 0 1 2 3 4 --n_particles 100 --n_em 200 --n_ic 20 --sigma_y 0.1"
OUTDIR=./figs_posterior_ou

python posterior_compare.py $COMMON --proposal uncond --guidance_type doob --guidance_eta 1.0 \
    --out_dir $OUTDIR > logs_ou_posterior/uncond.log 2>&1

for GT in doob natural; do
  for ETA in 0.1 0.5 1.0; do
    python posterior_compare.py $COMMON --proposal guided --guidance_type $GT --guidance_eta $ETA \
        --out_dir $OUTDIR > logs_ou_posterior/guided_${GT}_eta${ETA}.log 2>&1
  done
done

echo "[ou-sweep] done"
