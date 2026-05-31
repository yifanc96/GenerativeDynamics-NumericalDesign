#!/usr/bin/env bash
# Posterior sweep on sqlin (beta=t^2) checkpoints. Uses t_eps=0.05 because
# the sqlin Follmer drift b^F = (3-t)/(2-t) b - 2x/(t(2-t)) is stiff at t~0.
set -euo pipefail
cd "$(dirname "$0")"
source /home/yifanchen/miniconda3/etc/profile.d/conda.sh
conda activate gpu
export LD_LIBRARY_PATH=/home/yifanchen/miniconda3/envs/gpu/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}

OUTDIR=figs/posterior_sweep_sqlin
LOGDIR=logs_posterior_sweep_sqlin
mkdir -p "$OUTDIR" "$LOGDIR"

COMMON_ARGS="--n_particles 32 --n_em 100 --n_ic 8 --t_eps 0.05 --obs_factor 8 --sigma_y 0.3"

for LAG in 10 40; do
  CKPT=runs/sqlin/lag${LAG}_seed0.pt

  # uncond doob eta=1.0 (baseline — no guidance in proposal)
  python posterior_compare.py --ckpt $CKPT \
    --out $OUTDIR/lag${LAG}_uncond_doob_eta1.0.json \
    $COMMON_ARGS --proposal uncond --guidance_type doob --guidance_eta 1.0 \
    > $LOGDIR/lag${LAG}_uncond_doob_eta1.0.log 2>&1

  for GT in doob natural; do
    for ETA in 0.1 0.5 1.0; do
      python posterior_compare.py --ckpt $CKPT \
        --out $OUTDIR/lag${LAG}_guided_${GT}_eta${ETA}.json \
        $COMMON_ARGS --proposal guided --guidance_type $GT --guidance_eta $ETA \
        > $LOGDIR/lag${LAG}_guided_${GT}_eta${ETA}.log 2>&1
    done
  done
done

echo "[sweep] done"
