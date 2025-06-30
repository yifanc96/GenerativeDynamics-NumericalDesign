#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=40:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END
#SBATCH --mail-user=yc3400@nyu.edu
#SBATCH --output=/home/yc3400/scratch_research/Forecasting-Interpolants/logs/%A_%a.out
#SBATCH --error=/home/yc3400/scratch_research/Forecasting-Interpolants/logs/%A_%a.err

## convenience definitions
SINGULARITY_IMAGE=/scratch/work/public/singularity/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif
OVERLAY_FILE=/scratch/yc3400/my_env/overlay-15GB-500K.ext3:ro
LAUNCHER_FILE=/home/yc3400/scratch_research/Forecasting-Interpolants/forecasting_Gaussbase-spectrumnoise-matchingvar-02182025-rougher-mul-k-quarter.py
PY_DIR=/home/yc3400/scratch_research/Forecasting-Interpolants
output_folder=/home/yc3400/scratch_research/Forecasting-Interpolants/logs

## modules provided by the singularity image
module purge


## do the damn thing
# srun singularity exec --nv --overlay $OVERLAY_FILE $SINGULARITY_IMAGE \
#         /bin/bash -c "source /ext3/env.sh; \
#         cd $PY_DIR; \
#         python $LAUNCHER_FILE" \s
# exit

singularity exec --nv --overlay $OVERLAY_FILE $SINGULARITY_IMAGE \
        /bin/bash -c "source /ext3/env.sh; \
        cd $PY_DIR; \
        python $LAUNCHER_FILE \
        --time_lag 2 \
        --noise_strength 0.0 \
        --lo_size 128 \
        --hi_size 128 \
        --max_steps 100000 \
        --base_lr 0.0001 \
        --data_subsampling_ratio 1.0 \
        --test_every 1000 \
        --sample_every 1000 \
        --num_dataset 5 \
        --batch_size 100" \
exit
