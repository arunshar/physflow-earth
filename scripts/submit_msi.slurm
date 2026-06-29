#!/usr/bin/env bash
#SBATCH --job-name=physflow-earth
#SBATCH --account=shekhars
#SBATCH --partition=a100-4
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/physflow_%j.out
#SBATCH --error=logs/physflow_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=arun08sharma@gmail.com

set -euo pipefail
# Known-good MSI GPU env (a100-4). One-time setup, see scripts/MSI_RUNBOOK.md:
#   conda create -n physflow python=3.11 && source activate physflow && pip install -e .
module load gcc/13.1.0-5z64cho
module load cuda/12.4
source activate ${PROJ_ENV:-physflow}
unset LD_LIBRARY_PATH   # MSI modules clobber LD_LIBRARY_PATH and break conda CUDA libs
# 1-GPU smoke: --gres=gpu:a100:1 + torchrun --nproc_per_node=1; v100 fallback: --partition=v100 --gres=gpu:v100:1

EXP=${1:-sentinel2_x4}
OUT=outputs/${EXP}_${SLURM_JOB_ID}
mkdir -p logs "$OUT"

srun torchrun --nproc_per_node=4 -m physflow.training.train \
  --config-name $EXP \
  train.out_dir=$OUT
