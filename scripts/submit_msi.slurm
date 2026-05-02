#!/usr/bin/env bash
#SBATCH --job-name=physflow-earth
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=120:00:00
#SBATCH --output=logs/physflow_%j.out
#SBATCH --error=logs/physflow_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=arun08sharma@gmail.com

set -euo pipefail
module load python/3.11
module load cuda/12.4
source ~/envs/physflow-earth/bin/activate

EXP=${1:-sentinel2_x4}
OUT=outputs/${EXP}_${SLURM_JOB_ID}
mkdir -p logs "$OUT"

srun torchrun --nproc_per_node=4 -m physflow.training.train \
  --config-name $EXP \
  train.out_dir=$OUT
