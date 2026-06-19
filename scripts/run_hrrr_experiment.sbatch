#!/bin/bash
#SBATCH --job-name=physflow-hrrr
#SBATCH --account=shekhars
#SBATCH --partition=msigpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:30:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
# HRRR 8x SR de-risking run: physics vs no-physics vs bicubic, 3 seeds + conformal.
# Runs in the pcrf conda env directly (torch 2.5.1+cu121); no container needed.
set -euo pipefail

REPO="${PHYSFLOW_REPO:-$HOME/paper-work/physflow-earth}"
OUT="${PHYSFLOW_OUT:-/scratch.global/arunshar/physflow_hrrr}"
ROOT="${PHYSFLOW_DATA:-/scratch.global/arunshar/hrrr128}"
STEPS="${PHYSFLOW_STEPS:-3000}"
DEG="${PHYSFLOW_DEG:-pool}"
HIDDEN="${PHYSFLOW_HIDDEN:-256}"
DEPTH="${PHYSFLOW_DEPTH:-6}"
DIVW="${PHYSFLOW_DIVW:-0.1}"

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate pcrf

echo "[physflow-hrrr] node=$(hostname) gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"
echo "[physflow-hrrr] repo=$REPO out=$OUT steps=$STEPS deg=$DEG hidden=$HIDDEN depth=$DEPTH div_w=$DIVW"
mkdir -p "$OUT"

cd "$REPO"
PYTHONPATH="$REPO/src" python scripts/run_hrrr_experiment.py \
  --root "$ROOT" --out "$OUT" --steps "$STEPS" --seeds 0 1 2 --device cuda \
  --degradation "$DEG" --hidden "$HIDDEN" --depth "$DEPTH" --div-w "$DIVW"

echo "[physflow-hrrr] done rc=$?"
