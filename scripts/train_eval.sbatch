#!/bin/bash
# Train the accent classifier over three seeds and evaluate in-domain + out-of-domain.
# One GPU: the encoder is frozen and only LoRA plus the head are trained, so a single
# device is plenty and a 1-GPU request backfills into any partially-used node.
#
#   sbatch scripts/train_eval.sbatch [steps]
#
#SBATCH --job-name=ar-train-eval
#SBATCH --account=p2026_0038_neu
#SBATCH --partition=b200-batch
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/%u/logs/%x-%j.out
#SBATCH --error=/scratch/%u/logs/%x-%j.err
set -euo pipefail

source "${AICR_ENV:-/work/neu/p2026_0038_neu/$USER/accent-route/scripts/aicr_env.sh}"

export GLOBE_WAV="/scratch/$USER/accentroute/work/globe_wav"
export GLOBE_MANIFESTS="/scratch/$USER/accentroute/manifests"
export TRAINEVAL_RUN="/scratch/$USER/accentroute/runs/accent_v1"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

uv run python scripts/train_eval.py train "${1:-1500}"
uv run python scripts/train_eval.py evaluate

echo "DONE $SLURM_JOB_ID"
