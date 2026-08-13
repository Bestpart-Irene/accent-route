#!/bin/bash
# GPU stage of the EdAcc smoke test: LoRA training + evaluation.
#
# One GPU on b200-devel: this is a short diagnostic, devel has a 4 h limit and is
# usually idle, and a 1-GPU request backfills into any partially-used node instead of
# waiting for a whole one to drain.
#
#   sbatch --dependency=afterok:<prep_job_id> scripts/smoke_train.sbatch [steps]
#
#SBATCH --job-name=ar-smoke-train
#SBATCH --account=p2026_0038_neu
#SBATCH --partition=b200-devel
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/%u/logs/%x-%j.out
#SBATCH --error=/scratch/%u/logs/%x-%j.err
set -euo pipefail

source "${AICR_ENV:-/work/neu/p2026_0038_neu/$USER/accent-route/scripts/aicr_env.sh}"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

uv run python scripts/smoke_edacc.py train "${1:-600}"
uv run python scripts/smoke_edacc.py report

echo "DONE $SLURM_JOB_ID"
