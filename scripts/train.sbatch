#!/bin/bash
#SBATCH --job-name=ar-train
#SBATCH --account=p2026_0038_neu
#SBATCH --partition=rtx-batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/train-%j.out
#SBATCH --error=logs/train-%j.err
set -euo pipefail

# A single training run (frozen whisper-small encoder + LoRA, so VRAM stays small).
# Usage: sbatch scripts/train.sbatch <arm> <seed> [steps_c]
#   arm ∈ {a_gold, b_gold_oversampled, c_gold_weak, loso_l2}
#   b_gold_oversampled must be passed steps_c (the C arm's actual step count), otherwise
#   the budget protocol cannot be aligned.
ARM="${1:?usage: train.sbatch <arm> <seed> [steps_c]}"
SEED="${2:?usage: train.sbatch <arm> <seed> [steps_c]}"
STEPS_C="${3:-}"

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs runs

ARGS=(--common configs/train_common.yaml --arm "configs/arms/${ARM}.yaml" --seed "$SEED")
if [[ -n "$STEPS_C" ]]; then
  ARGS+=(--steps-c "$STEPS_C")
fi

uv run accentroute train run "${ARGS[@]}"
