#!/bin/bash
#SBATCH --job-name=ar-weaklabel
#SBATCH --account=p2026_0038_neu
#SBATCH --partition=rtx-batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/weaklabel-%j.out
#SBATCH --error=logs/weaklabel-%j.err
set -euo pipefail

# Qwen2-Audio-7B-Instruct weak-label inference (~17GB of BF16 weights, so a 24GB card is
# enough).
# Usage: sbatch scripts/weaklabel_qwen.sbatch <manifest.parquet> <out.parquet>
MANIFEST="${1:?usage: weaklabel_qwen.sbatch <manifest.parquet> <out.parquet>}"
OUT="${2:?usage: weaklabel_qwen.sbatch <manifest.parquet> <out.parquet>}"

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

# Fail fast while revision is unpinned — never mint weak labels from a drifting model
if grep -q 'revision: PIN_ME' configs/weaklabel.yaml; then
  echo "ERROR: configs/weaklabel.yaml still has revision: PIN_ME — pin the commit sha first" >&2
  exit 1
fi

uv run accentroute weaklabel run \
  --manifest "$MANIFEST" \
  --out "$OUT" \
  --config configs/weaklabel.yaml \
  --wav-dir data/work/youtube_wav
