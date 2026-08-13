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

# Qwen2-Audio-7B-Instruct 弱标注推理(~17GB BF16 权重,24GB 卡够用)。
# 用法:sbatch scripts/weaklabel_qwen.sbatch <manifest.parquet> <out.parquet>
MANIFEST="${1:?usage: weaklabel_qwen.sbatch <manifest.parquet> <out.parquet>}"
OUT="${2:?usage: weaklabel_qwen.sbatch <manifest.parquet> <out.parquet>}"

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

# revision 未 pin 时直接失败,避免用漂移的模型产出弱标签
if grep -q 'revision: PIN_ME' configs/weaklabel.yaml; then
  echo "ERROR: configs/weaklabel.yaml still has revision: PIN_ME — pin the commit sha first" >&2
  exit 1
fi

uv run accentroute weaklabel run \
  --manifest "$MANIFEST" \
  --out "$OUT" \
  --config configs/weaklabel.yaml \
  --wav-dir data/work/youtube_wav
