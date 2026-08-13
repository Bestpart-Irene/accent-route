#!/bin/bash
# Fetch a balanced GLOBE subset and run it through the pipeline. CPU only — streaming,
# decoding, VAD and splitting use no GPU, and an idle GPU allocation is charged to the
# shared fairshare account.
#
#   sbatch scripts/globe_prepare.sbatch [per_class] [max_per_speaker]
#
#SBATCH --job-name=ar-globe-prep
#SBATCH --account=p2026_0038_neu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/%u/logs/%x-%j.out
#SBATCH --error=/scratch/%u/logs/%x-%j.err
set -euo pipefail

source "${AICR_ENV:-/work/neu/p2026_0038_neu/$USER/accent-route/scripts/aicr_env.sh}"

export GLOBE_WAV="/scratch/$USER/accentroute/work/globe_wav"
export GLOBE_MANIFESTS="/scratch/$USER/accentroute/manifests"

uv run python scripts/globe_pipeline.py fetch "${1:-1200}" "${2:-15}"
uv run python scripts/globe_pipeline.py pipeline
uv run python scripts/globe_pipeline.py report

echo "DONE $SLURM_JOB_ID"
