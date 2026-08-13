#!/bin/bash
# CPU stage of the EdAcc smoke test: download → extract WAVs → ingest/filter/split.
# No GPU is requested, because none is used — an idle GPU allocation is charged to the
# shared fairshare account and lowers everyone's queue priority.
#
#   sbatch scripts/smoke_prepare.sbatch
#
#SBATCH --job-name=ar-smoke-prep
#SBATCH --account=p2026_0038_neu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/%u/logs/%x-%j.out
#SBATCH --error=/scratch/%u/logs/%x-%j.err
set -euo pipefail

source "${AICR_ENV:-/work/neu/p2026_0038_neu/$USER/accent-route/scripts/aicr_env.sh}"

uv run python scripts/smoke_edacc.py download
uv run python scripts/smoke_edacc.py extract
uv run python scripts/smoke_edacc.py pipeline

echo "DONE $SLURM_JOB_ID"
