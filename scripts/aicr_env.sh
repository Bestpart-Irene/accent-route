#!/bin/bash
# Shared environment header for AccentRoute jobs on AICR. SOURCED, not executed.
#
# Every byte of data, every cache and every checkpoint lives on /scratch: $HOME is a
# 100 GiB quota and a full home fails jobs instantly, and the project space is for code.

PROJ="${PROJ:-/work/neu/p2026_0038_neu}"
WORK="${WORK:-$PROJ/$USER}"
CODE_DIR="${CODE_DIR:-$WORK/accent-route}"
S="/scratch/$USER"

export PATH="$HOME/.local/bin:$PATH"          # uv
export UV_CACHE_DIR="$S/uv_cache"
export UV_PROJECT_ENVIRONMENT="$S/envs/accentroute"

cd "$CODE_DIR" || { echo "ERROR: CODE_DIR not found: $CODE_DIR" >&2; exit 1; }

mkdir -p "$S/logs" "$S/hf_cache/datasets" "$S/accentroute"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export HF_HOME="$S/hf_cache"
export HF_DATASETS_CACHE="$S/hf_cache/datasets"
export TOKENIZERS_PARALLELISM=false

# AccentRoute data layout on scratch (the smoke script reads these)
export ACCENTROUTE_RAW="$S/accentroute/raw/edacc_hf"
export ACCENTROUTE_WAV="$S/accentroute/work/edacc_wav"
export ACCENTROUTE_MANIFESTS="$S/accentroute/manifests"
export ACCENTROUTE_RUN="$S/accentroute/runs/smoke_edacc"
