#!/usr/bin/env bash
set -euo pipefail
export PYTHONUSERBASE="${PYTHONUSERBASE:-/home/jovyan/.local-efficient}"
export WANDB_MODE=offline
export WANDB_DIR=/tmp
export HF_HOME=/tmp/spectral-noembed-hf
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
export RANK_CODE_REVISION="$(git rev-parse HEAD)"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv || true
python scripts/cloud/spectral_noembed.py "$@"
