#!/usr/bin/env bash
set -euo pipefail
export PYTHONUSERBASE="${PYTHONUSERBASE:-/home/jovyan/.local-efficient}"
export WANDB_BASE_URL=https://wandb-radfan.ru
export HF_HOME=/tmp/rank-profile-hf
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
mkdir -p "${HF_HOME}"
if command -v nvidia-smi >/dev/null; then
    nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv
    nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv
fi
python scripts/cloud/rank_profiles.py "$@"
