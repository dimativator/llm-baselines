#!/usr/bin/env bash
set -u
export PYTHONUNBUFFERED=1
export HF_HOME=/tmp/connectivity-hf-cache
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export WANDB_BASE_URL=https://wandb-radfan.ru
export WANDB_CONSOLE=off
export WANDB_DISABLE_CODE=true
export WANDB_CACHE_DIR=/tmp/connectivity-wandb-cache
export WANDB_CONFIG_DIR=/tmp/connectivity-wandb-config
export CONNECTIVITY_REVISION="$(git rev-parse HEAD)"
python scripts/cloud/connectivity_campaign.py "$@"
code=$?
echo "CONNECTIVITY_EXIT=$code"
# Preserve diagnostic scheduler output even if a worker failed.
exit 0
