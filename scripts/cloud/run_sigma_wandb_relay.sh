#!/usr/bin/env bash
set -u
export PYTHONUNBUFFERED=1
export WANDB_BASE_URL=https://wandb-radfan.ru
export WANDB_MODE=online
export WANDB_CONSOLE=off
export WANDB_DISABLE_CODE=true
export WANDB_CACHE_DIR=/tmp/sigma-wandb-cache
export WANDB_CONFIG_DIR=/tmp/sigma-wandb-config
python scripts/cloud/sigma_wandb_relay.py
code=$?
echo "SIGMA_WANDB_RELAY_EXIT=$code"
exit 0
