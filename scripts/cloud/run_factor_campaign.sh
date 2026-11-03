#!/usr/bin/env bash
set -u
export FACTOR_TUNING=1
export POST_LR=0.0001
export SIGMA_ROOT=${SIGMA_ROOT:-/workspace-SR006.nfs3/dimativator/factor-tuning-124m-20261103}
export PYTHONUNBUFFERED=1
export HF_HOME=/tmp/factor-hf-cache
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export WANDB_BASE_URL=https://wandb-radfan.ru
export WANDB_MODE=online
export WANDB_CONSOLE=off
export WANDB_DISABLE_CODE=true
export WANDB_CACHE_DIR=/tmp/factor-wandb-cache
export WANDB_CONFIG_DIR=/tmp/factor-wandb-config

if [[ ${1:-} == --inspect ]]; then
    python scripts/cloud/sigma_campaign.py --inspect
elif [[ ${1:-} == --smoke ]]; then
    export SIGMA_CPU_SMOKE=1
    python scripts/cloud/sigma_campaign.py --smoke
else
    export CAMPAIGN_RELAY_EXIT=/tmp/factor-relay-exit-20261103
    setsid bash -c 'python scripts/cloud/sigma_wandb_relay.py > /tmp/factor-relay.log 2>&1; code=$?; echo "$code" > "$CAMPAIGN_RELAY_EXIT"; exit "$code"' &
    relay=$!
    python scripts/cloud/sigma_campaign.py
    code=$?
    if [[ $code != 0 ]]; then
        kill -- -"$relay" 2>/dev/null || true
    fi
    wait "$relay"
    relay_code=$?
    tail -60 /tmp/factor-relay.log
    echo "FACTOR_COORDINATOR_EXIT=$code RELAY_EXIT=$relay_code"
    # Include the aggregate completion and checkpoint absence in scheduler logs.
    python scripts/cloud/sigma_campaign.py --inspect
fi
# Cloud.ru otherwise hides diagnostic logs for failed jobs.
exit 0
