#!/usr/bin/env bash
set -u
export CONNECTIVITY_MODEL_SIZE=124m
export CONNECTIVITY_ROOT=${CONNECTIVITY_ROOT:-/workspace-SR006.nfs3/dimativator/mode-connectivity-124m-20261103}
export CONNECTIVITY_HF_PREFIX=mode_connectivity_124m_20261103
export CONNECTIVITY_CHECKPOINT_RESERVE=2200000000
bash scripts/cloud/run_connectivity_campaign.sh "$@"
