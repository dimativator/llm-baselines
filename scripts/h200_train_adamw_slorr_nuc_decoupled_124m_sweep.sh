#!/bin/bash
set -euo pipefail

GPU=${GPU:?Set the physical GPU index}
read -r -a COEFFICIENT_VALUES <<< "${COEFFICIENTS:?Set space-separated COEFFICIENTS}"

DATASETS_DIR=${DATASETS_DIR:-"/data/users/dimativator/llm-baselines-soap/datasets"}
TOKENIZED_DATA_DIR=${TOKENIZED_DATA_DIR:-"/data/users/dimativator/llm-baselines-soap/tokenized"}
RUN_ROOT=${RUN_ROOT:-"$PWD/exps/adamw_slorr_nuc_decoupled_124m_h200"}
PYTHON_BIN=${PYTHON_BIN:-"/data/users/dimativator/anaconda3/envs/eff-pretrain/bin/python"}
LR=${LR:-5e-3}
WEIGHT_DECAY=${WEIGHT_DECAY:-1e-1}
ITERATIONS=${ITERATIONS:-19000}
WARMUP_STEPS=${WARMUP_STEPS:-2000}
BATCH_SIZE=${BATCH_SIZE:-16}
ACC_STEPS=${ACC_STEPS:-8}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-16}
EVAL_BATCHES=${EVAL_BATCHES:-256}
LATEST_CKPT_INTERVAL=${LATEST_CKPT_INTERVAL:-1000}
DOWNSTREAM_EVAL_INTERVAL=${DOWNSTREAM_EVAL_INTERVAL:-2000}

if [ $((BATCH_SIZE * ACC_STEPS)) -ne 128 ]; then
    echo "Expected effective batch 128, got batch_size=$BATCH_SIZE acc_steps=$ACC_STEPS" >&2
    exit 2
fi

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

test -s "$TOKENIZED_DATA_DIR/train.bin"
test -s "$TOKENIZED_DATA_DIR/val.bin"
mkdir -p "$RUN_ROOT"

"$PYTHON_BIN" -c 'import torch; free, total = torch.cuda.mem_get_info(); print(f"CUDA_PREFLIGHT=OK gpu={torch.cuda.get_device_name(0)} free={free / 2**30:.1f}GiB total={total / 2**30:.1f}GiB")'

for COEFFICIENT in "${COEFFICIENT_VALUES[@]}"; do
    EXP_NAME="h200_gpu${GPU}_llama124m_adamw_slorr_nuc_decoupled_cf${COEFFICIENT}_lr${LR}_bs${BATCH_SIZE}a${ACC_STEPS}_finewebedu"
    EXP_DIR="$RUN_ROOT/$EXP_NAME"
    if [ -f "$EXP_DIR/COMPLETE" ]; then
        echo "SKIP_COMPLETE cf=$COEFFICIENT exp=$EXP_NAME"
        continue
    fi

    echo "RUN_START cf=$COEFFICIENT exp=$EXP_NAME"
    "$PYTHON_BIN" ./src/main.py \
        --experiment_name "$EXP_NAME" \
        --results_base_folder "$RUN_ROOT" \
        --device cuda:0 \
        --model llama \
        --datasets_dir "$DATASETS_DIR" \
        --dataset finewebedu \
        --opt adamw-spectral-l1-reg \
        --lr "$LR" \
        --iterations "$ITERATIONS" \
        --n_embd 768 \
        --n_head 12 \
        --n_layer 12 \
        --batch_size "$BATCH_SIZE" \
        --sequence_length 1024 \
        --acc_steps "$ACC_STEPS" \
        --grad_clip 0.5 \
        --seed 0 \
        --weight_decay "$WEIGHT_DECAY" \
        --spectral_l1_reg_coef "$COEFFICIENT" \
        --spectral_l1_reg_pre_update \
        --spectral_l1_svt_interval 0 \
        --scheduler cos \
        --warmup_steps "$WARMUP_STEPS" \
        --dropout 0 \
        --beta1 0.9 \
        --beta2 0.95 \
        --eval_interval 137 \
        --eval_batches "$EVAL_BATCHES" \
        --eval_batch_size "$EVAL_BATCH_SIZE" \
        --latest_ckpt_interval "$LATEST_CKPT_INTERVAL" \
        --permanent_ckpt_interval 0 \
        --log_interval 4 \
        --finewebedu_max_files 5 \
        --tokenized_data_dir "$TOKENIZED_DATA_DIR" \
        --effective_rank_interval 500 \
        --downstream_eval_enabled \
        --downstream_eval_interval "$DOWNSTREAM_EVAL_INTERVAL" \
        --downstream_task_group basic_v2

    touch "$EXP_DIR/COMPLETE"
    echo "RUN_COMPLETE cf=$COEFFICIENT exp=$EXP_NAME"
done

echo "SWEEP_COMPLETE gpu=$GPU coefficients=${COEFFICIENT_VALUES[*]}"
