#!/bin/bash
set -x

# ============================================================
# Visual Aux Decoder Pretrain (Stage 0.5) — Pure HF Trainer
# ============================================================

# ---------- Environment ----------
source /root/autodl-tmp/OneVL_training/venv/onevl/bin/activate

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=3
export HF_ENDPOINT="https://hf-mirror.com"

# ---------- Distributed settings ----------
nproc_per_node=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
NNODES=${WORKER_NUM:-1}
NODE_RANK=${ROLE_INDEX:-0}
MASTER_ADDR=${WORKER_0_HOST:-127.0.0.1}
MASTER_PORT=${WORKER_0_PORT:-29500}

# ---------- Paths ----------
MODEL_NAME_OR_PATH="${SCRIPT_DIR}/outputs/navsim/qwen3_vl_visual_aux_pretrain_stage0_5_vis4_txt2/checkpoint-2300"
DATASET_PATH="${SCRIPT_DIR}/data/navsim/data.jsonl"
OUTPUT_DIR="${SCRIPT_DIR}/outputs/navsim/stage05_pure"
LOG_DIR="${SCRIPT_DIR}/logs/navsim"
DEEPSPEED_CONFIG="${SCRIPT_DIR}/ds_configs/zero3.json"

mkdir -p "${LOG_DIR}"
mkdir -p "$(dirname "${OUTPUT_DIR}")"

torchrun \
    --nproc_per_node=${nproc_per_node} \
    --nnodes=${NNODES} \
    --node_rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    --module scripts.stage05_train \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --data_path "${DATASET_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --max_steps 2000000 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-4 \
    --lr_scheduler_type cosine \
    --warmup_steps 10 \
    --weight_decay 0.05 \
    --logging_steps 1 \
    --save_steps 100 \
    --save_total_limit 1 \
    --dataloader_num_workers 0 \
    --gradient_checkpointing True \
    --deepspeed "${DEEPSPEED_CONFIG}" \
    --freeze_vit True \
    --remove_unused_columns false \
    2>&1 | tee "${LOG_DIR}/stage05_pure.log"