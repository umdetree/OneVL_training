#!/bin/bash
set -x

# ============================================================
# Visual Aux Decoder Pretrain (Stage 0.5)
#
# Pretrain the visual aux decoder as an independent future-frame
# generator before integrating it into the full OneVL pipeline.
#
# Main model (frozen): extract ViT embeddings only
# Visual aux decoder (trainable): [ViT_embeds] → [future_tokens]
# No latent tokens, no text aux decoder.
# ============================================================

# ---------- Environment ----------
source /root/autodl-tmp/OneVL_training/venv/onevl/bin/activate

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=3
export USE_HF=1

# use hf mirror to avoid download failure in China
export HF_ENDPOINT="https://hf-mirror.com"

# ---------- Distributed settings ----------
nproc_per_node=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
NNODES=${WORKER_NUM:-1}
NODE_RANK=${ROLE_INDEX:-0}
MASTER_ADDR=${WORKER_0_HOST:-127.0.0.1}
MASTER_PORT=${WORKER_0_PORT:-29500}

# ---------- Model paths ----------
MODEL_PATH="Qwen/Qwen3-VL-2B-Instruct"
VISUAL_AUX_MODEL_PATH="Qwen/Qwen3-VL-2B-Instruct"
DATASET_PATH="${SCRIPT_DIR}/demo_data/navsim/navsim_vis4_text2_demo100.jsonl"

# ---------- Visual Aux Pretrain configuration ----------
export LATENT_COT_VISUAL_AUX_MODEL_PATH="${VISUAL_AUX_MODEL_PATH}"
export LATENT_COT_VISUAL_AUX_VISUAL_CONDITION=true
export LATENT_COT_VISUAL_EXPLAIN_LOSS_WEIGHT=1.0

# ---------- Launch training ----------
mkdir -p "${SCRIPT_DIR}/logs/navsim"
mkdir -p "${SCRIPT_DIR}/outputs/navsim"

CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((nproc_per_node-1))) \
NPROC_PER_NODE=$nproc_per_node \
NNODES=$NNODES \
NODE_RANK=$NODE_RANK \
MASTER_ADDR=$MASTER_ADDR \
MASTER_PORT=$MASTER_PORT \
swift sft \
    --model "${MODEL_PATH}" \
    --model_type qwen3_vl_visual_aux_pretrain \
    --template qwen3_vl_visual_aux_pretrain \
    --train_type full \
    --dataset "${DATASET_PATH}" \
    --torch_dtype bfloat16 \
    --num_train_epochs 20000 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --loss_type latent_cot \
    --lr_scheduler_type cosine \
    --gradient_accumulation_steps 2 \
    --save_steps 100 \
    --eval_steps 100 \
    --save_total_limit 1 \
    --logging_steps 5 \
    --max_length 4096 \
    --warmup_steps 10 \
    --weight_decay 0.05 \
    --freeze_vit true \
    --freeze_llm true \
    --freeze_aligner true \
    --dataloader_num_workers 2 \
    --output_dir "${SCRIPT_DIR}/outputs/navsim/qwen3_vl_visual_aux_pretrain_stage0_5_vis4_txt2" \
    --resume_from_checkpoint "${SCRIPT_DIR}/outputs/navsim/qwen3_vl_visual_aux_pretrain_stage0_5_vis4_txt2/v0-20260715-113754/checkpoint-100" \
    --gradient_checkpointing true \
    --deepspeed zero3 \
  2>&1 | tee "${SCRIPT_DIR}/logs/navsim/qwen3_vl_visual_aux_pretrain_stage0_5_vis4_txt2.log"
