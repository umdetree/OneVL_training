#!/bin/bash
# Multi-Node + Multi-GPU parallel OneVL inference using qwen3_vl_infer_onevl.py
# Supports single-node (same as original) or multi-node via env vars.
# Requires shared filesystem for OUTPUT_DIR and TEST_SET_PATH (all nodes must see same paths).
#
# Single-node: ./infer_qwen3_vl_all_multinodes.sh   (same behavior as original script)
#
# Multi-node: use same env as train.sh (* from launcher).
#   NNODES=${WORKER_NUM:-1}, NODE_RANK=${ROLE_INDEX:-0}
#   When NNODES>1, set RUN_ID to same value on all nodes if launcher does not set it.
#   Assumes same number of GPUs per node.
set -e

PYTHON=projects/ms-swift/.venv/bin/python3

# ---- Configuration (edit these) ----
MODEL_PATH=outputs/navsim/qwen3_vl_latent_cot_stage2_vis4_txt2_fixbug_256_bs64_notext/v0-20260323-132232/checkpoint-4842
TEST_SET_PATH=projects/ms-swift/data/navsim_test_cot_full_idx_trainfmt.json
OUTPUT_PATH=${MODEL_PATH}/infer_results_prefill/qwen3_vl_infer_onevl_merged.json
OUTPUT_PATH_EVAL=${MODEL_PATH}/infer_results_prefill/qwen3_vl_infer_onevl_merged_eval.json

# ---- OneVL / Latent CoT hyperparameters ----
NUM_LATENT=2
NUM_LATENT_VIS=4
MAX_NEW_TOKENS=1024

# Decoder explain: set to "true" to enable aux text decoder explaining latent reasoning
# Requires AUX_MODEL_PATH to be set.
DECODER_EXPLAIN=${DECODER_EXPLAIN:-false}
AUX_MODEL_PATH=${AUX_MODEL_PATH:-"/qwen3vl/Qwen3-VL-2B-Instruct"}
AUX_VISUAL_CONDITION=${AUX_VISUAL_CONDITION:-false}
C_THOUGHT=${C_THOUGHT:-2}
MAX_EXPLAIN_TOKENS=${MAX_EXPLAIN_TOKENS:-512}
ADD_ASSISTANT_PREFIX="--add_assistant_prefix"

# Visual decoder explain: set to "true" to enable visual aux decoder
# Requires VISUAL_AUX_MODEL_PATH to be set.
VISUAL_DECODER_EXPLAIN=${VISUAL_DECODER_EXPLAIN:-false}
VISUAL_AUX_MODEL_PATH=${VISUAL_AUX_MODEL_PATH:-"models/visual_aux_decoder/qwen3_vl_visual_aux_decoder_ad/checkpoints/global_step_13040/hf_ckpt"}
VISUAL_AUX_VISUAL_CONDITION=${VISUAL_AUX_VISUAL_CONDITION:-true}
C_THOUGHT_VISUAL=${C_THOUGHT_VISUAL:-4}
MAX_VISUAL_TOKENS=${MAX_VISUAL_TOKENS:-1024}

# Original vocab / all subtokens / separate visual latent tokens (match training)
USE_ORIGINAL_VOCAB=${USE_ORIGINAL_VOCAB:-true}
USE_ALL_SUBTOKENS=${USE_ALL_SUBTOKENS:-true}
USE_SEPARATE_VISUAL_LATENT_TOKENS=${USE_SEPARATE_VISUAL_LATENT_TOKENS:-true}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
INFER_SCRIPT="${SCRIPT_DIR}/../qwen3_vl_infer_onevl.py"

# Build extra flags for decoder explain
EXTRA_FLAGS=""
if [ "${DECODER_EXPLAIN}" = "true" ]; then
    EXTRA_FLAGS="${EXTRA_FLAGS} --decoder_explain --aux_model_path ${AUX_MODEL_PATH} --c_thought ${C_THOUGHT} --max_explain_tokens ${MAX_EXPLAIN_TOKENS}"
    if [ "${AUX_VISUAL_CONDITION}" = "true" ]; then
        EXTRA_FLAGS="${EXTRA_FLAGS} --aux_visual_condition"
    fi
fi
if [ "${VISUAL_DECODER_EXPLAIN}" = "true" ]; then
    EXTRA_FLAGS="${EXTRA_FLAGS} --visual_decoder_explain --visual_aux_model_path ${VISUAL_AUX_MODEL_PATH} --c_thought_visual ${C_THOUGHT_VISUAL} --max_visual_tokens ${MAX_VISUAL_TOKENS}"
    if [ "${VISUAL_AUX_VISUAL_CONDITION}" = "true" ]; then
        EXTRA_FLAGS="${EXTRA_FLAGS} --visual_aux_visual_condition"
    fi
fi
if [ "${USE_ORIGINAL_VOCAB}" = "true" ]; then
    EXTRA_FLAGS="${EXTRA_FLAGS} --use_original_vocab"
fi
if [ "${USE_ALL_SUBTOKENS}" = "true" ]; then
    EXTRA_FLAGS="${EXTRA_FLAGS} --use_all_subtokens"
fi
if [ "${USE_SEPARATE_VISUAL_LATENT_TOKENS}" = "true" ]; then
    EXTRA_FLAGS="${EXTRA_FLAGS} --use_separate_visual_latent_tokens"
fi

echo "=== OneVL Inference Configuration ==="
echo "  MODEL_PATH:              ${MODEL_PATH}"
echo "  DECODER_EXPLAIN:         ${DECODER_EXPLAIN}"
echo "  AUX_VISUAL_CONDITION:    ${AUX_VISUAL_CONDITION}"
echo "  C_THOUGHT:               ${C_THOUGHT}"
echo "  VISUAL_DECODER_EXPLAIN:  ${VISUAL_DECODER_EXPLAIN}"
echo "  VISUAL_AUX_VISUAL_COND:  ${VISUAL_AUX_VISUAL_CONDITION}"
echo "  C_THOUGHT_VISUAL:        ${C_THOUGHT_VISUAL}"
echo "  USE_ORIGINAL_VOCAB:      ${USE_ORIGINAL_VOCAB}"
echo "  USE_ALL_SUBTOKENS:       ${USE_ALL_SUBTOKENS}"
echo "  USE_SEPARATE_VIS_TOKENS: ${USE_SEPARATE_VISUAL_LATENT_TOKENS}"
echo "  EXTRA_FLAGS:             ${EXTRA_FLAGS}"
echo "======================================"

# ---- Multi-node: same as train.sh (WORKER_NUM, ROLE_INDEX) ----
NNODES=${WORKER_NUM:-1}
NODE_RANK=${ROLE_INDEX:-0}
MASTER_ADDR=${WORKER_0_HOST:-"127.0.0.1"}
MASTER_PORT=${WORKER_0_PORT:-$(shuf -i 10000-50000 -n1)}

OUTPUT_DIR=$(dirname "${OUTPUT_PATH}")
RUN_ID=${RUN_ID:-${JOB_ID:-${SLURM_JOB_ID:-$$}}}

if [ "${NNODES}" -gt 1 ] && [ "${RUN_ID}" = "$$" ]; then
    INFER_RUN_ID_FILE="${OUTPUT_DIR}/._infer_multinode_run_id"
    if [ "${NODE_RANK}" -eq 0 ]; then
        mkdir -p "${OUTPUT_DIR}"
        RUN_ID=$(date +%s)
        echo "${RUN_ID}" > "${INFER_RUN_ID_FILE}"
        sync
        echo "Rank 0: set RUN_ID=${RUN_ID} for this multi-node run"
    else
        echo "[Node ${NODE_RANK}] Waiting for RUN_ID file from rank 0 (ensure OUTPUT_DIR is on shared storage) ..."
        while [ ! -f "${INFER_RUN_ID_FILE}" ]; do sleep 2; done
        RUN_ID=$(cat "${INFER_RUN_ID_FILE}")
        echo "[Node ${NODE_RANK}] Got RUN_ID=${RUN_ID}"
    fi
fi

# ---- Step 1: Detect local GPUs ----
NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "[Node ${NODE_RANK}] === Detected ${NUM_GPUS} GPUs (node ${NODE_RANK}/${NNODES}) ==="

TOTAL_WORKERS=$((NNODES * NUM_GPUS))
MY_SHARD_START=$((NODE_RANK * NUM_GPUS))
MY_SHARD_END=$(((NODE_RANK + 1) * NUM_GPUS))

SPLIT_DIR="${OUTPUT_DIR}/_splits_${RUN_ID}"
BARRIER_DIR="${SPLIT_DIR}"
DONE_FILE="${BARRIER_DIR}/done.${NODE_RANK}"
SPLIT_DONE_FILE="${BARRIER_DIR}/split_done"

# ---- Step 2: Split data (only rank 0 in multi-node, or single-node) ----
if [ "${NODE_RANK}" -eq 0 ]; then
    mkdir -p "${SPLIT_DIR}"
    echo "=== Splitting test set into ${TOTAL_WORKERS} shards ==="
    $PYTHON -c "
import json, math
with open('${TEST_SET_PATH}') as f:
    data = json.load(f)
n = ${TOTAL_WORKERS}
shard_size = math.ceil(len(data) / n) if n else 0
for i in range(n):
    shard = data[i*shard_size : (i+1)*shard_size]
    if not shard:
        continue
    with open(f'${SPLIT_DIR}/shard_{i}.json', 'w') as f:
        json.dump(shard, f, ensure_ascii=False)
    print(f'Shard {i}: {len(shard)} samples')
"
    touch "${SPLIT_DONE_FILE}"
else
    echo "[Node ${NODE_RANK}] Waiting for rank 0 to finish splitting ..."
    while [ ! -f "${SPLIT_DONE_FILE}" ]; do sleep 2; done
    echo "[Node ${NODE_RANK}] Split ready."
fi

# ---- Step 3: Launch inference for this node's shards only ----
echo "[Node ${NODE_RANK}] === Launching inference for shards ${MY_SHARD_START}..$((MY_SHARD_END - 1)) on ${NUM_GPUS} GPUs ==="
PIDS=()
LOCAL_GPU=0
for SHARD_ID in $(seq ${MY_SHARD_START} $((MY_SHARD_END - 1))); do
    SHARD_INPUT="${SPLIT_DIR}/shard_${SHARD_ID}.json"
    SHARD_OUTPUT="${SPLIT_DIR}/predict_${SHARD_ID}.json"

    if [ ! -f "${SHARD_INPUT}" ]; then
        echo "  [Node ${NODE_RANK}] Shard ${SHARD_ID}: no file, skipping"
        LOCAL_GPU=$((LOCAL_GPU + 1))
        continue
    fi

    echo "  [Node ${NODE_RANK}] Shard ${SHARD_ID} on cuda:${LOCAL_GPU} ..."
    $PYTHON "${INFER_SCRIPT}" \
        --model_path "${MODEL_PATH}" \
        --test_set_path "${SHARD_INPUT}" \
        --output_path "${SHARD_OUTPUT}" \
        --device "cuda:${LOCAL_GPU}" \
        --num_latent ${NUM_LATENT} \
        --num_latent-vis ${NUM_LATENT_VIS} \
        --max_new_tokens ${MAX_NEW_TOKENS} \
        ${ADD_ASSISTANT_PREFIX} \
        ${EXTRA_FLAGS} &

    PIDS+=($!)
    LOCAL_GPU=$((LOCAL_GPU + 1))
done

FAIL=0
for PID in "${PIDS[@]}"; do
    wait ${PID} || FAIL=1
done

if [ ${FAIL} -ne 0 ]; then
    echo "[Node ${NODE_RANK}] ERROR: one or more inference processes failed"
    exit 1
fi

touch "${DONE_FILE}"
echo "[Node ${NODE_RANK}] === Inference done, wrote ${DONE_FILE} ==="

# ---- Step 4: Merge (only rank 0); others wait then exit ----
if [ "${NODE_RANK}" -ne 0 ]; then
    echo "[Node ${NODE_RANK}] Exit (merge on rank 0)."
    exit 0
fi

echo "=== Rank 0: Waiting for all nodes to finish inference ==="
for r in $(seq 0 $((NNODES - 1))); do
    while [ ! -f "${BARRIER_DIR}/done.${r}" ]; do sleep 3; done
    echo "  Node ${r} done."
done

echo "=== All nodes done, merging results ==="
$PYTHON -c "
import json, glob
shards = sorted(glob.glob('${SPLIT_DIR}/predict_*.json'), key=lambda p: int(p.rsplit('_', 1)[1].replace('.json', '')))
merged = []
for path in shards:
    with open(path) as f:
        data = json.load(f)
        merged.extend(data)
with open('${OUTPUT_PATH}', 'w') as f:
    json.dump(merged, f, indent=4, ensure_ascii=False)
print(f'Merged {len(merged)} samples from {len(shards)} shards -> ${OUTPUT_PATH}')
"

rm -rf "${SPLIT_DIR}"
echo "=== Done. Output saved to ${OUTPUT_PATH} ==="

## Step 5: Convert to eval format
$PYTHON "${SCRIPT_DIR}/convert_to_eval.py" \
    --input_path "${OUTPUT_PATH}" \
    --output_path "${OUTPUT_PATH_EVAL}"