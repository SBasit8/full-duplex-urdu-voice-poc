#!/bin/bash
# run_generate_few.sh — Generate a small number of sample continuations.
# Used for POC validation, not full eval.
set -e

NUM_GPUS=${NUM_GPUS:-1}
MODEL_DIR=${MODEL_DIR:-/workspace/output/moshi-urdu-finetuned/step_110}
EVAL_DATA=${EVAL_DATA:-/workspace/processed/train-001-of-001.parquet}
NUM_EXAMPLES=${NUM_EXAMPLES:-10}
PROMPT_LEN=${PROMPT_LEN:-125}
GEN_LEN=${GEN_LEN:-250}
TEMP=${TEMP:-0.8}

cd /workspace/moshi-finetune

accelerate launch \
    --num_machines 1 \
    --num_processes "$NUM_GPUS" \
    generate.py \
    --output_dir "${MODEL_DIR}/continuations" \
    --model_dir "${MODEL_DIR}" \
    --model_dtype float16 \
    --eval_data_files "${EVAL_DATA}" \
    --num_examples "${NUM_EXAMPLES}" \
    --example_length $((PROMPT_LEN + GEN_LEN)) \
    --prompt_length "${PROMPT_LEN}" \
    --generation_length "${GEN_LEN}" \
    --temperature "${TEMP}"
