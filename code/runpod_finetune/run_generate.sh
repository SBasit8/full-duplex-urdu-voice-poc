#!/bin/bash
# run_generate.sh — Test the fine-tuned model on held-out audio prompts.
# Run AFTER training completes and you've converted a checkpoint to fp32 safetensors.
set -e

NUM_GPUS=${NUM_GPUS:-1}
MODEL_DIR=${MODEL_DIR:-"/workspace/output/moshi-urdu-finetuned/step_LAST_fp32"}
EVAL_DATA=${EVAL_DATA:-"/workspace/processed/train-001-of-001.parquet"}   # for POC just reuse train, sample first few
PROMPT_LEN=${PROMPT_LEN:-125}    # 10s of context (125 frames @ 12.5Hz)
GEN_LEN=${GEN_LEN:-250}          # generate 20s of continuation
TEMP=${TEMP:-0.8}

cd /workspace/moshi-finetune

accelerate launch \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    generate.py \
        --output_dir "${MODEL_DIR}/continuations" \
        --model_dir "$MODEL_DIR" \
        --model_dtype float16 \
        --eval_data_files "$EVAL_DATA" \
        --example_length $((PROMPT_LEN + GEN_LEN)) \
        --prompt_length $PROMPT_LEN \
        --generation_length $GEN_LEN \
        --temperature $TEMP

echo ""
echo "=== Generation complete ==="
echo "Outputs in: ${MODEL_DIR}/continuations"
echo "Look for .wav files (synthesized audio) and JSON (text token sequences)"
