#!/bin/bash
# run_finetune.sh — Launch fine-tuning with accelerate + deepspeed Zero 3.
# Tweak NUM_GPUS to match your pod (2, 4, or 8).
set -e

# ====================================
# Configuration — edit these as needed
# ====================================
NUM_GPUS=${NUM_GPUS:-2}                       # number of GPUs in your pod
TRAIN_DATA="/workspace/processed/train-*.parquet"
INIT_MODEL_DIR="/workspace/init_models/moshiko-urdu-float32"
OUTPUT_DIR="/workspace/output/moshi-urdu-finetuned"
NUM_EPOCHS=${NUM_EPOCHS:-5}                   # POC: 3-5, real run: 10-15
PER_DEVICE_BATCH=${PER_DEVICE_BATCH:-4}       # lower to 2 if OOM on smaller GPUs
GRAD_ACCUM=${GRAD_ACCUM:-8}                   # effective batch = NUM_GPUS * BATCH * GRAD_ACCUM
MAX_LENGTH=${MAX_LENGTH:-2048}                # frames per training example (164s @ 12.5Hz)
SAVE_STEPS=${SAVE_STEPS:-100}                 # save checkpoint every N steps
LOGGING_STEPS=${LOGGING_STEPS:-1}             # print loss every N steps
DS_CONFIG="ds_configs/zero3-fp16-warmlr-act_ckpt.json"

# Wandb (optional)
USE_WANDB=${USE_WANDB:-0}                     # 1 to enable, 0 to skip
PROJECT_NAME="moshi-urdu"

# ====================================
echo "=== Fine-tune config ==="
echo "  GPUs:               $NUM_GPUS"
echo "  Per-device batch:   $PER_DEVICE_BATCH"
echo "  Grad accum:         $GRAD_ACCUM"
echo "  Effective batch:    $((NUM_GPUS * PER_DEVICE_BATCH * GRAD_ACCUM))"
echo "  Epochs:             $NUM_EPOCHS"
echo "  Max length:         $MAX_LENGTH frames ($((MAX_LENGTH * 80 / 1000))s)"
echo "  Save every:         $SAVE_STEPS steps"
echo "  Train data:         $TRAIN_DATA"
echo "  Init model:         $INIT_MODEL_DIR"
echo "  Output:             $OUTPUT_DIR"
echo "========================="
echo ""

cd /workspace/moshi-finetune

export NO_TORCH_COMPILE=1   # avoids torch.compile issues with deepspeed

REPORT_TO=""
if [ "$USE_WANDB" = "1" ]; then
    REPORT_TO="--report_to wandb --project_name $PROJECT_NAME"
fi

accelerate launch \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --deepspeed_config_file $DS_CONFIG \
    finetune.py \
        --launcher accelerate \
        --output_dir "$OUTPUT_DIR" \
        --train_data_files "$TRAIN_DATA" \
        --model_dir "$INIT_MODEL_DIR" \
        --model_dtype float32 \
        --model_user_stream \
        --max_length $MAX_LENGTH \
        --min_length 128 \
        --num_train_epochs $NUM_EPOCHS \
        --per_device_train_batch_size $PER_DEVICE_BATCH \
        --gradient_accumulation_steps $GRAD_ACCUM \
        --num_warmup_steps 100 \
        --activation_checkpointing \
        --logging_steps $LOGGING_STEPS \
        --save_steps $SAVE_STEPS \
        $REPORT_TO

echo ""
echo "=== Training complete ==="
echo "Checkpoints in: $OUTPUT_DIR"
echo "To convert latest ckpt to safetensors for inference:"
echo "  python -m tools.zero_to_fp32 \\"
echo "      $OUTPUT_DIR/step_<N> \\"
echo "      $OUTPUT_DIR/step_<N>_fp32 \\"
echo "      --moshi_lm_kwargs_path $INIT_MODEL_DIR/moshi_lm_kwargs.json"
