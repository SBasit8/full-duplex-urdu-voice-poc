export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
export NO_TORCH_COMPILE=1

# train_data_files="processed_data/jchat/train-*.parquet"
train_data_files="path/to/proccessed_data/train-*.parquet"

uv run accelerate launch \
    --num_processes 8 \
    --num_machines 1 \
    --use_deepspeed \
    --deepspeed_config_file ds_configs/zero3-fp16-warmlr-act_ckpt.json \
    finetune.py \
        --launcher accelerate \
        --output_dir output/moshiko-finetuned \
        --train_data_files "${train_data_files}" \
        --model_dir init_models/moshiko-both_streams-float32 \
        --model_dtype float32 \
        --model_user_stream \
        --max_length 2048 \
        --min_length 128 \
        --num_train_epochs 1 \
        --per_device_train_batch_size 4 \
        --gradient_accumulation_steps 16 \
        --num_warmup_steps 500 \
        --activation_checkpointing \
        --logging_steps 1 \
        --report_to wandb \
        --project_name moshi-finetuning \
        --save_steps 1000
