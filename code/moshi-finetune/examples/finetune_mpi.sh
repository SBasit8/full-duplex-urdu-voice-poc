#!/bin/bash -x
#PJM -L rscgrp=cx-large
#PJM -L node=32
#PJM -L elapse=48:00:00
#PJM -S
#PJM -j
#PJM -o finetune_mpi.out

module load gcc/11.3.0 cuda/12.1.1
module load cudnn openmpi_cuda nccl

export CUDA_VISIBLE_DEVICES="0,1,2,3"
export NO_TORCH_COMPILE="1"

# train_data_files="processed_data/jchat/train-*.parquet"
train_data_files="path/to/proccessed_data/train-*.parquet"

mpirun -n 128 -machinefile $PJM_O_NODEINF -map-by ppr:2:socket \
    uv run finetune.py \
        --launcher mpi \
        --use_deepspeed \
        --deepspeed_config_file ds_configs/zero3-fp16-warmlr-act_ckpt.json \
        --output_dir output/moshiko-finetuned \
        --train_data_files "${train_data_files}" \
        --model_dir init_models/moshiko-both_streams-float32 \
        --model_dtype float32 \
        --model_user_stream \
        --max_length 2048 \
        --min_length 128 \
        --num_train_epochs 1 \
        --per_device_train_batch_size 4 \
        --gradient_accumulation_steps 1 \
        --num_warmup_steps 500 \
        --activation_checkpointing \
        --logging_steps 1 \
        --report_to wandb \
        --project_name moshi-finetuning \
        --save_steps 1000
