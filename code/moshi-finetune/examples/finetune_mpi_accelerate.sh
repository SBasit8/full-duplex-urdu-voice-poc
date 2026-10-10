#!/bin/bash -x
#PJM -L rscgrp=cx-large
#PJM -L node=32
#PJM -L elapse=48:00:00
#PJM -S
#PJM -j
#PJM -o finetune_mpi_accelerate.out

module load gcc/11.3.0 cuda/12.1.1
module load cudnn openmpi_cuda nccl

export NO_TORCH_COMPILE="1"

export NNODES="32"
export NPROC_PER_NODE="4"
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export MASTER_ADDR=$(head -n 1 $PJM_O_NODEINF)
export MASTER_PORT="29500"

launch_cmd() {
    # train_data_files="processed_data/jchat/train-*.parquet"
    train_data_files="path/to/proccessed_data/train-*.parquet"

    uv run accelerate launch \
        --num_processes $(($NNODES * $NPROC_PER_NODE)) \
        --num_machines $NNODES \
        --machine_rank $OMPI_COMM_WORLD_RANK \
        --main_process_ip $MASTER_ADDR \
        --main_process_port $MASTER_PORT \
        --num_cpu_threads_per_process 12 \
        --use_deepspeed \
        --deepspeed_config_file ds_configs/zero3-fp16-warmlr-act_ckpt.json \
        --deepspeed_multinode_launcher standard \
        finetune.py \
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
}
export -f launch_cmd

mpirun \
    -np $NNODES \
    -machinefile $PJM_O_NODEINF \
    -map-by ppr:1:node \
    -bind-to none \
    bash -c launch_cmd
