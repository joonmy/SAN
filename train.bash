#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3 \
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port=1236 \
    --use_env train_vlp_v2.py \
    --batch-size 64 \
    --epochs 100 \
    --opt sgd \
    --lr 0.01 \
    --loss_lambda 0.4 \
    --num_hard 5 \
    --neg_table_name "path to negative table" \
    --output_dir "path to output"