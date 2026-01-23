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
    --neg_table_name /home/user/jmlee/final_exp/phoenix_batch32_half_frame_cico/hard_neg_pkl/SAN/train/train_hard_word_777.pkl.gz \
    --output_dir /home/user/jmlee/out/ablation/test