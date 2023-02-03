#!/bin/bash

# dataset
train_data="data/DuRecDial/data/en_train.txt"
dev_data="data/DuRecDial/data/en_dev.txt"
bert_dir="bert-base-cased"
cache_dir="caches/rl_pretrain"

# train args
num_epochs=10
batch_size=4

export CUDA_VISIBLE_DEVICES=5

log_dir="logs/rl_pretrain"
python pretraining.py --mode train \
    --train_data ${train_data} \
    --dev_data ${dev_data} \
    --bert_dir ${bert_dir} \
    --cache_dir ${cache_dir} \
    --log_dir ${log_dir} \
    --num_epochs ${num_epochs} \
    --batch_size ${batch_size} \
    --random_seed 42 \
    --validate_steps 2000 \

