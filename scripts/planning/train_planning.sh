#!/bin/bash

# dataset
train_data="data/DuRecDial/data/en_train.txt"
dev_data="data/DuRecDial/data/en_dev.txt"
bert_dir="bert-base-cased"
cache_dir="caches/planning"

# train args
num_epochs=10
batch_size=6

export CUDA_VISIBLE_DEVICES=0
seed=1

log_dir="logs/planning_wo_goal_${seed}"
python main_planning.py --mode train \
    --train_data ${train_data} \
    --dev_data ${dev_data} \
    --bert_dir ${bert_dir} \
    --cache_dir ${cache_dir} \
    --log_dir ${log_dir} \
    --num_epochs ${num_epochs} \
    --batch_size ${batch_size} \
    --random_seed ${seed} \
    --validate_steps 2000 \

