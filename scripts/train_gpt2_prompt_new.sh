#!/bin/bash
use_tcp="true"
train_path="caches/path/train.pkl"
valid_path="caches/path/valid.pkl"
cache_dir="caches/prefix_gpt2_new"

model_checkpoint="gpt2"
n_epochs=5
lr=5e-5
warmup_steps=3000
train_batch_size=8
valid_batch_size=8
num_tokens=50
n_action_toks=2
n_topic_toks=2

export CUDA_VISIBLE_DEVICES=0
seed=1

log_dir="logs/prefix_gpt2_new_2_2_${seed}"
python backbones/GPT2_gen_prompt/run_train.py --use_tcp ${use_tcp} \
        --train_path ${train_path} \
        --valid_path ${valid_path} \
        --cache_dir ${cache_dir} \
        --n_action_toks ${n_action_toks} \
        --n_topic_toks ${n_topic_toks} \
        --num_tokens ${num_tokens} \
        --model_checkpoint ${model_checkpoint} \
        --log_dir ${log_dir} \
        --n_epochs ${n_epochs} \
        --lr ${lr} \
        --warmup_steps ${warmup_steps} \
        --train_batch_size ${train_batch_size} \
        --valid_batch_size ${valid_batch_size} \
        --random_seed ${seed}
