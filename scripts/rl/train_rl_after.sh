#!/bin/bash
reward_cache_dir="caches/BERT_reward_path"
reward_model_dir="logs/BERT_reward_path"
train_path="caches/path/train.pkl"
log_dir="logs/RL_full_info"
batch_size=16
random_seed=34
mode="train"
random_seed=45
lr_actor=5e-5
lr_critic=1e-4

export CUDA_VISIBLE_DEVICES=1

python main_rl_after.py --reward_model_dir ${reward_model_dir} \
                  --reward_cache_dir ${reward_cache_dir} \
                  --train_path ${train_path} \
                  --mode ${mode} \
                  --lr_actor ${lr_actor} \
                  --lr_critic ${lr_critic} \
                  --random_seed ${random_seed} \
                  --batch_size ${batch_size}

