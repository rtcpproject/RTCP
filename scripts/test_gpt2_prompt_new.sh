#!/bin/bash
use_tcp="true"
# tcp_path="None"
test_path="caches/path/test.pkl"
cache_dir="caches/prefix_gpt2_new"
max_length=80
num_tokens=50
n_action_toks=2
n_topic_toks=2

export CUDA_VISIBLE_DEVICES=2
seed = 1

model_dir="logs/prefix_gpt2_new_2_2_${seed}"
tcp_path="outputs/planning_wo_goal_1/best_model_test.txt"
# tcp_path="preds/final_planning.txt" ### if use_rl
checkpoint_name="checkpoint_mymodel_13.pth"
output_dir="outputs/prefix_gpt2_new_2_2_${seed}"
python backbones/GPT2_gen_prompt/run_infer.py --use_tcp ${use_tcp} \
        --tcp_path ${tcp_path} \
        --test_path ${test_path} \
        --cache_dir ${cache_dir} \
        --n_action_toks ${n_action_toks} \
        --n_topic_toks ${n_topic_toks} \
        --num_tokens ${num_tokens} \
        --model_dir ${model_dir} \
        --checkpoint_name ${checkpoint_name} \
        --output_dir ${output_dir} \
        --max_dec_len ${max_length} \
        --random_seed 1
