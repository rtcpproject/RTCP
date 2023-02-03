#!/bin/bash

# dataset
test_data="data/DuRecDial/data/en_test.txt"
bert_dir="bert-base-cased"
cache_dir="caches/planning"

use_ssd="false"
test_batch_size=6

export CUDA_VISIBLE_DEVICES=2

log_dir="logs/planning_wo_goal_1"
output_dir="outputs/planning_wo_goal_1"
python main_planning.py --mode test \
    --test_data ${test_data} \
    --bert_dir ${bert_dir} \
    --cache_dir ${cache_dir} \
    --log_dir ${log_dir} \
    --output_dir ${output_dir} \
    --test_batch_size ${test_batch_size}
