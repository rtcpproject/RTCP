
train_path="caches/path/train.pkl"
valid_path="caches/path/valid.pkl"
cache_dir="caches/BERT_reward_path"

n_epochs=2
lr=5e-5
warmup_steps=3000
train_batch_size=16
valid_batch_size=16
random_seed=456

export CUDA_VISIBLE_DEVICES=1

log_dir="logs/BERT_reward_path"
python backbones/BERT/run_train.py --train_path ${train_path} \
        --valid_path ${valid_path} \
        --cache_dir ${cache_dir} \
        --log_dir ${log_dir} \
        --n_epochs ${n_epochs} \
        --lr ${lr} \
        --warmup_steps ${warmup_steps} \
        --train_batch_size ${train_batch_size} \
        --valid_batch_size ${valid_batch_size} \
        --random_seed 456