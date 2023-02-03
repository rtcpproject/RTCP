reward_cache_dir="caches/BERT_reward_path"
reward_model_dir="logs/BERT_reward_path"
test_path="caches/path/test.pkl"
out_dir="outputs/rl_after_version_4/"
checkpoint="logs_rl/version_0/checkpoints/best.ckpt"
mode="test"

export CUDA_VISIBLE_DEVICES=6

python main_rl_after.py --reward_model_dir ${reward_model_dir} \
                  --reward_cache_dir ${reward_cache_dir} \
                  --out_dir ${out_dir} \
                  --test_path ${test_path} \
                  --mode ${mode}
                  --checkpoint ${checkpoint}
