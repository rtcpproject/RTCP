train_data="data/DuRecDial/data/en_train.txt"
dev_data="data/DuRecDial/data/en_dev.txt"
test_data="data/DuRecDial/data/en_test.txt"
cache_dir="caches/path/"

python data_preprocessing.py --train_path ${train_data} \
        --valid_path ${dev_data} \
        --test_path ${test_data} \
        --cache_dir ${cache_dir}