import argparse
from utils.data_utils import save_data_txt, read_file, create_groundtruth_examples, save_data
import os
import pickle

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default=None,
                        help="Path of the train dataset for dist dataset. ")
    parser.add_argument("--valid_path", type=str, default=None,
                        help="Path of the valid dataset for dist dataset. ")
    parser.add_argument("--test_path", type=str, default=None,
                        help="Path of the valid dataset for dist dataset. ")
    parser.add_argument("--cache_dir", type=str, default="dataset_cache",
                        help="Path or url of the dataset cache dir.")

    args = parser.parse_args()

    if not os.path.exists(args.cache_dir):
        
        os.mkdir(args.cache_dir)
        print('Read data and generate cached examples .......')

        #### generate the preprocessed data and save them for future usages.
        ### generate the training data.
        train_data =  read_file(args.train_path)
        train_examples, train_goals, train_topics  = create_groundtruth_examples(train_data)
        save_data(train_examples, args.cache_dir + '/train.pkl')
        save_data_txt(train_examples, args.cache_dir + '/train.txt')

        ### generate the valid data
        valid_data = read_file(args.valid_path)
        valid_examples, valid_goals, valid_topics =  create_groundtruth_examples(valid_data)
        save_data(valid_examples, args.cache_dir + '/valid.pkl')
        save_data_txt(valid_examples, args.cache_dir + '/valid.txt')

        ### generate the valid data
        test_data = read_file(args.test_path)
        test_examples, test_goals, test_topics =  create_groundtruth_examples(test_data)
        save_data(test_examples, args.cache_dir + '/test.pkl')
        save_data_txt(test_examples, args.cache_dir + '/test.txt')

        all_goals = list(set(train_goals + valid_goals + test_goals))
        all_topics = list(set(train_topics + valid_topics + test_topics))

        with open('data/all_goals.pkl', 'wb') as f:
            pickle.dump(all_goals, f)
        
        with open('data/all_topics.pkl', 'wb') as f:
            pickle.dump(all_topics, f)

if __name__ == '__main__':
    run()