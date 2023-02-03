import pickle
from transformers import GPT2LMHeadModel, BertTokenizer, AdamW, WEIGHTS_NAME, CONFIG_NAME, GPT2Tokenizer
from backbones.GPT2_gen_prompt.data_utils import load_data, NEW_ADD_TOKENS
from backbones.GPT2_gen_prompt.dataset_gpt2 import GPT2Dataset, IGNORE_INDEX
import numpy as np
import json
from tqdm import tqdm

from utils.dataset import ID2GOAL, ID2TOPIC

import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--short_path', type=str, default=None)
    parser.add_argument('--long_path', type=str, default=None)
    parser.add_argument('--alpha', type=float, default=0.0)
    return parser.parse_args()


def read_file(file_path):
    with open(file_path,'rb') as f:
        data = pickle.load(f)
    return data

def get_action_topic_vocab(all_actions, all_topics, tokenizer):
    vocab = np.zeros(len(tokenizer))
    for action in all_actions:
        tokens = tokenizer.tokenize(action)
        token_ids = tokenizer.convert_tokens_to_ids(tokens)  
        vocab[token_ids] = 1.0
    
    for topic in all_topics:
        tokens = tokenizer.tokenize(topic)
        token_ids = tokenizer.convert_tokens_to_ids(tokens)  
        vocab[token_ids] = 1.0

    return vocab

def get_all_targets(file_path):
    with open(file_path, 'rb') as f:
        examples = pickle.load(f)
        all_targets = []
        for sample in examples:
            dic = {
                "action": sample["target"][0],
                "topic": sample["target"][1]
            }
            all_targets.append(dic)
        return all_targets

def test_join_planning(local_file, rl_file, alpha = 0.0):

    ### currently alpha = 0.6
    ### alpha = 0.4
    with open(local_file, 'rb') as f:
        local_data = pickle.load(f)
    
    with open(rl_file, 'rb') as f:
        rl_data = pickle.load(f)

    with open(f"preds/final_planning_{alpha}.txt", 'w') as f:
        for local_sample, rl_sample in tqdm(list(zip(local_data, rl_data))):

            local_prob = local_sample["prob"]
            rl_prob = rl_sample["join_class"]

            final_prob = local_prob * (1 - alpha) + alpha * rl_prob
            final_preb = final_prob.argmax(dim=-1)

            action = int(final_preb.item() / 646)
            topic = int(final_preb.item() % 646)

            action = ID2GOAL[action]
            topic = ID2TOPIC[topic]

            plan = {"action": action, "topic":topic}
            line = json.dumps(plan, ensure_ascii=False)
            f.write(line + "\n")
            f.flush()

if __name__=='__main__':

    args = parse_args()
    short_path = args.short_path
    long_path = args.long_path
    alpha = args.alpha 
    test_join_planning(short_path, long_path, alpha)