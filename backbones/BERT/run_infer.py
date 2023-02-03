# # Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import os
import json
import logging
import random
import numpy as np
import argparse
from pprint import pformat
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from transformers import BertTokenizer, BertForSequenceClassification
# from data_utils import load_data, NEW_ADD_TOKENS
from data_utils_full_info import load_data, NEW_ADD_TOKENS
from dataset_bert import BERTDataset
from sklearn.metrics import classification_report

from urllib3 import ProxyManager, make_headers
default_headers = make_headers(proxy_basic_auth='huydq44:Tsuchimikadoshin140598')
http = ProxyManager("http://10.16.29.21:8080/", headers=default_headers)

import wandb
wandb.login(key="655a3789b6d0fa02057ce048953fe5b00f1a84a9")
config = wandb.config

def str2bool(v):
    if v.lower() in ('true', 'yes', 't', 'y', '1'):
        return True
    elif v.lower() in ('false',' no', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError("Unsupported value encountered.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_tcp', type=str2bool, default="False", help="Whether or not use TCP-enhanced generation")
    parser.add_argument('--tcp_path', type=str, default=None, help="Path of the decoded plans by TCP.")
    parser.add_argument("--test_path", type=str, default=None, help="Path of the test dataset.")
    parser.add_argument("--cache_dir", type=str, default="dataset_cache",
                        help="Path or url of the dataset cache dir.")
    parser.add_argument("--model_dir", type=str, default="", help="Path, url or short name of the model")
    parser.add_argument("--output_dir", type=str, default="", help="Dir for storing generated output")
    parser.add_argument("--test_batch_size", type=int, default=1, help="Batch size for generation")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")
    parser.add_argument("--max_seq_len", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--max_dec_len", type=int, default=80, help="Maximum length of the output utterances")
    parser.add_argument("--min_dec_len", type=int, default=1, help="Minimum length of the output utterances")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed")
    parser.add_argument("--temperature", type=float, default=1, help="Sampling softmax temperature")
    parser.add_argument("--top_k", type=int, default=0, help="Filter top-k tokens before sampling (<=0: no filtering)")
    parser.add_argument("--top_p", type=float, default=0.0, help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__file__)
    logger.info(pformat(args))

    if args.model_dir is None or args.model_dir == "":
        logging.error("Checkpoint needed!")
        return

    random.seed(args.random_seed)
    torch.random.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)

    model_dir = args.model_dir
    
    logger.info("Loading tokenizer from [{}]".format(model_dir))
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    special_tokens_dict = {'additional_special_tokens': NEW_ADD_TOKENS}
    tokenizer.add_special_tokens(special_tokens_dict)
    
    logger.info("Loading model from [{}]".format(model_dir))
    model = BertForSequenceClassification.from_pretrained(model_dir)
    model.to(args.device)
    model.eval()

    wandb.init(project='Conversational_Planning', name = f"Reward_Test_BERT_{args.random_seed}")
    wandb.watch(model)

    # prepare data
    test_data = load_data(tokenizer, logger, args.test_path, args.cache_dir, 
        data_partition="test", use_tcp=False)
    
    logger.info("Evaluating...")
    eval_dataset = BERTDataset(test_data, tokenizer, max_seq_len=args.max_seq_len)
    eval_loader = DataLoader(eval_dataset, collate_fn=eval_dataset.collate, batch_size=16, shuffle=False)
    losses = []
    avg_acc = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch in tqdm(eval_loader):
            input_ids, seg_ids, pos_ids, labels = tuple(input_tensor.to(args.device) for input_tensor in batch)
            output = model(input_ids = input_ids, token_type_ids = seg_ids, position_ids = pos_ids, labels=labels)
            loss = output[0]
            pred = output[1].argmax(dim=-1)
            acc = torch.eq(pred, labels).sum()
            avg_acc += acc
            losses.append(float(loss))
            all_preds.extend(pred.detach().cpu().numpy().tolist())
            all_targets.extend(labels.detach().cpu().numpy().tolist())

    avg_loss = np.mean(losses)
    logger.info("Avg loss: {}".format(avg_loss))
    logger.info("Avg Acc: {}".format(avg_acc / len(eval_dataset)))

    print(classification_report(all_targets, all_preds))
    
    # set output dir
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_path = os.path.join(output_dir, "output_test.txt")
    
    logger.info("Generating...")
    # test_dataset = BERTDataset(test_data, tokenizer, max_seq_len=args.max_seq_len, lm_labels=False)
    
    with open(output_path, 'w', encoding="utf-8") as f:
        # generate responses
        with torch.no_grad():
            for pred, tar in tqdm(list(zip(all_preds, all_targets))):
                # Only work for batch size 1 for now
                line = f"prediction:{pred}, target:{tar}"
                f.write(line + "\n")
                f.flush()
    logger.info("Saved output to [{}]".format(output_path))


if __name__ == "__main__":
    main()
