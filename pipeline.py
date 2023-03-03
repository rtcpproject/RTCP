# -*- coding: utf-8 -*-
import argparse
import os
import sys
import logging
import json
import numpy as np
import random
from tqdm import tqdm
import pickle

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from models.policy import PolicyModel
from utils.dataset import DuRecDialDataset, ID2GOAL, ID2TOPIC, convert_input_to_features
from utils.data_utils import get_tokenizer
from utils.data_collator import PlanCollator
from utils.trainer import Trainer

from backbones.GPT2_gen_prompt.data_utils import convert_input_to_response_features
from backbones.GPT2_gen_prompt.data_utils import NEW_ADD_TOKENS
from backbones.GPT2_gen_prompt.dataset_gpt2 import GPT2Dataset
from backbones.GPT2_gen_prompt.prefix_tuning import PrefixTuningTemplate
from backbones.GPT2_gen_prompt.model import PromptGPT2

from transformers import BertModel, GPT2Tokenizer, GPT2LMHeadModel, GPT2Config

logging.basicConfig(
  level=logging.INFO,
  format="%(asctime)s [%(levelname)s] %(message)s",
  handlers=[
      logging.StreamHandler(sys.stdout)
  ]
)

def top_filtering(logits, top_k=0, top_p=0.0, threshold=-float('Inf'), filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
            threshold: a minimal threshold to keep logits
    """
    assert logits.dim() == 1  # Only work for batch size 1 for now - could update but it would obfuscate a bit the code
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Back to unsorted indices and set them to -infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value

    return logits


def sample_sequence(model, context, action_id, topic_id, tokenizer, args, contrained_vocab = None):
    special_tokens_ids = [tokenizer.pad_token_id, tokenizer.cls_token_id, tokenizer.sep_token_id]
    context = torch.tensor(context, dtype=torch.long, device=args.device).unsqueeze(0)
    generated = context
    n_ctx = model.plm.config.n_ctx
    output_ids = []
    action_tensor = torch.LongTensor([action_id]).unsqueeze(0).to(args.device)
    topic_tensor = torch.LongTensor([topic_id]).unsqueeze(0).to(args.device)

    for i in range(args.max_dec_len):
        input_ids = generated[0][-(n_ctx - 1):].unsqueeze(0)
        batch = {
            "input_ids": input_ids,
            "action_id": action_tensor,
            "topic_id": topic_tensor,
            "labels": None
        }
        lm_output = model(batch)
        #### we only consider token that belong to the contrained vocabulary.
        logits = lm_output["logits"]
        logits = logits[0, -1, :] / args.temperature

        if args.top_k > 0 or (args.top_p > 0 and args.top_p <= 1):
            filtered_logits = top_filtering(logits, top_k=args.top_k, top_p=args.top_p)
            probs = F.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            probs = F.softmax(logits, dim=-1)
            next_token = torch.topk(probs, 1)[1]
        
        if i < args.min_dec_len and next_token.item() in special_tokens_ids:
            while next_token.item() in special_tokens_ids:
                next_token = torch.multinomial(probs, num_samples=1)
        output_ids.append(next_token.item())
        generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)

        if next_token.item() in special_tokens_ids:
            break
    
    output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
    # output_text = output_text.replace("<|endoftext|>","")
    # output_text = output_text.replace(" ", "")
    return output_text

def str2bool(v):
    if v.lower() in ('true', 'yes', 't', 'y', '1'):
        return True
    elif v.lower() in ('false',' no', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError("Unsupported value encountered.")

def print_args(args):
    print("=============== Args ===============")
    for k in vars(args):
        print("%s: %s" % (k, vars(args)[k]))

def read_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        example = lines[0]
        print(example)
        example = json.loads(example)
        return example

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default="train", choices=["train", "test"])
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--bert_dir', type=str, default="bert-base-cased")
    parser.add_argument('--cache_dir', type=str, default="caches")
    parser.add_argument('--log_dir', type=str, default="logs/planning_wo_goal_1")

    #==================== planning ====================
    parser.add_argument('--infer_checkpoint', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default="outputs")
    parser.add_argument('--test_batch_size', type=int, default=1)

    #==================== generation ====================
    parser.add_argument('--model_dir', type=str, default="logs/prefix_gpt2_new_2_2_4")
    parser.add_argument('--checkpoint_name', type=str, default="checkpoint_mymodel_13.pth")
    parser.add_argument('--num_tokens', type=int, default=50)
    parser.add_argument('--n_action_toks', type=int, default=2)
    parser.add_argument('--n_topic_toks', type=int, default=2)
    parser.add_argument('--use_goal_topic', type=int, default=1)
    parser.add_argument('--freeze_plm', type=int, default=1)

    parser.add_argument("--top_k", type=int, default=0, help="Filter top-k tokens before sampling (<=0: no filtering)")
    parser.add_argument("--top_p", type=float, default=0.0, help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")
    parser.add_argument("--max_seq_len", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--max_dec_len", type=int, default=80, help="Maximum length of the output utterances")
    parser.add_argument("--min_dec_len", type=int, default=1, help="Minimum length of the output utterances")
    parser.add_argument("--temperature", type=float, default=1, help="Sampling softmax temperature")
    parser.add_argument("--contrain_vocab", type=int, default=0, help="contrain the output vocabulary")

    return parser.parse_args()


def predict_topic(model, tokenizer, target_item, target_action, dialog_history, knowledge, topic_path, action_path, profile = None, device = None, pad_token_id = None):
    features = convert_input_to_features(tokenizer, target_item, target_action, dialog_history, knowledge, topic_path, action_path, profile, is_test = True)
    test_dataset = [features]
    collator = PlanCollator(device=device, padding_idx = pad_token_id)
    test_loader = DataLoader(test_dataset, batch_size = 1, shuffle=False, collate_fn=collator.custom_collate)
    for batch in test_loader:
        output = model(batch)
        pred_action = output['goal_logits'].argmax(dim=-1).detach().cpu().numpy().tolist()
        pred_topic = output['topics_logits'].argmax(dim=-1).detach().cpu().numpy().tolist()
        goal = ID2GOAL[pred_action[0]]
        topic = ID2TOPIC[pred_topic[0]]
        return goal, topic

def generate_response(model, tokenizer, dialog_history, knowledge, action, topic, profile = None, device = None):

    input_str, response, action_id, topic_id  = convert_input_to_response_features(dialog_history, knowledge, topic = topic, action = action, profile = profile)
    tokenized_data = []
    input_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(input_str))
    response_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(response))
    tokenized_data.append([input_ids, response_ids, action_id, topic_id])

    test_dataset = GPT2Dataset(tokenized_data, tokenizer, max_seq_len=512, lm_labels=False)
    with torch.no_grad():
        for instance in tqdm(test_dataset, mininterval=1):
            # Only work for batch size 1 for now
            history = instance["input_ids"]
            action_id = instance["action_id"]
            topic_id = instance["topic_id"]
            ### contrain the vocabulary
            output_text = sample_sequence(model, history, action_id, topic_id, tokenizer, args)
            return output_text

def initilize(args):

    plan_tokenizer, num_added_tokens, token_id_dict = get_tokenizer(config_dir=args.bert_dir)
    vocab_size = len(plan_tokenizer)
    pad_token_id = token_id_dict["pad_token_id"]

    ### load the planning model
    if args.infer_checkpoint is not None:
        model_path = os.path.join(args.log_dir, args.infer_checkpoint)
    else:
        model_path = os.path.join(args.log_dir, "best_model.bin")

    plan_model = torch.load(model_path)
    logging.info("Planning Model loaded from [{}]".format(model_path))
    plan_model.to(args.device)
    plan_model.eval()

    ### load the generation model
    gen_model_dir = args.model_dir
    logging.info("Loading tokenizer for generation from [{}]".format(gen_model_dir))
    gen_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    plm = GPT2LMHeadModel.from_pretrained("gpt2")
    config = GPT2Config.from_pretrained("gpt2")

    special_tokens_dict = {'additional_special_tokens': NEW_ADD_TOKENS}
    num_added_toks = gen_tokenizer.add_special_tokens(special_tokens_dict)
    logging.info("We have added {} special tokens".format(num_added_toks))
    # # Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e., the length of the tokenizer.
    plm.resize_token_embeddings(len(gen_tokenizer))
    prefix_model = PrefixTuningTemplate(
        config=config,
        num_token = args.num_tokens,  
        n_action_toks=args.n_action_toks,
        n_topic_toks= args.n_topic_toks,
        use_goal_topic = args.use_goal_topic
    )
    gen_model = PromptGPT2(plm = plm, prefix_model = prefix_model, freeze_plm= args.freeze_plm)
    gen_model.load_state_dict(torch.load(os.path.join(args.model_dir, args.checkpoint_name)))
    gen_model.to(args.device)
    gen_model.eval()

    return plan_model, plan_tokenizer, gen_model, gen_tokenizer, pad_token_id

if __name__=="__main__":

    random.seed(1234)

    example_path = "example/example.txt"
    example = read_file(example_path)
    args = parse_config()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    args.device = device
    plan_model, plan_tokenizer, gen_model, gen_tokenizer, pad_token_id = initilize(args)

    knowledge = example['knowledge']
    dialog_history = ['Hello, Mr. Huy !']
    goals  = ["Greetings"]
    topics = ["Greetings"]

    print("Hello, Mr. X.")
    while True:
        user_res = input()
        dialog_history.append(user_res)
        goal, topic = predict_topic(
            plan_model, 
            plan_tokenizer, 
            target_item = example['target_topic'], 
            target_action = example['target_goal'], 
            dialog_history = dialog_history, 
            knowledge = example['knowledge'], 
            topic_path = topics, 
            action_path = goals, 
            profile = example['user_profile'], 
            device = device, 
            pad_token_id = pad_token_id
        )
        # goals.extend([goal])
        # topics.extend([topic])
        print(goal, topic)
        response = generate_response(
            model= gen_model, 
            tokenizer = gen_tokenizer, 
            dialog_history = example['conversation'], 
            knowledge = example['knowledge'], 
            action = goal, 
            topic = topic
        )
        dialog_history.append(response)
        print(response)
   