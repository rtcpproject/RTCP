import os
from tqdm import tqdm
import numpy as np
import argparse
import logging
from pprint import pformat
import json
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint, progress
import pytorch_lightning as pl
from models.actor_critic import A2CPolicyNetwork
from utils.data_utils import read_binary_file
from tqdm import tqdm
import pickle

SEP = "[SEP]"
USER = "[USER]"  # additional special token
BOT = "[BOT]"    # additional special token
ACTION = "[A]"
TOPIC = "[T]"
TARGET = "[TARGET]"
PATH = "[PATH]"

all_goals = read_binary_file("data/all_goals.pkl")
all_topics = read_binary_file("data/all_topics.pkl")

GOAL2ID = {k:id for id, k in enumerate(all_goals)}
TOPIC2ID = {k:id for id, k in enumerate(all_topics)}

ID2GOAL = {id:k for id, k in enumerate(all_goals)}
ID2TOPIC = {id:k for id, k in enumerate(all_topics)}

def parse_arguments():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--n_goals", type = int, default = 19)
    parser.add_argument("--n_topics", type = int, default = 646)
    parser.add_argument("--use_gpu", type = int, default = 1)
    parser.add_argument("--reward_model_dir", type = str, default="")
    parser.add_argument("--reward_cache_dir", type = str, default="")
    parser.add_argument("--checkpoint", type = str, default="logs_rl/version_0/checkpoints/best.ckpt")
    parser.add_argument("--train_path", type = str, default="", required=False)
    parser.add_argument("--test_path", type = str, default="", required=False)
    parser.add_argument("--out_dir", type = str, default="", required=False)
    parser.add_argument("--mode", type = str, default="train")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--lam", type=float, default=0.95, help="advantage discount factor")
    parser.add_argument("--lr_actor", type=float, default=1e-4, help="learning rate of actor network")
    parser.add_argument("--lr_critic", type=float, default=5e-5, help="learning rate of critic network")
    parser.add_argument("--max_episode_len", type=int, default=15, help="capacity of the replay buffer")
    parser.add_argument("--batch_size", type=int, default=256, help="batch_size when training network")
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=256,
        help="how many action-state pairs to rollout for trajectory collection per epoch",
    )
    parser.add_argument(
        "--nb_optim_iters", type=int, default=4, help="how many steps of gradient descent to perform on each batch"
    )
    parser.add_argument(
        "--clip_ratio", type=float, default=0.2, help="hyperparameter for clipping in the policy objective"
    )
    parser.add_argument("--random_seed", type=int, default=67, help="random seed")

    return parser.parse_args()

def test(args):
    # train_dataset = read_binary_file(args.train_path)
    model = A2CPolicyNetwork.load_from_checkpoint(args.checkpoint, args = args, train_dataset = None)
    test_dataset = read_binary_file(args.test_path)
    
    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)
    data = []
    with open(args.out_dir + 'best_model_test.txt', 'w', encoding='utf-8') as f:
        for i in tqdm(range(len(test_dataset))):
            sample = test_dataset[i]

            ### update the state
            #########################
            input_str = ""
            ### get the current action, topic path.
            # sample = self.train_dataset[self.train_idx]
            action_path = sample["goals"]
            topic_path = sample['topics']

            ### get the target action, topic
            target_action = sample['target_goal']
            target_topic = sample['target_topic']

            ### add path token to the beginning of the query
            input_str += PATH

            ### append the previous planned path into the end of the input string
            for g, t in list(zip(action_path, topic_path)):
                input_str += ACTION + g + TOPIC + t + SEP
            
            ### append the target goal/topic to the input string
            input_str += TARGET
            input_str += ACTION + target_action + TOPIC + target_topic + SEP

            token_ids = model.tokenizer.convert_tokens_to_ids(model.tokenizer.tokenize(input_str))
            real_state = torch.LongTensor(token_ids).cuda().unsqueeze(0)
            ### compute the hidden state
            real_state = model.encoder(real_state)[0]
            ### get the cls token embedding
            real_state = real_state[:, 0, :]
            ### compute the action and value
            distribution, logits = model.actor(real_state)
            pred = logits.argmax(dim=-1)
            pred_prob = torch.softmax(logits, dim =-1)

            goal = int(pred.item() / 646)
            topic = pred.item() % 646

            goal = ID2GOAL[goal]
            topic = ID2TOPIC[topic]

            plan = {"action": goal, "topic": topic, "join_class":pred_prob.detach().cpu(), "prob":pred_prob.max(dim=-1)[0].item()}
            plan2 = {"action": goal, "topic": topic}
            data.append(plan)
            line = json.dumps(plan2, ensure_ascii=False)
            f.write(line + "\n")
            f.flush()

    with open("preds/rl_pred.pkl", 'wb') as f:
        pickle.dump(data, f)

if __name__ == "__main__":
    args = parse_arguments()
    pl.seed_everything(args.random_seed)
    if args.mode == "train":
        tb_logger = pl.loggers.TensorBoardLogger('logs_rl/', name='')
        checkpoint_callback = ModelCheckpoint(
            save_weights_only=True,
            save_last=True,
            verbose=True,
            filename='best',
            monitor='avg_ep_reward',
            mode='max'
        )
        bar_callback = progress.TQDMProgressBar(refresh_rate=50)
        logger = logging.getLogger(__file__)
        train_dataset = read_binary_file(args.train_path)
        model = A2CPolicyNetwork(args=args, train_dataset=train_dataset)
        args.gpus = [0]
        args.logger = tb_logger
        args.detect_anomaly = True
        args.gradient_clip_val = 0.5
        args.callbacks = [checkpoint_callback, bar_callback]
        trainer = pl.Trainer.from_argparse_args(args)
        trainer.fit(model)

    elif args.mode == "test":
        test(args)
