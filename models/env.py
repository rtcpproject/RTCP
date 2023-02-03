import json
import torch
import random
import os
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
from utils.data_utils import read_binary_file
from backbones.BERT.dataset_bert import BERTDataset

SEP = "[SEP]"
USER = "[USER]"  # additional special token
BOT = "[BOT]"    # additional special token
ACTION = "[A]"
TOPIC = "[T]"
TARGET = "[TARGET]"
PATH = "[PATH]"

NEW_ADD_TOKENS = ["[USER]", "[BOT]","[A]","[T]","[TARGET]", "[PATH]"]

all_goals = read_binary_file("data/all_goals.pkl")
all_topics = read_binary_file("data/all_topics.pkl")

GOAL2ID = {k:id for id, k in enumerate(all_goals)}
TOPIC2ID = {k:id for id, k in enumerate(all_topics)}

ID2GOAL = {id:k for id, k in enumerate(all_goals)}
ID2TOPIC = {id:k for id, k in enumerate(all_topics)}

class Env(object):
    def __init__(self, train_dataset, tokenizer, args, device, id2goal = None, id2topic = None):
        super().__init__()

        self.device = device
        self.train_dataset = train_dataset

        self.reward_model = BertForSequenceClassification.from_pretrained(args.reward_model_dir)
        self.tokenizer = tokenizer        

        self.id2goal = ID2GOAL
        self.id2topic = ID2TOPIC

        ### frozen the reward model
        self.reward_model.to(self.device)
        self.reward_model.eval()

        self.bert_dataset = BERTDataset(None, self.tokenizer)
        self.train_idx = -1
        self.context = []

    def reset(self, target_type=None):

        self.state = None
        self.sample = None
        self.context = []
        ### the reset function
        ### it randomly produce a state to the policy model
        ### we construct the previous planned paths using the training dataset
        ### Then we convert the paths into a list of token ids.
        self.train_idx = self.train_idx + 1
        if self.train_idx >= len(self.train_dataset) - 1:
            self.train_idx = 0

        ### get the current example
        sample = self.train_dataset[self.train_idx]
        ### define the input string
        input_str = ""

        ### get the current action, topic path.
        action_path = sample["goals"]
        topic_path = sample['topics']

        ### get the target action, topic
        target_action = sample['target_goal']
        target_topic = sample['target_topic']

        ### add path token to the beginning of the query
        input_str += PATH

        ### append the previous planned path into the end of the input string
        for action, topic in list(zip(action_path, topic_path)):
            input_str += ACTION + action + TOPIC + topic + SEP

        ### append the target goal/topic to the input string
        input_str += TARGET
        input_str += ACTION + target_action + TOPIC + target_topic + SEP

        # print(input_str)
        # print(self.tokenizer.tokenize(input_str))
        # assert 1==0

        ## convert tokens to token ids
        ### the state of the current step is the previous planned path.
        ### then we will use the path to predict the next action topic using the actor network.
        token_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(input_str))
        self.state = token_ids
        self.sample = sample
        self.context.append(token_ids)

        return token_ids

    def step(self, action):
        
        ### the step function
        ### it takes as input the current action and produce the next state and the reward to the policy model.
        ### we need to combine the state and the action to produce a new state to the policy model.

        ### extract the predicted goal and topic from the action.
        goal_idx = int(action.item() / len(self.id2topic))
        topic_idx = action.item() % len(self.id2topic)
        goal = self.id2goal[goal_idx] 
        topic = self.id2topic[topic_idx]

        ### contruct the input for the policy model.
        # print(ACTION, goal, TOPIC, topic)
        act = ACTION + goal + TOPIC + str(topic)
        
        act_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(act))
        instance = self.bert_dataset._process(self.state, act_ids, -1)
        
        # print(instance)
        # assert 1==0
        reward = 0.0
        ### compute the intermidiate reward
        with torch.no_grad():
            logits = self.reward_model(
                input_ids = torch.LongTensor(instance["input_ids"]).to(self.device).unsqueeze(0), 
                token_type_ids = torch.LongTensor(instance["seg_ids"]).to(self.device).unsqueeze(0), 
                position_ids = torch.LongTensor(instance["pos_ids"]).to(self.device).unsqueeze(0)
            )[0]
            reward = torch.sigmoid(logits[0, 1]).item()

        ### the predicted goal is the target goal, and the predicted topic is the target topic and current turn id is greater than 4
        ### then we give the model a high reward
        if goal == self.sample["target"][0] and topic == self.sample["target"][1] and self.sample["turn_id"] <= 4:
            reward += 3.0
        ## the predicted goal is the target goal and the current turn id greater than 4
        # elif goal == self.sample["target"][0] and self.sample["turn_id"] >= 4:
        #     reward += 2.0
        ## the predicted topic is the target topic and the current turn id greater than 4
        # elif topic == self.sample["target"][1] and self.sample["turn_id"] >= 4:
        #     reward += 2.0
        ### if the predicted goal is the target goal but the current turn id is smaller than 4
        # elif goal == self.sample["target"][0] and self.sample["turn_id"] < 4:
        #     reward += -1.0

        ### update the state
        #########################
        input_str = ""
        ### get the current action, topic path.
        # sample = self.train_dataset[self.train_idx]
        action_path = self.sample["goals"]
        topic_path = self.sample['topics']

        ### get the target action, topic
        target_action = self.sample['target_goal']
        target_topic = self.sample['target_topic']

        ### add path token to the beginning of the query
        input_str += PATH

        ### append the previous planned path into the end of the input string
        for g, t in list(zip(action_path, topic_path)):
            input_str += ACTION + g + TOPIC + t + SEP
        
        ### append the new action to the planned path
        input_str += ACTION + goal + TOPIC + str(topic) + SEP

        ### append the target goal/topic to the input string
        input_str += TARGET
        input_str += ACTION + target_action + TOPIC + target_topic + SEP

        # with open("log_128.txt", "a") as f:
        #     dic = {
        #         "pred_goal": goal,
        #         "pred_topic": str(topic),
        #         "reward": float(reward),
        #         "target_goal": self.sample["target"][0],
        #         "target_topic": self.sample["target"][1],
        #         "next_goal": self.sample["next_goal"],
        #         "next_topic": self.sample["next_topic"]
        #     }
        #     dic = json.dumps(dic)
        #     f.write(dic + "\n")

        ### update the state
        token_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(input_str))
        done = 0
        if self.sample["turn_id"] == 0:
            done = 1

        return token_ids, float(reward), done, 0

