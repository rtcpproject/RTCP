# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

IGNORE_INDEX = -100
PAD = "[PAD]"
CLS = "[CLS]"
SEP = "[SEP]"
ACTION = "[A]"
TOPIC = "[T]"

class BERTDataset(Dataset):
    def __init__(self, data, tokenizer, max_seq_len=512, batch_first=True, reward_labels=True):
        self.data = data
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.tokenizer.pad_token = PAD
        self.tokenizer.cls_token = CLS
        self.tokenizer.sep_token = SEP
        self.pad = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        self.bos = tokenizer.convert_tokens_to_ids(tokenizer.cls_token)
        self.eos = tokenizer.convert_tokens_to_ids(tokenizer.sep_token)
        self.batch_first = batch_first
        self.reward_labels = reward_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.reward_labels:
            history = self.data[index][0]
            next_act = self.data[index][1]
            label = self.data[index][2]
        else:
            history = self.data[index][0]
            next_act = self.data[index][1]
            label = IGNORE_INDEX
        return self._process(history, next_act, label)

    def _process(self, history, next_act, label):

        # truncate previous tokens if dialogue history is too long
        ## including the sep between the input and the next action, topic.

        if len(history) + len(next_act) > self.max_seq_len - 2:
            tmp = len(history) - (self.max_seq_len - 2 - (len(next_act)))
            history = history[tmp:]
        
        seg_ids = [0] * (len(history)) + [1] * len(next_act)
        input_ids = self.tokenizer.convert_tokens_to_ids([CLS]) + history + next_act + self.tokenizer.convert_tokens_to_ids([SEP])
        seg_ids = [1] + seg_ids + [1]
        pos_ids = list(range(len(input_ids)))
        
        assert len(input_ids) == len(seg_ids) == len(pos_ids)

        instance = {}
        instance["input_ids"] = input_ids
        instance["seg_ids"] = seg_ids
        instance["pos_ids"] = pos_ids
        instance["label"] = label

        return instance

    def collate(self, batch):
        input_ids = pad_sequence(
            [torch.tensor(instance["input_ids"], dtype=torch.long) for instance in batch],
            batch_first=self.batch_first, padding_value=self.pad)

        seg_ids = pad_sequence(
            [torch.tensor(instance["seg_ids"], dtype=torch.long) for instance in batch],
            batch_first=self.batch_first, padding_value=self.pad)

        pos_ids = pad_sequence(
            [torch.tensor(instance["pos_ids"], dtype=torch.long) for instance in batch],
            batch_first=self.batch_first, padding_value=self.pad)
        
        labels = torch.tensor([instance["label"] for instance in batch], dtype=torch.long) 
        
        return input_ids, seg_ids, pos_ids, labels

    def rl_collate(self, batch):

        input_ids = pad_sequence([torch.tensor(x, dtype=torch.long) for x in batch],  batch_first=self.batch_first, padding_value=self.pad)
        return input_ids