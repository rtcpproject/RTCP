import os
import logging
import pickle
import dataclasses
import json
from dataclasses import dataclass
from typing import List
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.data_utils import create_groundtruth_examples, read_binary_file, read_file

PAD = "[PAD]"
UNK = "[UNK]"
CLS = "[CLS]"
SEP = "[SEP]"

ACTION = "[A]"       # denote an action
TOPIC = "[T]"       # denote a topic
BOP = "[BOP]"     # begin of knowledge hop
EOP = "[EOP]"     # end of knowledge hop
BOS = "[BOS]"     # begin of sequence
EOS = "[EOS]"     # end of sequence

all_goals = read_binary_file("data/all_goals.pkl")
all_topics = read_binary_file("data/all_topics.pkl")

GOAL2ID = {k:id for id, k in enumerate(all_goals)}
TOPIC2ID = {k:id for id, k in enumerate(all_topics)}

ID2GOAL = {id:k for id, k in enumerate(all_goals)}
ID2TOPIC = {id:k for id, k in enumerate(all_topics)}

def convert_context_to_features(history, tokenizer, turn_type_size = 16, max_seq_len = 512, is_test = False):
    if len(history) > turn_type_size - 1:
        history = history[len(history) - (turn_type_size - 1):]
    cur_turn_type = len(history) % 2
    tokens, segs = [], []
    for h in history:
        h = tokenizer.tokenize(h)
        tokens = tokens + h + [SEP]
        segs = segs + len(h)*[cur_turn_type] + [cur_turn_type]
        cur_turn_type = cur_turn_type ^ 1
    ids = tokenizer.convert_tokens_to_ids(tokens)
    if len(ids) > max_seq_len - 2:
        ids = ids[2 - max_seq_len:]
        segs = segs[2 - max_seq_len:]
        ids = tokenizer.convert_tokens_to_ids([CLS]) + ids + tokenizer.convert_tokens_to_ids([SEP])
        segs = [1] + segs + [1]
    else:
        ids = tokenizer.convert_tokens_to_ids([CLS]) + ids
        segs = [1] + segs
    poss = list(range(len(ids)))
    assert len(ids) == len(poss) == len(segs)
    return ids, segs, poss

def convert_knowledge_to_features(knowledge, tokenizer, max_seq_len = 512, is_test = False):
    tokens, segs = [], []
    for h in knowledge:
        if len(h) == 0:
            continue
        ### source, relation and destination entities
        s,r,d = h
        ### concatenate theses entiteis into a string
        h = s + " " + r + " " + d
        ### tokenize the constructed string
        h = tokenizer.tokenize(h)
        ### create the tokens
        tokens = tokens + h + [SEP]
        ### create the segment ids
        segs = segs + len(h)*[0] + [0]
    ids = tokenizer.convert_tokens_to_ids(tokens)
    if len(ids) > max_seq_len - 2:
        ids = ids[2 - max_seq_len:]
        segs = segs[2 - max_seq_len:]
        ids = tokenizer.convert_tokens_to_ids([CLS]) + ids + tokenizer.convert_tokens_to_ids([SEP])
        segs = [1] + segs + [1]
    else:
        ids = tokenizer.convert_tokens_to_ids([CLS]) + ids
        segs = [1] + segs
    poss = list(range(len(ids)))
    assert len(ids) == len(poss) == len(segs)
    return ids, segs, poss

def convert_profile_to_features(user_profile, tokenizer, max_seq_len = 512, is_test = False):
    tokens, segs = [], []
    for k,v in user_profile.items():
        if isinstance(v, list):
            h = ""
            for a in v:
                h = k + " " + a
                ### tokenize the constructed string
                h = tokenizer.tokenize(h)
                ### create the tokens
                tokens = tokens + h + [SEP]
                ### create the segment ids
                segs = segs + len(h)*[0] + [0]
        else:
            h = k + " " + v
            h = tokenizer.tokenize(h)
            ### create the tokens
            tokens = tokens + h + [SEP]
            ### create the segment ids
            segs = segs + len(h)*[0] + [0]

    ids = tokenizer.convert_tokens_to_ids(tokens)
    if len(ids) > max_seq_len - 2:
        ids = ids[2 - max_seq_len:]
        segs = segs[2 - max_seq_len:]
        ids = tokenizer.convert_tokens_to_ids([CLS]) + ids + tokenizer.convert_tokens_to_ids([SEP])
        segs = [1] + segs + [1]
    else:
        ids = tokenizer.convert_tokens_to_ids([CLS]) + ids
        segs = [1] + segs
    poss = list(range(len(ids)))
    assert len(ids) == len(poss) == len(segs)
    return ids, segs, poss

def convert_path_to_features(action_path, topic_path, target_action, target_topic, tokenizer, max_seq_len = 512, is_test = False):
    
    tokens, segs = [], []
    
    assert len(action_path) == len(topic_path)
    
    #### tokenize the target action and target topic
    ### append the computed tokens to the beginning of the sequence.
    t_action_tokens = tokenizer.tokenize(target_action)
    t_topic_tokens = tokenizer.tokenize(target_topic)
    
    #### seg = [ACTION] + TARGET_ACTION + [TOPIC] + TARGET_TOPIC + [SEP]
    tokens += [ACTION] + t_action_tokens + [TOPIC] + t_topic_tokens + [SEP]
    segs += [0] + len(t_action_tokens) * [0] + [0] + len(t_topic_tokens) * [0] + [0]
    
    for action, topic in list(zip(action_path, topic_path)):
        
        ### tokenize the current action and topic
        h_action = tokenizer.tokenize(action)
        h_topic = tokenizer.tokenize(topic)
        
        tokens += [ACTION] + h_action + [TOPIC] + h_topic + [SEP]
        segs += [1] + len(h_action) * [1] + [1] + len(h_topic) * [1] + [1]
    
    ids = tokenizer.convert_tokens_to_ids(tokens)
    if len(ids) > max_seq_len - 2:
        ids = ids[2 - max_seq_len:]
        segs = segs[2 - max_seq_len:]
        ids = tokenizer.convert_tokens_to_ids([CLS]) + ids + tokenizer.convert_tokens_to_ids([SEP])
        segs = [1] + segs + [1]
    else:
        ids = tokenizer.convert_tokens_to_ids([CLS]) + ids
        segs = [1] + segs
    poss = list(range(len(ids)))
    
    assert len(ids) == len(poss) == len(segs)
    
    return ids, segs, poss

@dataclass(frozen=True)
class InputFeature:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """
    profile_ids: List[int]
    profile_segs: List[int]
    profile_poss: List[int]
    knowledge_ids: List[int]
    knowledge_segs: List[int]
    knowledge_poss: List[int]
    conversation_ids: List[int]
    conversation_segs: List[int]
    conversation_poss: List[int]
    next_topic: int
    next_goal: int
    
    path_ids: List[int]
    path_segs: List[int]
    path_poss: List[int]

    def to_json_string(self):
        """Serialize this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self)) + "\n"


class DuRecDialDataset(Dataset):
    """
    Self-defined Dataset class for the DuRecDial dataset.
    Args:
        Dataset ([type]): [description]
    """
    def __init__(self,
        data_path, 
        tokenizer, 
        data_partition, 
        cache_dir=None, 
        max_seq_len=512, 
        max_plan_len=512, 
        turn_type_size=16,
        use_knowledge_hop=False,
        is_test=False
    ):
        self.tokenizer = tokenizer
        self.data_partition = data_partition
        
        self.cache_dir = cache_dir
        self.max_seq_len = max_seq_len
        self.max_plan_len = max_plan_len
        self.turn_type_size = turn_type_size
        self.use_knowledge_hop = use_knowledge_hop
        self.is_test = is_test
        
        self.instances = []
        self._cache_instances(data_path)

    def _cache_instances(self, data_path):
        """
        Load data tensors into memory or create the dataset when it does not exist.
        """
        signature = "{}_cache.pkl".format(self.data_partition)
        if self.cache_dir is not None:
            if not os.path.exists(self.cache_dir):
                os.makedirs(self.cache_dir)
            cache_path = os.path.join(self.cache_dir, signature)
        else:
            cache_dir = os.mkdir("caches")
            cache_path = os.path.join(cache_dir, signature)
        
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                logging.info("Loading cached instances from {}".format(cache_path))
                self.instances = pickle.load(f)
        else:          
            logging.info("Loading raw data from {}".format(data_path))
            data = read_file(data_path)
            all_samples, _, _ = create_groundtruth_examples(data)

            logging.info("Creating cache instances {}".format(signature))
            for row in tqdm(all_samples):
                
                h_ids, h_segs, h_poss = convert_context_to_features(row["conversation"], self.tokenizer, self.turn_type_size, self.max_seq_len, self.is_test)
                k_ids, k_segs, k_poss = convert_knowledge_to_features(row['knowledge'], self.tokenizer, self.max_seq_len, self.is_test)
                p_ids, p_segs, p_poss = convert_profile_to_features(row['user_profile'], self.tokenizer, self.max_seq_len, self.is_test)
                t_ids, t_segs, t_poss = convert_path_to_features(row['goals'], row['topics'], row['target'][0], row['target'][1], self.tokenizer, self.max_seq_len, self.is_test)
                inputs = {
                    "conversation_ids": h_ids,
                    "conversation_segs": h_segs,
                    "conversation_poss": h_poss,
                    "knowledge_ids": k_ids,
                    "knowledge_segs": k_segs,
                    "knowledge_poss": k_poss,
                    "profile_ids": p_ids,
                    "profile_segs": p_segs,
                    "profile_poss": p_poss,
                    "next_topic": TOPIC2ID[row['next_topic']],
                    "next_goal": GOAL2ID[row['next_goal']],
                    "path_ids": t_ids,
                    "path_segs": t_segs,
                    "path_poss": t_poss,
                }
                feature = InputFeature(**inputs)
                self.instances.append(feature)            
            with open(cache_path, 'wb') as f:
                pickle.dump(self.instances, f)

        logging.info("Total of {} instances were cached.".format(len(self.instances)))

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index):
        return self.instances[index]



def convert_input_to_features(tokenizer, target_item, target_action, dialog_history, knowledge, topic_path, action_path, profile = None, turn_type_size = 16,  is_test = True, max_seq_len = 512):

    h_ids, h_segs, h_poss = convert_context_to_features(dialog_history, tokenizer, turn_type_size, max_seq_len, is_test)
    k_ids, k_segs, k_poss = convert_knowledge_to_features(knowledge, tokenizer, max_seq_len, is_test)
    p_ids, p_segs, p_poss = convert_profile_to_features(profile, tokenizer, max_seq_len, is_test)
    t_ids, t_segs, t_poss = convert_path_to_features(topic_path, action_path, target_action= target_action, target_topic = target_item, tokenizer = tokenizer, max_seq_len = max_seq_len, is_test = is_test)
    inputs = {
        "conversation_ids": h_ids,
        "conversation_segs": h_segs,
        "conversation_poss": h_poss,
        "knowledge_ids": k_ids,
        "knowledge_segs": k_segs,
        "knowledge_poss": k_poss,
        "profile_ids": p_ids,
        "profile_segs": p_segs,
        "profile_poss": p_poss,
        "next_topic": 0,
        "next_goal": 0,
        "path_ids": t_ids,
        "path_segs": t_segs,
        "path_poss": t_poss,
    }
    feature = InputFeature(**inputs)
    return feature