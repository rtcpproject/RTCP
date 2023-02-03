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
from transformers import BertTokenizer

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
TARGET = "[TARGET]"
PATH = "[PATH]"

all_goals = read_binary_file("data/all_goals.pkl")
all_topics = read_binary_file("data/all_topics.pkl")

GOAL2ID = {k:id for id, k in enumerate(all_goals)}
TOPIC2ID = {k:id for id, k in enumerate(all_topics)}

ID2GOAL = {id:k for id, k in enumerate(all_goals)}
ID2TOPIC = {id:k for id, k in enumerate(all_topics)}

SPECIAL_TOKENS_MAP = {"additional_special_tokens": [ACTION, TOPIC, BOP, EOP, BOS, EOS, TARGET, PATH]}

def get_tokenizer(config_dir):
    tokenizer = BertTokenizer.from_pretrained(config_dir)
    num_added_tokens = tokenizer.add_special_tokens(SPECIAL_TOKENS_MAP)
    special_token_id_dict = {
        "pad_token_id": tokenizer.pad_token_id,
        "bos_token_id": tokenizer.convert_tokens_to_ids(BOS),
        "eos_token_id": tokenizer.convert_tokens_to_ids(EOS),
    }
    return tokenizer, num_added_tokens, special_token_id_dict

def convert_path_to_features(action_path, topic_path, target_action, target_topic, tokenizer, max_seq_len = 512, is_test = False):
    
    assert len(action_path) == len(topic_path)

    ### update the state
    #########################
    input_str = ""
    ### get the current action, topic path.
    # sample = self.train_dataset[self.train_idx]
    action_path = action_path
    topic_path = topic_path

    ### get the target action, topic
    target_action = target_action
    target_topic = target_topic

    ### add path token to the beginning of the query
    input_str += PATH

    ### append the previous planned path into the end of the input string
    for g, t in list(zip(action_path, topic_path)):
        input_str += ACTION + g + TOPIC + t + SEP

    ### append the target goal/topic to the input string
    input_str += TARGET
    input_str += ACTION + target_action + TOPIC + target_topic + SEP

    #### tokenize the target action and target topic
    ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(input_str))    
    return ids

@dataclass(frozen=True)
class InputFeature:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """
    path_ids: List[int]
    label: int

    def to_json_string(self):
        """Serialize this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self)) + "\n"


class RLPretrainingDataset(Dataset):
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
                t_ids = convert_path_to_features(row['goals'], row['topics'], row['target'][0], row['target'][1], self.tokenizer, self.max_seq_len, self.is_test)
                inputs = {
                    "path_ids": t_ids,
                    "label":  (GOAL2ID[row['next_goal']] * len(ID2TOPIC)) + TOPIC2ID[row['next_topic']]
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
