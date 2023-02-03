# -*- coding: utf-8 -*-
import os
import json
import pickle

SEP = "[SEP]"
PAD = "[PAD]"
NEW_ADD_TOKENS = [PAD, SEP]
IGNORE_INDEX = -100

def read_binary_file(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
        return data

all_goals = read_binary_file("data/all_goals.pkl")
all_topics = read_binary_file("data/all_topics.pkl")

ACTION2ID = {k:id for id, k in enumerate(all_goals)}
TOPIC2ID = {k:id for id, k in enumerate(all_topics)}

def extract_knowledge(kg_list, center_topic):
    """Extract knowledge according to the center topic"""
    sub_kg = []
    if center_topic == "NULL":
        for triple in kg_list:
            if len(triple) == 0:
                continue
            s, p, o = triple
            sub_kg.append(triple)
    else:
        for triple in kg_list:
            if len(triple) == 0:
                continue
            s, p, o = triple
            if s.lower() == center_topic.lower() or o.lower() == center_topic.lower():
                if not triple in sub_kg:
                    sub_kg.append(triple)
    return sub_kg

def convert_data(fp, extract_kg=False, tcp_path=None):
    cur_actions, cur_topics = [], []
    if tcp_path is not None:
        with open(tcp_path, 'r', encoding='utf-8') as fr:
            for line in fr:
                sample = json.loads(line)
                action = sample["action"]
                topic = sample["topic"]
                cur_actions.append(action)
                cur_topics.append(topic)
    data = []
    with open(fp, 'rb') as fr:
        for idx, sample in enumerate(pickle.load(fr)):
            # sample = json.loads(line)
            # original_goal = sample["original_goal"]
            user_profile = sample["user_profile"]
            history = sample["conversation"]
            resp_str = sample["response"]
            
            knowledge_str = ""
            context_str = ""
            input_str = ""
            action = None
            topic = None
            if extract_kg:
                if tcp_path is not None:
                    # extract knowledge according to generated plans
                    kg_list = extract_knowledge(sample["knowledge"], cur_topics[idx])
                    for triple in kg_list:
                        kd = " ".join(triple)
                        knowledge_str += f"[{kd}]"
                        # knowledge_str += " "
                        # knowledge_str += " | "
                    # input_str += cur_actions[idx] + cur_topics[idx] + SEP
                    action = cur_actions[idx]
                    topic = cur_topics[idx]
                else:
                    # extract knowledge according to current labeled topic
                    kg_list = extract_knowledge(sample["knowledge"], sample["next_topic"])
                    for triple in kg_list:
                        kd = " ".join(triple)
                        knowledge_str += kd
                        knowledge_str += " "
                        # knowledge_str += " | "
                    # input_str += sample["next_goal"] + sample["next_topic"] + SEP
                    action = sample["next_goal"]
                    topic =  sample["next_topic"]
            else:
                kg_list = sample["knowledge"]
                for triple in kg_list:
                    kd = " ".join(triple)
                    knowledge_str += kd
                    knowledge_str += " "

                # input_str += sample["target"][0] + sample["target"][1] + SEP
                action = sample["target"][0]
                topic =  sample["target"][1]
            
            ### append the user profile into the end of the input string
            ### removing the user profile
            # for k, v in user_profile.items():
            #     input_str += k
            #     temp = " "
            #     if isinstance(v, list):
            #         for t in v:
            #             temp += t
            #             temp += " "
            #     else:
            #         temp += v
            #     input_str += temp
            #     input_str += SEP
            
            # led_by_bot = False
            # if "Greetings" in sample["goals"][0]:
            #     led_by_bot = True
            for hdx, utt_str in enumerate(history):
                context_str += f"[{utt_str}]"

            input_str = f"{action} {SEP} {topic} {SEP} {knowledge_str} {SEP} {context_str}"
            data.append([input_str, resp_str])
    return data


def tokenize(tokenizer, obj):
    if isinstance(obj, str):
        return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
    if isinstance(obj, dict):
        return dict((n, tokenize(tokenizer, o)) for n, o in obj.items())
    return list(tokenize(tokenizer, o) for o in obj)


def load_data(tokenizer, logger, dataset_path, cache_dir, data_partition="train", use_tcp=False, tcp_path=None):
    """ Load data from cache or create from raw data."""
    if use_tcp:
        cache_dir = cache_dir + '_w_TCP'
    
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    cache_path = os.path.join(cache_dir, "{}.pkl".format(data_partition))
    
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            logger.info("Loading cached data from [{}]".format(cache_path))
            tokenized_data = pickle.load(f)
    else:          
        print("Creating data [{}]".format(cache_path))

        if use_tcp:
            if data_partition == "test":
                assert tcp_path is not None
                print("Loading from [{}] to prepare test data for TCP-enhanced generation.".format(tcp_path))
                # prepare test data for TCP-enhanced generation
                data = convert_data(fp=dataset_path, extract_kg=True, tcp_path=tcp_path)
            else:
                print("Prepare train/valid data for TCP-enhanced generation.")
                # prepare train/valid data for TCP-enhanced fine-tuning
                data = convert_data(fp=dataset_path, extract_kg=True)
        else:
            # prepare data for GPT2 fine-tuning
            data = convert_data(fp=dataset_path)  
        
        # tokenize data
        logger.info("Tokenizing ...")
        tokenized_data = tokenize(tokenizer, data)

        # caching data
        with open(cache_path, 'wb') as f:
            pickle.dump(tokenized_data, f)

        # for debugging
        #data_dict = {data_partition: data}
        #save_fp = os.path.join(cache_dir, "{}_cache.json".format(data_partition))
        #with open(save_fp, 'w', encoding='utf-8') as fw:
        #    json.dump(data_dict, fw, ensure_ascii=False, indent=0)
        
        print("Total of {} instances were cached.".format(len(data)))
    return tokenized_data


# if __name__=="__main__":
#     load_data(None, None, "../caches/path/train.pkl", "../caches/prefix_new", data_partition="train", use_tcp=True)
#     load_data(None, None, "../caches/path/valid.pkl", "../caches/prefix_new", data_partition="valid", use_tcp=True)
#     load_data(None, None, "../caches/path/test.pkl", "../caches/prefix_new", data_partition="test", use_tcp=True, tcp_path="../outputs/planning_wo_goal/best_model_test.txt")