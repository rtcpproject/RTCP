# -*- coding: utf-8 -*-
import os
import json
import pickle
import copy
import random
import pickle

SEP = "[SEP]"
USER = "[USER]"  # additional special token
BOT = "[BOT]"    # additional special token
ACTION = "[A]"
TOPIC = "[T]"
TARGET = "[TARGET]"
PATH = "[PATH]"
# NEW_ADD_TOKENS = ["[USER]", "[BOT]","[A]","[T]"]
NEW_ADD_TOKENS = ["[USER]", "[BOT]","[A]","[T]","[TARGET]", "[PATH]"]


def read_file(file_path):
    with open(file_path, 'rb') as f:
        b = pickle.load(f)
        return b

ALL_ACTIONS = read_file('data/all_goals.pkl')
ALL_TOPICS = read_file('data/all_topics.pkl')

def uniform_sampling(item, list_item):
    tmp = copy.deepcopy(list_item)
    tmp.remove(item)
    neg_item = random.choice(tmp)
    return neg_item

def extract_knowledge(kg_list, center_topic):
    """Extract knowledge according to the center topic"""
    sub_kg = []
    if center_topic == "NULL":
        for triple in kg_list:
            if len(triple) > 0:
                s, p, o = triple
                sub_kg.append(triple)
    else:
        for triple in kg_list:
            if len(triple) > 0 :
                s, p, o = triple
                if s.lower() == center_topic.lower() or o.lower() == center_topic.lower():
                    if not triple in sub_kg:
                        sub_kg.append(triple)
    return sub_kg

def convert_data(fp, extract_kg=False, tcp_path=None):

    #### The data construction pipeline used in the GPT2 model.
    #### We keep the same pipeline for this BERT-based reward model.
    cur_actions, cur_topics = [], []
    data = []
    with open(fp, 'rb') as f:
        fr = pickle.load(f)
        for idx, sample in enumerate(fr):

            #### loop over the data corpus
            # sample = json.loads(line)
            
            ### the previous goal and topic paths
            ### knowledge base and the user profile
            action_path = sample["goals"]
            topic_path = sample['topics']
            user_profile = sample["user_profile"]
            history = sample["conversation"]

            ### the groundtruth response
            # resp_str = sample["response"]
            next_topic = sample['next_topic']
            next_action = sample['next_goal']
            ### the groundtruth response is contructed by using the next topic and action.
            ### [A] + action + [T] + topic
            resp_str = ACTION + next_action + TOPIC + next_topic

            ### create the input string
            input_str = ""
            ### if we use additional information from the knowledge base
            # if extract_kg:
            #     ### temporally ignore this branch
            #     if tcp_path is not None:
            #         # extract knowledge according to generated plans
            #         kg_list = extract_knowledge(sample["knowledge"], cur_topics[idx])
            #         for triple in kg_list:
            #             kd = "".join(triple)
            #             input_str += kd
            #             input_str += SEP
            #         input_str += cur_actions[idx] + cur_topics[idx] + SEP
            #     else:
            #         # extract knowledge according to current labeled topic
            #         kg_list = extract_knowledge(sample["knowledge"], sample["topics"][-1])
            #         ### append the knowledge information into the input string
            #         for triple in kg_list:
            #             kd = " ".join(triple)
            #             input_str += kd
            #             input_str += SEP
            # else:
            #     kg_list = sample["knowledge"]
            #     for triple in kg_list:
            #         kd = " ".join(triple)
            #         input_str += kd
            #         input_str += SEP
            
            # ### append the user profile into the end of the input string
            # # for k, v in user_profile.items():
            # #     for t in v:
            # #         input_str += k
            # #         input_str += t
            # #         input_str += SEP
            
            # led_by_bot = False
            # # if "Bot主动" in original_goal[0]:
            # #     led_by_bot = True
            # ### we assume that the user always start the conversation.
            # for hdx, utt_str in enumerate(history):
            #     if hdx % 2 == 0:
            #         if led_by_bot:
            #             input_str += BOT
            #         else:
            #             input_str += USER
            #     else:
            #         if led_by_bot:
            #             input_str += USER
            #         else:
            #             input_str += BOT
            #     input_str += utt_str
            # input_str += BOT
            # input_str += SEP

            target_action = sample['target_goal']
            target_topic = sample['target_topic']

            ### add path token to the beginning of the query
            input_str += PATH

            ### append the previous planned path into the end of the input string
            for action, topic in list(zip(action_path, topic_path)):
                input_str += ACTION + action + TOPIC + topic + SEP

            input_str += TARGET
            input_str += ACTION + target_action + TOPIC + target_topic + SEP
    
            ### Now we construct negative examples by corrupting the oberserved planned paths
            sampled_action = uniform_sampling(next_action, ALL_ACTIONS)
            sampled_topic = uniform_sampling(next_topic, ALL_TOPICS)

            # ### positive example
            # positive_example = input_str + SEP + resp_str

            # ### negative examples sampled by randomly replacing either the original topic and action with ramdom ones.
            # positive_example = ACTION + next_action + TOPIC + next_topic
            negative_example1 = ACTION + sampled_action + TOPIC + next_topic
            negative_example2 = ACTION + next_action + TOPIC + sampled_topic
            negative_example3 = ACTION + sampled_action + TOPIC + sampled_topic

            ### construct the data.
            data.append([input_str, resp_str, 1])
            data.append([input_str, negative_example1, 0])
            data.append([input_str, negative_example2, 0])
            data.append([input_str, negative_example3, 0])

            # data.append([positive_example, 1])
            # data.append([negative_example1, 0])
            # data.append([negative_example2, 0])
            # data.append([negative_example3, 0])

    return data


def tokenize(tokenizer, obj):
    if isinstance(obj, str):
        return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
    if isinstance(obj, dict):
        return dict((n, tokenize(tokenizer, o)) for n, o in obj.items())
    return list(tokenize(tokenizer, o) for o in obj)


def load_data(tokenizer, logger, dataset_path, cache_dir, data_partition="train", use_tcp = False):
    """ Load data from cache or create from raw data."""
    
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    cache_path = os.path.join(cache_dir, "{}_cache.pkl".format(data_partition))
    
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            logger.info("Loading cached data from [{}]".format(cache_path))
            tokenized_data = pickle.load(f)
    else:          
        logger.info("Creating cached data [{}]".format(cache_path))

        ### prepare data for GPT2 fine-tuning
        ### use information from the knowledge base
        data = convert_data(fp=dataset_path, extract_kg = True)  
        
        print(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(data[0][0])))
        print(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(data[0][1])))

        assert 1==0

        # tokenize data
        logger.info("Tokenizing ...")
        tokenized_data = tokenize(tokenizer, [x[:2] for x in data])
        for t, d in list(zip(tokenized_data,data)):
            t.append(d[-1])

        # caching data
        with open(cache_path, 'wb') as f:
            pickle.dump(tokenized_data, f)
        
        # for debugging
        #data_dict = {data_partition: data}
        #save_fp = os.path.join(cache_dir, "{}_cache.json".format(data_partition))
        #with open(save_fp, 'w', encoding='utf-8') as fw:
        #    json.dump(data_dict, fw, ensure_ascii=False, indent=0)
        
    logger.info("Total of {} instances were cached.".format(len(tokenized_data)))
    return tokenized_data
