import json
import pickle
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

SPECIAL_TOKENS_MAP = {"additional_special_tokens": [ACTION, TOPIC, BOP, EOP, BOS, EOS]}

def get_tokenizer(config_dir):
    tokenizer = BertTokenizer.from_pretrained(config_dir)
    num_added_tokens = tokenizer.add_special_tokens(SPECIAL_TOKENS_MAP)
    special_token_id_dict = {
        "pad_token_id": tokenizer.pad_token_id,
        "bos_token_id": tokenizer.convert_tokens_to_ids(BOS),
        "eos_token_id": tokenizer.convert_tokens_to_ids(EOS),
    }
    return tokenizer, num_added_tokens, special_token_id_dict

def read_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        return lines


def create_count_chart(data, ax = None, title = ''):
    if ax is not None:
        sns.countplot(data, ax = ax).set_title(title)
    sns.countplot(data).set_title(title)


def create_dist_chart(data, ax = None, title = ''):
    if ax is not None:
        sns.distplot(data, ax = ax).set_title(title)
    sns.distplot(data).set_title(title)


def create_bar_chart(data, ax = None, title = '', x_title = '', y_title = ''):
    if ax is not None:
        sns.barplot(data, ax = ax).set_title(title)

    sns.barplot(data).set_title(title)


def analyze_data(file_path):

    
    lines = read_file(file_path)
    all_numbers_of_turns = []
    all_numbers_of_goal_sequence_lengths = []
    all_numbers_of_goals = []
    all_numbers_of_topics = []
    all_goals = []
    all_topics = []


    for line in lines:

        line = json.loads(line)
        number_of_turn = get_number_of_turns(line)
        goal_sequence_length = get_goal_sequence_length(line)
        n_goals = get_number_of_goals(line)
        n_topics = get_number_of_topics(line)

        goals = get_goals(line)
        topics = get_topics(line)

        all_numbers_of_turns.append(number_of_turn)
        all_numbers_of_goal_sequence_lengths.append(goal_sequence_length)
        all_numbers_of_goals.append(n_goals)
        all_numbers_of_topics.append(n_topics)

        all_goals.extend(goals)
        all_topics.extend(topics)
    
    create_dist_chart(all_numbers_of_turns, ax = None, title = 'Conversation Length in DuRecDial 2.0')
    plt.show()
    create_count_chart(all_numbers_of_goal_sequence_lengths, ax = None, title = 'Goal Sequence Length in DuRecDial 2.0')
    plt.show()
    create_count_chart(all_numbers_of_goals, ax = None, title = 'Numbers of Goals')
    plt.show()
    create_count_chart(all_numbers_of_topics, ax = None, title = 'Numbers of Topics')
    plt.show()
    create_count_chart(all_goals, ax = None, title = 'Goals')
    plt.show()

    return all_numbers_of_turns, all_numbers_of_goal_sequence_lengths, all_numbers_of_goals, all_numbers_of_topics, all_goals, all_topics
 

def analyze_all_data(train_path, dev_path, test_path):

    keys = ['train', 'dev', 'test']
    paths = [train_path, dev_path, test_path]

    number_of_turns_dict = {}
    number_of_goal_sequence_lengths_dict = {}
    number_of_goals_dict = {}
    number_of_topics_dict = {}
    goal_dict = {}
    topic_dict = {}

    all_unique_goals = []
    all_unique_topics = []

    for key, path in list(zip(keys, paths)):

        all_numbers_of_turns, all_numbers_of_goal_sequence_lengths, all_numbers_of_goals, all_numbers_of_topics, all_goals, all_topics = analyze_data(path)

        number_of_turns_dict[key] = all_numbers_of_turns
        number_of_goal_sequence_lengths_dict[key] = all_numbers_of_goal_sequence_lengths
        number_of_goals_dict[key] = all_numbers_of_goals
        number_of_topics_dict[key] = number_of_topics_dict
        goal_dict[key] =  all_goals
        topic_dict[key] =  all_topics

        all_unique_goals.extend(all_goals)
        all_unique_topics.extend(all_topics)

    
    all_unique_goals = list(set(all_unique_goals))
    all_unique_topics = list(set(all_unique_topics))

    print('number of unique goals: ', len(all_unique_goals))
    print('number of unique topics: ', len(all_unique_topics))
    
    print(len(number_of_goals_dict))


def convert_text_data_to_json(text_data):
    return json.loads(text_data)


def create_groundtruth_examples(data):

    all_examples = []
    all_topics = []
    all_goals = []
    count = 0
    for i, sample in enumerate(data):
        sample = convert_text_data_to_json(sample)

        assert len(sample["conversation"]) == len(sample["goal_type_list"])
        assert len(sample["conversation"]) == len(sample["goal_topic_list"])
        assert len(sample["conversation"]) == len(sample["knowledge"])

        all_goals.extend(sample["goal_type_list"])
        all_topics.extend(sample['goal_topic_list'])

        ### notice that the user always starts the conversation.
        ### the last response is always user's one.
        ### data spliting
        user_start = True
        if sample['goal_type_list'][0] == "Greetings":
            user_start = False

        ## first step is to find the target topic and goal.
        ## only consider recommendation cases.
        curr = len(sample['goal_type_list']) - 1
        while ('recommendation' not in sample["goal_type_list"][curr] and 'Play' not in sample['goal_type_list'][curr]) and curr >= 0: 
            curr -= 1
        
        ### if the target action of the current example is recomemndation case.
        if curr >= 0 :
            target_topic = sample["goal_topic_list"][curr]
            target_goal = sample["goal_type_list"][curr]
            curr = 0
            if not user_start:     
                curr = 1
            turn = 0
            while curr < len(sample["conversation"]) - 1:
                example = {
                    'id': i,
                    'turn_id': turn,
                    'conversation': sample['conversation'][:curr + 1],
                    'response': sample['conversation'][curr + 1],
                    # 'knowledge': sample['knowledge'][:curr + 1],
                    'goals': sample['goal_type_list'][:curr + 1],
                    'topics': sample['goal_topic_list'][:curr + 1],
                    'action_path': sample['goal_type_list'][curr+1:],
                    'topic_path': sample['goal_topic_list'][curr+1:],
                    'user_profile': sample['user_profile'],
                    'target_topic': target_topic,
                    'target_goal': target_goal,
                    'target': [target_goal, target_topic],
                    'next_goal': sample['goal_type_list'][curr + 1],
                    'next_topic': sample['goal_topic_list'][curr + 1]
                }
                ### removing dupplicated elements
                knowledge = sample['knowledge']
                new_knowledge = []
                for x in knowledge:
                    if len(x) > 0 and x not in new_knowledge:
                        new_knowledge.append(x)
                example['knowledge'] = new_knowledge

                curr += 2
                turn += 1
                all_examples.append(example)

    all_goals = list(set(all_goals))
    all_topics = list(set(all_topics))

    return all_examples, all_goals, all_topics


def save_data(data, file_path):
    with open(file_path,'wb') as f:
        pickle.dump(data, f)

def save_data_txt(data, file_path):
    with open(file_path,'w') as f:
        for example in data:
            example = json.dumps(example)
            f.write(example + '\n')

def read_binary_file(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
        return data

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

def get_all_preds(file_path):
    with open(file_path, 'rb') as f:
        examples = pickle.load(f)
        all_targets = []
        for sample in examples:
            dic = {
                "action": sample["next_goal"],
                "topic": sample["next_topic"]
            }
            all_targets.append(dic)
        return all_targets