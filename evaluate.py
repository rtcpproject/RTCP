import json
import pickle

ACTION = "[A]"
TOPIC = "[T]"

def split_action_topic(response):
    global count
    try:
        action, topic = response.split(TOPIC)
        action = action.replace(ACTION, "")
        action = action.strip()
        topic = topic.strip()
    except:
        action = ""
        topic = ""
        count += 1

    return action, topic

def read_targets(file_path):
    all_targets = []
    all_actions = []
    all_topics = []
    all_ids = []
    with open(file_path, 'rb') as f:
        lines = pickle.load(f)
        # lines = f.readlines()
        for line in lines:
            # line = json.loads(line.strip())

            all_ids.append(int(line['id']))

            target_action = line['next_goal']
            target_topic = line['next_topic']

            all_targets.append("")
            all_actions.append(target_action)
            all_topics.append(target_topic)

    bi_actions = []
    bi_topics = []
    prev_id = -1
    for idx, cur_id in enumerate(all_ids):
        if cur_id == prev_id:
            bi_acts = [all_actions[idx-1], all_actions[idx]]
            bi_tops = [all_topics[idx-1], all_topics[idx]]
        else:
            bi_acts = [all_actions[idx]]
            bi_tops = [all_topics[idx]]

        bi_actions.append(bi_acts)
        bi_topics.append(bi_tops)
        prev_id = cur_id

    return all_targets, all_actions, all_topics, bi_actions, bi_topics


def read_preditions(file_path):
    all_predictions = []
    all_actions = []
    all_topics = []
    with open(file_path,'r') as f:
        lines = f.readlines()
        for line in lines:
            line = json.loads(line)

            action = line['action']
            topic = line['topic']
            prediction = 'NULL'

            all_predictions.append(prediction)
            all_actions.append(action)
            all_topics.append(topic)

    return all_predictions, all_actions, all_topics

def compute_acc(targets, predicts):
    count = 0
    for (tar, pred) in list(zip(targets, predicts)):
        if tar.strip() == pred.strip():
            count += 1
    
    return count / len(targets)

def compute_bi_acc(bi_targers, bi_predicts):
    count = 0
    for (tar, pred) in list(zip(bi_targers, bi_predicts)):
        tar = [x.strip() for x in tar]
        if pred.strip() in tar:
            count += 1
    
    return count / len(targets)

def compute_joint_acc(target_actions, target_topics, predict_actions, predict_topics):
    count = 0
    for (tar_act, tar_top, pred_act, pred_top) in list(zip(target_actions, target_topics, predict_actions, predict_topics )):
        if tar_act.strip() == pred_act.strip() and tar_top.strip() == pred_top.strip():
            count += 1
    
    return count / len(targets)

def compute_acc_at_target_turn(gold_file, eval_file):

    ### read gold file
    with open(gold_file, 'rb') as f:
        gold_data = pickle.load(f)
    
    ### read eval file
    with open(eval_file, 'r') as f:
        eval_data = f.readlines()

    ### compute the accuracy at the target turn
    goal_acc = 0
    topic_acc = 0
    count = 0 
    assert len(gold_data) == len(eval_data)
    for gold_sample, eval_sample in list(zip(gold_data, eval_data)):
        if gold_sample["next_goal"] == gold_sample["target"][0] and gold_sample["next_topic"] == gold_sample["target"][1] and gold_sample["target"][1].lower() in gold_sample["response"].lower():
            eval_sample = json.loads(eval_sample)

            count += 1
            if eval_sample["action"] == gold_sample["next_goal"]:
                goal_acc += 1
            
            if eval_sample["topic"] == gold_sample["next_topic"]:
                topic_acc += 1
    
    print(count)
    print(goal_acc / count)
    print(topic_acc/ count)

if __name__=="__main__":

    test_target_path = "caches/path/test.pkl"
    test_pred_path = "outputs/planning_wo_goal_1/best_model_test.txt"

    targets, tar_actions, tar_topics, bi_actions, bi_topics = read_targets(test_target_path)
    predicts, pred_actions, pred_topics = read_preditions(test_pred_path)

    join_acc =  compute_joint_acc(tar_actions, tar_topics, pred_actions, pred_topics)

    action_acc = compute_acc(tar_actions, pred_actions)
    topic_acc = compute_acc(tar_topics, pred_topics )

    action_bi_acc = compute_bi_acc(bi_actions, pred_actions)
    topic_bi_acc = compute_bi_acc(bi_topics, pred_topics)

    print(join_acc, action_acc, action_bi_acc, topic_acc, topic_bi_acc)    
