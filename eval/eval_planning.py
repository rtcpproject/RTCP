import json
import pickle
import argparse

ACTION = "[A]"
TOPIC = "[T]"

count = 0

def calc_succ(eval_fp, gold_fp):
    all_eval, all_gold = [], []
    with open(eval_fp, 'r', encoding='utf-8') as fr:
        for line in fr:
            sample = json.loads(line)
            all_eval.append(sample)
    with open(gold_fp, 'r', encoding='utf-8') as fr:
        for line in fr:
            raw_sample = json.loads(line)
            sample = {
                "target": raw_sample["target"],
                "action_path": raw_sample["action_path"],
                "topic_path": raw_sample["topic_path"],
                "response": raw_sample["response"]
            }
            all_gold.append(sample)
    assert len(all_eval) == len(all_gold)

    topic_hit, topic_total = 0, 0
    movie_hit, music_hit, poi_hit, food_hit = 0, 0, 0, 0
    movie_total, music_total, poi_total, food_total = 0, 0, 0, 0
    
    for eval_sample, gold_sample in zip(all_eval, all_gold):
        if gold_sample["action_path"][0] == gold_sample["target"][0] and \
            gold_sample["topic_path"][0] == gold_sample["target"][1] and \
                gold_sample["target"][1].lower() in gold_sample["response"].lower():
            # eval this turn
            eval_action = gold_sample["target"][0]
            eval_topic = gold_sample["target"][1]

            topic_total += 1
            if eval_topic.lower() in eval_sample["response"].lower():
                topic_hit += 1
            
            if eval_action == "Movie recommendation":
                movie_total += 1
                if eval_topic.lower().replace(" ","") in eval_sample["response"].lower():
                    movie_hit += 1
            elif eval_action == "Music recommendation" or eval_action == "Play music":
                music_total += 1
                if eval_topic.lower().lower().replace(" ","") in eval_sample["response"].lower():
                    music_hit += 1
            elif eval_action == "POI recommendation":
                poi_total += 1
                if eval_topic.lower().lower().replace(" ","") in eval_sample["response"].lower():
                    poi_hit += 1
            elif eval_action == "Food recommendation":
                food_total += 1
                if eval_topic.lower().lower().replace(" ","") in eval_sample["response"].lower():
                    food_hit += 1
    succ_rate = float(topic_hit) / topic_total
    movie_rec_sr = float(movie_hit) / movie_total
    music_rec_sr = float(music_hit) / music_total
    poi_rec_sr = float(poi_hit) / poi_total
    food_rec_sr = float(food_hit) / food_total
    print("Succ.: {:.2f}%".format(succ_rate*100))
    print("SR - Movie: {}/{} = {:.2f}%".format(movie_hit, movie_total, movie_rec_sr*100))
    print("SR - Music: {}/{} = {:.2f}%".format(music_hit, music_total, music_rec_sr*100))
    print("SR - POI: {}/{} = {:.2f}%".format(poi_hit, poi_total, poi_rec_sr*100))
    print("SR - Food: {}/{} = {:.2f}%".format(food_hit, food_total, food_rec_sr*100))


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

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_file", type=str)
    parser.add_argument("--gold_file", type=str)
    args = parser.parse_args()

    targets, tar_actions, tar_topics, bi_actions, bi_topics = read_targets(args.gold_file)
    predicts, pred_actions, pred_topics = read_preditions(args.eval_file)

    join_acc =  compute_joint_acc(tar_actions, tar_topics, pred_actions, pred_topics)

    action_acc = compute_acc(tar_actions, pred_actions)
    topic_acc = compute_acc(tar_topics, pred_topics )

    action_bi_acc = compute_bi_acc(bi_actions, pred_actions)
    topic_bi_acc = compute_bi_acc(bi_topics, pred_topics)

    print(join_acc, action_acc, action_bi_acc, topic_acc, topic_bi_acc)

    print(count)