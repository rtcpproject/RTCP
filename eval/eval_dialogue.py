#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import json
import numpy as np
from collections import Counter
from nltk.translate import bleu_score
from nltk.translate.bleu_score import SmoothingFunction
import pickle
import unicodedata
import re
from collections import Counter

re_art = re.compile(r'\b(a|an|the)\b')
re_punc = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re_art.sub(' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        return re_punc.sub(' ', text)  # convert punctuation to spaces

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def _prec_recall_f1_score(pred_items, gold_items):
    """
    Compute precision, recall and f1 given a set of gold and prediction items.
    :param pred_items: iterable of predicted values
    :param gold_items: iterable of gold values
    :return: tuple (p, r, f1) for precision, recall, f1
    """

    common = Counter(gold_items) & Counter(pred_items)
    num_same = sum(common.values())
    if num_same == 0:
        return 0, 0, 0
    precision = 1.0 * num_same / len(pred_items)
    recall = 1.0 * num_same / len(gold_items)
    f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1


def calc_f1(guess, answers):

    f1s = []
    for pred, tar in list(zip(guess, answers)):
        """Return the max F1 score between the guess and *any* answer."""
        g_tokens = normalize_answer(pred).split()
        p,r,f1 = _prec_recall_f1_score(g_tokens, normalize_answer(tar).split())
        f1s.append(f1)
    return sum(f1s) / len(f1s)

def calc_avg_turns(eval_fp, gold_fp):
    all_eval, all_gold = [], []
    with open(eval_fp, 'r', encoding='utf-8') as fr:
        for line in fr:
            sample = json.loads(line)
            all_eval.append(sample)
    with open(gold_fp, 'rb') as fr:
        for raw_sample in pickle.load(fr):
            # raw_sample = json.loads(line)
            sample = {
                "target": raw_sample["target"],
                "next_goal": raw_sample["next_goal"],
                "next_topic": raw_sample["next_topic"],
                "action_path": raw_sample["goals"],
                "topic_path": raw_sample["topics"],
                "response": raw_sample["response"],
                "turn_id": raw_sample["turn_id"],
                "conversation": raw_sample["conversation"]
            }
            all_gold.append(sample)
    assert len(all_eval) == len(all_gold)
    conver_maps = []
    for sample in all_gold:
        conver_maps.append(sample["turn_id"])

    avg_turns = []
    succ = 0
    count = 0
    sr5 = 0
    sr10 = 0
    sr15 = 0
    num_conversations = 0
    for idx, t_id in enumerate(conver_maps):
        ### starting point of a conversation:
        if t_id == 0:
            p_idx = idx + 1
            check = False
            n_turn = 0
            num_conversations += 1
            succ_turn = None
            ### we are still in the current conversation
            while p_idx < len(conver_maps) and conver_maps[p_idx] != 0:
                n_turn += 1
                eval_sample = all_eval[p_idx]
                gold_sample = all_gold[p_idx]
                ## if the current turn is the target turn (topic=target_topic and action = target_action)
                ## if the topic is in the gold response as well as in the predicted response
                ## then we consider the current turn a success turn
                ## but we only consider the first success turn for each conversation
                if check:
                    p_idx +=1
                    continue
                ### if the targeted topic appear in the generated response.
                if gold_sample["target"][1].lower() in eval_sample["response"].lower() and not check:
                    succ_turn = n_turn
                    if succ_turn <= 1:
                        sr5 += 1
                    if succ_turn <= 3:
                        sr10 += 1 
                    succ += 1
                    # avg_turns.append(conver_maps[p_idx])
                    count += 1
                    check = True
                            # break
                p_idx +=1

            ### if the system's fail to recommend the target item
            if succ_turn is None:
                avg_turns.append(n_turn)
            else:
                assert succ_turn <= n_turn
                avg_turns.append(succ_turn)

        else:
            ### nothing to do
            pass

    print(len(avg_turns))
    print(num_conversations)
    assert len(avg_turns) == num_conversations

    print(sum(avg_turns)/ num_conversations)
    print("success rate: ", succ / num_conversations * 100)
    print("sr@1: ", sr5 / num_conversations * 100)
    print("sr@3: ", sr10 / num_conversations * 100)


    


def calc_succ(eval_fp, gold_fp):
    all_eval, all_gold = [], []
    with open(eval_fp, 'r', encoding='utf-8') as fr:
        for line in fr:
            sample = json.loads(line)
            all_eval.append(sample)
    with open(gold_fp, 'rb') as fr:
        for raw_sample in pickle.load(fr):
            # raw_sample = json.loads(line)
            sample = {
                "conversation": raw_sample["conversation"],
                "target": raw_sample["target"],
                "next_action": raw_sample["next_goal"],
                "next_topic": raw_sample["next_topic"],
                "action_path": raw_sample["goals"],
                "topic_path": raw_sample["topics"],
                "response": raw_sample["response"]
            }
            all_gold.append(sample)
    assert len(all_eval) == len(all_gold)

    topic_hit, topic_total = 0, 0
    movie_hit, music_hit, poi_hit, food_hit = 0, 0, 0, 0
    movie_total, music_total, poi_total, food_total = 0, 0, 0, 0

    hit5, hit10, hit15 = 0, 0, 0
    
    count = 0
    for eval_sample, gold_sample in zip(all_eval, all_gold):
        count +=1
        #### 
        if gold_sample["next_action"] == gold_sample["target"][0] and \
            gold_sample["next_topic"] == gold_sample["target"][1] and \
                gold_sample["target"][1].lower() in gold_sample["response"].lower():
            # eval this turn
            eval_action = gold_sample["target"][0]
            eval_topic = gold_sample["target"][1]

            temp = unicodedata.normalize("NFKD", eval_sample["response"])

            topic_total += 1
            if eval_topic.lower().replace(" ","") in temp.lower().replace(" ",""):
                topic_hit += 1

                if len(gold_sample["conversation"]) < 5:
                    hit5 +=1
                if len(gold_sample["conversation"]) < 10:
                    hit10 +=1
                if len(gold_sample["conversation"]) < 15:
                    hit15 +=1
            
            if eval_action == "Movie recommendation":
                movie_total += 1
                if eval_topic.lower() in eval_sample["response"].lower():
                    movie_hit += 1
            elif eval_action == "Music recommendation" or eval_action == "Play music":
                music_total += 1
                if eval_topic.lower() in eval_sample["response"].lower():
                    music_hit += 1
            elif eval_action == "POI recommendation":
                poi_total += 1
                if eval_topic.lower() in eval_sample["response"].lower():
                    poi_hit += 1
            elif eval_action == "Food recommendation":
                food_total += 1
                if eval_topic.lower() in eval_sample["response"].lower():
                    food_hit += 1
    succ_rate = float(topic_hit) / topic_total
    movie_rec_sr = float(movie_hit) / movie_total
    music_rec_sr = float(music_hit) / music_total
    poi_rec_sr = float(poi_hit) / poi_total
    food_rec_sr = float(food_hit) / food_total
    print("Succ.: {:.2f}%, topic hit: {:.2f}, topic total: {:.2f}".format(succ_rate*100, topic_hit, topic_total))
    print("SR - Movie: {}/{} = {:.2f}%".format(movie_hit, movie_total, movie_rec_sr*100))
    print("SR - Music: {}/{} = {:.2f}%".format(music_hit, music_total, music_rec_sr*100))
    print("SR - POI: {}/{} = {:.2f}%".format(poi_hit, poi_total, poi_rec_sr*100))
    print("SR - Food: {}/{} = {:.2f}%".format(food_hit, food_total, food_rec_sr*100))

    print("SR@5: {:.2f}%".format(hit5/ topic_total * 100))
    print("SR@10: {:.2f}%".format(hit10/ topic_total * 100))
    print("SR@15: {:.2f}%".format(hit15/ topic_total * 100))


# def calc_f1(hyps, refs):
#     """ Calculate char-level f1 score """
#     golden_char_total = 0.0
#     pred_char_total = 0.0
#     hit_char_total = 0.0
#     for response, golden_response in zip(hyps, refs):
#         golden_response = "".join(golden_response)
#         response = "".join(response)
#         common = Counter(response) & Counter(golden_response)
#         hit_char_total += sum(common.values())
#         golden_char_total += len(golden_response)
#         pred_char_total += len(response)
#     p = hit_char_total / pred_char_total if pred_char_total > 0 else 0
#     r = hit_char_total / golden_char_total if golden_char_total > 0 else 0
#     f1 = 2 * p * r / (p + r) if p + r > 0 else 0
#     return f1

# def calc_f1(preds, refs):
#     ### word-level f1 score
#     f1s = []
#     for pred_items, gold_items in zip(preds, refs):
#         pred_items = "".join(pred_items).split()
#         gold_items = "".join(gold_items).split()
#         common = Counter(gold_items) & Counter(pred_items)
#         num_same = sum(common.values())
#         if num_same == 0:
#             f1 = 0
#         else:
#             precision = 1.0 * num_same / len(pred_items)
#             recall = 1.0 * num_same / len(gold_items)
#             f1 = (2 * precision * recall) / (precision + recall)
#         f1s.append(f1)
#     return sum(f1s)/len(f1s)


def calc_bleu(hyps, refs):
    """ Calculate bleu 1/2 """
    bleu_1 = []
    bleu_2 = []
    for hyp, ref in zip(hyps, refs):
        try:
            score = bleu_score.sentence_bleu(
                [ref], hyp,
                smoothing_function=SmoothingFunction().method1,
                weights=[1, 0, 0, 0])
        except:
            score = 0
        bleu_1.append(score)
        try:
            score = bleu_score.sentence_bleu(
                [ref], hyp,
                smoothing_function=SmoothingFunction().method1,
                weights=[0, 1.0, 0, 0])
        except:
            score = 0
        bleu_2.append(score)
    bleu_1 = np.average(bleu_1)
    bleu_2 = np.average(bleu_2)
    return bleu_1, bleu_2


def calc_distinct(seqs):
    """ Calculate intra/inter distinct 1/2 """
    seqs = [seq.split(' ') for seq in seqs]
    intra_dist1, intra_dist2 = [], []
    unigrams_all, bigrams_all = Counter(), Counter()
    for seq in seqs:
        unigrams = Counter(seq)
        bigrams = Counter(zip(seq, seq[1:]))
        intra_dist1.append((len(unigrams)+1e-12) / (len(seq)+1e-5))
        intra_dist2.append((len(bigrams)+1e-12) / (max(0, len(seq)-1)+1e-5))

        unigrams_all.update(unigrams)
        bigrams_all.update(bigrams)

    inter_dist1 = (len(unigrams_all)+1e-12) / (sum(unigrams_all.values())+1e-5)
    inter_dist2 = (len(bigrams_all)+1e-12) / (sum(bigrams_all.values())+1e-5)
    intra_dist1 = np.average(intra_dist1)
    intra_dist2 = np.average(intra_dist2)
    return (intra_dist1, intra_dist2, inter_dist1, inter_dist2)


## word-level knowledge f1 score
def calc_knowledge_f1(hyps, knowledge_refs, knowledge_alls):
    """" Calculate knowledge f1 score """
    golden_total = 0.0
    pred_total = 0.0
    hit_total = 0.0
    for response, golden_kd, all_kd in zip(hyps, knowledge_refs, knowledge_alls):
        # response = "".join(response)
        golden_total += len(golden_kd)
        for kd in golden_kd:
            if is_knowledge_hit(response, kd):
                hit_total += 1
        for kd in all_kd:
            if is_knowledge_hit(response, kd):
                pred_total += 1
    p = hit_total / pred_total if pred_total > 0 else 0
    r = hit_total / golden_total if golden_total > 0 else 0
    f1 = 2 * p * r / (p + r) if p + r > 0 else 0
    return f1


def is_knowledge_hit(utterance, kg_obj, threshold=1.0):
    flag = False
    if kg_obj in utterance:
        flag = True
    else:
        # knowledge recall
        count = 0
        for t in utterance.split():
            if t in kg_obj.split():
                count += 1
        ### if all tokens in the utterance also are in the knowledge object
        if count == len(utterance.split()):
            flag = True
    return flag

def label_knowledge(utterance, kg_list, lower_case=True):
    gold_knowledge = []
    all_objs = set()
    for triple in kg_list:
        if len(triple) == 0:
            continue
        assert len(triple) == 3
        all_objs.add(triple[0].lower() if lower_case else triple[0])
        all_objs.add(triple[2].lower() if lower_case else triple[2])
    for obj in all_objs:
        if is_knowledge_hit(utterance, obj):
            gold_knowledge.append(obj)
    all_objs = list(all_objs)
    return all_objs, gold_knowledge


def load_data(fp, is_gold=False, lower_case=True):
    samples = []
    all_knowledges = []
    gold_knowledges = []
    if not is_gold:
        with open(fp, 'r', encoding='utf-8') as fr:
            for line in fr:
                sample = json.loads(line)
                response = sample["response"].lower() if lower_case else sample["response"]
                # resp = [tok for tok in response.split]   # token-level list
                resp = response
                samples.append(resp)
                if is_gold:
                    knowledge = sample["knowledge"]
                    all, gold = label_knowledge(response, knowledge, lower_case=lower_case)
                    all_knowledges.append(all)
                    gold_knowledges.append(gold)
    else:
        with open(fp, 'rb') as fr:
            for sample in pickle.load(fr):
                response = sample["response"].lower() if lower_case else sample["response"]
                # resp = [tok for tok in response]   # token-level list
                resp = response
                samples.append(resp)
                if is_gold:
                    knowledge = sample["knowledge"]
                    all, gold = label_knowledge(response, knowledge, lower_case=lower_case)
                    all_knowledges.append(all)
                    gold_knowledges.append(gold)
    if is_gold:
        assert len(samples) == len(all_knowledges)
        assert len(samples) == len(gold_knowledges)
        return (samples, all_knowledges, gold_knowledges)
    else:
        return samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_file", type=str)
    parser.add_argument("--gold_file", type=str)
    args = parser.parse_args()

    preds = load_data(args.eval_file)
    refs, all_knowledges, ref_knowlwedges = load_data(args.gold_file, is_gold=True)
    # assert len(preds) == len(refs)

    # calculate f1
    f1 = calc_f1(preds, refs)

    # calculate bleu
    bleu1, bleu2 = calc_bleu(preds, refs)

    # calculate distinct
    _, _, inter_dist1, inter_dist2 = calc_distinct(preds)

    # calculate knowledge-F1
    kg_f1 = calc_knowledge_f1(preds, ref_knowlwedges, all_knowledges)

    output_str = "F1: %.2f%%\n" % (f1 * 100)
    output_str += "BLEU1: %.3f\n" % bleu1
    output_str += "BLEU2: %.3f\n" % bleu2
    output_str += "DISTINCT1: %.3f\n" % inter_dist1
    output_str += "DISTINCT2: %.3f\n" % inter_dist2
    output_str += "Knowledge F1: %.2f%%" % (kg_f1 * 100)

    print(output_str)

    # calculate target success rate
    calc_succ(args.eval_file, args.gold_file)
    calc_avg_turns(args.eval_file, args.gold_file)
