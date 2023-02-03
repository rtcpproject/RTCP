# Reinforced Target-driven Conversational Promotion

This repo contains code and data for the paper: "Reinforced Target-driven Conversational Promotion".

## 1. Setup

Please install libraries/packages listed in the requirements.txt file. Make sure that you are using CUDA 11.6. Otherwise, some unexpected behaviors might happend.

## 2. Data Preprocessing

To preprocess and repurpose the DuRecDial 2.0 dataset for our task, please run:

```
sh scripts/preprocess.sh
```

## 3. Short-term Planning

To train our short-term planning model, please run:

```
sh scripts/planning/train_planning.sh
```

To produce the plan with the trained short-term planning model, please run the following command:

```
sh scripts/planning/test_planning.sh
```

To evaluate the trained short-term planning model on the next goal and topic prediction tasks, please run:

```
python evaluate.py 
```

## 4. Long-term Planning

To train our long-term planning model, you need to first train the reward model. The reward model take as inputs a sequence of dialogue actions and output whether the given sequence is smooth or not. To train the reward model, please run the following command:

```
sh scripts/rl/train_reward.sh
```

To train our short-term planning model, please following commands:

```
sh scripts/rl/pretrain_rl.sh
sh scripts/rl/train_rl_after.sh
```

## 5. Strategic Balancing

The strategic balancing method computes a weighted combination of two probability distributions, one from short-term planning and the other from long-term planning. Then based on the computed distribution, we will sample the next dialogue action. To perform strategic balancing, please run the following command.

```
sh scripts/balancing.sh
```

## 6. Action-guided Response Generation

Given a generated plan from the planning part, you could run the action-guided prefix tuning method with the following command.

```
sh scripts/train_gpt2_prompt_new.sh
```

To generate responses with the generation model, you could run the following command:

```
sh scripts/test_gpt2_prompt_new.sh
```

To evaluate the performance of the trained generation model, please run the following command:

```
python eval/eval_dialogue.py --eval_file ${eval_file} --gold_file caches/path/test.pkl
```



