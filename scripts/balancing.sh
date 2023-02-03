#!/bin/bash
short_path="preds/local_pred.pkl"
long_path="preds/rl_pred.pkl"
alpha=0.6

python balancing.py --short_path ${short_path} --long_path ${long_path} --alpha ${alpha}