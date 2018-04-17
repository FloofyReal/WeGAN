#!/bin/bash
# Copy local git repository into docker - run with pull

set CUDA_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0

name=$1

python main_train.py --mode=predict_1to1 --experiment_name=$name
