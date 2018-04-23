#!/bin/bash

name=$1
channels=$1
wvars=$1

python main_train.py --mode=predict_1to1 --experiment_name=$name --channels=3 --wvars='11100'
