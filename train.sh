#!/bin/bash

name=$1

python main_train.py --mode=predict_1to1 --experiment_name=$name
