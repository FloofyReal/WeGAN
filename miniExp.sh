#!/bin/bash

python main_train.py --mode=predict_1to1 --experiment_name=train_allwvars_32x32_10years_200epochs_lr-0001 --channels=5 --wvars='11111' --dataset='32x32' --num_epochs=250 --learning_rate=0.0001
python main_sample.py --mode=predict_1to1 --experiment_name=train_allwvars_32x32_10years_200epochs_lr-0001 --channels=5 --wvars='11111' --dataset='32x32'
