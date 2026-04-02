#!/bin/bash

# Q1 - Baseline DNN (mean pooling)
python main.py --no_show --model baseline_mean --dataset MR

# Q2 - Baseline DNN (mean+max pooling)
python main.py --no_show --model baseline_mean_max --dataset MR

# Q3 - Bidirectional LSTM
python main.py --no_show --model lstm --dataset MR

# Q4 - Simple Self-Attention and Multi-Head Attention
python main.py --no_show --model attention --dataset MR

python main.py --no_show --model multihead --dataset MR --n_head 4
python main.py --no_show --model multihead --dataset MR --n_head 8

# Q5 - Transformer Encoder (experimenting with n_head and n_layer)
# Classic Transformer defaults: n_head=8, n_layer=6
python main.py --no_show --model transformer --dataset MR --n_head 4 --n_layer 3
python main.py --no_show --model transformer --dataset MR --n_head 8 --n_layer 6
python main.py --no_show --model transformer --dataset MR --n_head 4 --n_layer 6
