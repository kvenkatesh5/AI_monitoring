#!/bin/bash
DATASET=${1-"MedMNIST-AbdominalCT"}

python3 train.py \
    --dataset $DATASET \
    --method "conv-autoencoder" \
    --learning_rate 0.001 \
    --batch_size 128 \
    --max_epochs 100 \
    --c_hid 16 \
    --latent_dim 100 \
     #--use-gpus "2,3"
