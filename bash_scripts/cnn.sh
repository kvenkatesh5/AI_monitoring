#!/bin/bash
DATASET=${1-"MedMNIST-AbdominalCT"}

python3  train.py \
    --dataset $DATASET \
    --method "supervised-cnn" \
    --learning_rate 0.001 \
    --batch_size 128 \
    --max_epochs 100 \
    --positive_dataset "organamnist" \
    --pretrained "n" \
    --base_model "resnet18" \
     #--use-gpus "0,1"
