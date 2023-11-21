#!/bin/bash
python3 /home/kesavan.venkatesh/ai_monitoring/train.py \
    --dataset "MedMNIST-AbdominalCT" \
    --method "supervised-cnn" \
    --learning_rate 0.001 \
    --batch_size 128 \
    --max_epochs 100 \
    --positive_dataset "organamnist" \
    --pretrained "n" \
    --base_model "resnet18" \
    --use-gpus "0,1"
