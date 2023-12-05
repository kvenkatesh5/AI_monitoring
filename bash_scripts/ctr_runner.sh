#!/bin/bash
python3 /home/kesavan.venkatesh/ai_monitoring/train.py \
    --dataset "MedMNIST-AbdominalCT" \
    --method "supervised-ctr" \
    --learning_rate 0.001 \
    --batch_size 128 \
    --max_epochs 100 \
    --positive_dataset "organamnist" \
    --base_model "resnet18" \
    --projection "mlp" \
    --temp 0.07 \
    --use-gpus "6,7"
