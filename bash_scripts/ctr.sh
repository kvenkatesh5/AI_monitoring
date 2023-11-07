#!/bin/bash
python3 /home/kesavan.venkatesh/contrastive-ood/train_contrastive.py \
    --learning_rate 0.001 \
    --batch_size 128 \
    --max_epochs 100 \
    --positive_dataset "organamnist" \
    --method "SupCon" \
    --model "resnet18" \
    --projection "mlp" \
    --temp 0.07 \
    --pretrained --use-gpus "6,7"
