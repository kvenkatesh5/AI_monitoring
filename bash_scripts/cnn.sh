#!/bin/bash
python3 /home/kesavan.venkatesh/contrastive-ood/train_cnn.py \
    --learning_rate 0.001 \
    --batch_size 128 \
    --max_epochs 100 \
    --positive_dataset "organamnist" \
    --pretrained "n" --use-gpus "0,1"
