#!/bin/bash
DATASET=${1-"MedMNIST-AbdominalCT"}

python3 get_features.py \
    --dataset "MedMNIST-AbdominalCT" \
    --positive_dataset "organamnist" \
    --autoencoder_path "./saves/conv-autoencoder_*.pt" \
    --cnn_path "./saves/supervised-cnn*.pt" \
    --ctr_path "./saves/supervised-ctr*.pt" \
