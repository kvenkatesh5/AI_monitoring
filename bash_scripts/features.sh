#!/bin/bash

# Provide final model paths here
python3 get_features.py \
    --dataset "MedMNIST-AbdominalCT" \
    --positive_dataset "organamnist" \
    --autoencoder_path "./model_saves/conv-autoencoder_None_MedMNIST-AbdominalCT_dtfmdefault_lr0.001_bsz128_nep100_indistAxial_time1701767136.174378.pt" \
    --cnn_path "./model_saves/supervised-cnn_resnet18_MedMNIST-AbdominalCT_dtfmdefault_lr0.001_bsz128_nep100_indistAxial_time1701767128.9677894.pt" \
    --ctr_path "./model_saves/supervised-ctr_resnet18_MedMNIST-AbdominalCT_dtfmdefault_lr0.001_bsz128_nep100_indistAxial_time1701767133.7412488.pt" \
