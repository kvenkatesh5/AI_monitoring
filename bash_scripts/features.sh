#!/bin/bash
python3 get_features.py \
    --autoencoder_path "./saves/autoencoder_lr0.001_bsz128_nep100_indistAxial_time1698727260.3633106.pt" \
    --cnn_path "./saves/resnet18_lr0.001_bsz128_nep100_indistAxial_time1698721794.5067384.pt" \
    --ctr_path "./saves/SupCon_resnet18_lr0.001_decay0.0001_bsz128_temp0.07_time1698727745.6544845.pt" \
