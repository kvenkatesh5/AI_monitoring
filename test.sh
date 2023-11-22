#!/bin/bash
MAX_EPOCHS=10

echo 'Model training [1/3] Now running: bash_scripts/autoencoder_runner.sh...'
python3 train.py \
    --method "conv-autoencoder" \
    --max_epochs $MAX_EPOCHS \

echo 'Model training [2/3] Now running: bash_scripts/cnn.sh...'
python3  train.py \
    --method "supervised-cnn" \
    --max_epochs $MAX_EPOCHS \

echo 'Model training [3/3] Now running: bash bash_scripts/ctr.sh...'
python3  train.py \
    --method "supervised-ctr" \
    --max_epochs $MAX_EPOCHS \
    --positive_dataset "organamnist" \

echo 'Generating features'
python3 features.py \
    --autoencoder_path "./saves/autoencoder_lr0.001_bsz128_nep${MAX_EPOCHS}_indistAxial_time1698727260.3633106.pt" \
    --cnn_path "./saves/resnet18_lr0.001_bsz128_nep${MAX_EPOCHS}_indistAxial_time1698721794.5067384.pt" \
    --ctr_path "./saves/SupCon_resnet18_lr0.001_decay0.0001_bsz128_temp0.07_time1698727745.6544845.pt" \
