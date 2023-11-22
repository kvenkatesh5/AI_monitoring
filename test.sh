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
    --autoencoder_path $(ls saves/conv-autoencoder_lr0.001_bsz128_nep${MAX_EPOCHS}* -t1 | head -n 1) \
    --cnn_path $(ls saves/supervised-cnn_lr0.001_bsz128_nep${MAX_EPOCHS}_indistAxial_time* -t1 | head -n 1) \
    --ctr_path $(ls saves/supervised-ctr_lr0.001_bsz128_nep* -t1 | head -n 1) \
#grabs most recent saved model